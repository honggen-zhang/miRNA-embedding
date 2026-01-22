"""
miRNA-mRNA Interaction Prediction Training Script

This script trains a contrastive learning model to predict miRNA-mRNA interactions.
It uses distributed data parallel (DDP) training with mixed precision (AMP) for
efficient multi-GPU training.

The model learns to encode miRNA and mRNA sequences into a shared embedding space
where interacting pairs are close together.

Usage:
    torchrun --nproc_per_node=NUM_GPUS train_simple.py --data_path dataset

"""

# =============================================================================
# Imports
# =============================================================================

# Standard library imports
import argparse
import os
import random

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
import wandb
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.utils import data

# Local imports
from DataLoad_simple import RNA_Seq
from model_cmm import contrastive_mRNA, contrastive_mRNA2, miRNAModel, mRNA_encoder

# =============================================================================
# Environment Configuration
# =============================================================================

# Set visible GPU devices (modify as needed for multi-GPU training)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Uncomment for multi-GPU

# Limit OpenMP threads to prevent CPU oversubscription
os.environ['OMP_NUM_THREADS'] = '1'


# =============================================================================
# Utility Functions
# =============================================================================

def setup_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The random seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def info_nce_loss(A: torch.Tensor, B: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute the InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
    
    This loss encourages matching pairs (A[i], B[i]) to have high similarity
    while pushing non-matching pairs apart.
    
    Args:
        A: First set of embeddings with shape (N, d).
        B: Second set of embeddings with shape (N, d).
        temperature: Temperature parameter for scaling (not currently used in logits).
    
    Returns:
        The InfoNCE loss value.
    """
    # Normalize embeddings to unit vectors for cosine similarity
    A = F.normalize(A, p=2, dim=1)  # (N, d)
    B = F.normalize(B, p=2, dim=1)  # (N, d)
    
    # Compute similarity matrix: logits[i,j] = similarity between A[i] and B[j]
    logits = torch.matmul(A, B.T)  # (N, N)
    
    # Positive pairs are on the diagonal (A[i] should match B[i])
    labels = torch.arange(A.size(0)).to(A.device)
    
    # Cross-entropy loss treats this as N-way classification
    criterion = nn.CrossEntropyLoss()
    loss_nce = criterion(logits, labels)
    
    return loss_nce


def calculate_metric(clf_result_neg: torch.Tensor) -> float:
    """
    Calculate top-10 hit accuracy for the classification results.
    
    For each sample, checks if the true positive (diagonal element) is among
    the top-10 predictions.
    
    Args:
        clf_result_neg: Classification logits with shape (batch_size, batch_size, 1).
    
    Returns:
        Top-10 accuracy as a float between 0 and 1.
    """
    clf_result_neg = clf_result_neg.squeeze(-1)  # (batch_size, batch_size)
    batch_size, _ = clf_result_neg.shape

    # Ground truth: positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=clf_result_neg.device)

    # Convert logits to probabilities
    probabilities = torch.softmax(clf_result_neg, dim=1)

    # Get top-10 predictions for each row
    top10_indices = torch.topk(probabilities, k=10, dim=1).indices  # (batch_size, 10)

    # Check if true label is in top-10 predictions
    correct_top10 = top10_indices.eq(labels.unsqueeze(1)).any(dim=1)  # (batch_size,)

    # Compute hit accuracy
    top10_accuracy = correct_top10.float().mean().item()

    return top10_accuracy


def calculate_accuracy(clf_result_neg: torch.Tensor, threshold: float = 0.5) -> tuple:
    """
    Calculate accuracy, true positive rate (TPR), and true negative rate (TNR).
    
    Args:
        clf_result_neg: Classification logits with shape (batch_size, batch_size, 1).
        threshold: Probability threshold for positive prediction.
    
    Returns:
        Tuple of (accuracy, TPR, TNR).
    """
    clf_result_neg = clf_result_neg.squeeze(-1)  # (batch_size, batch_size)
    batch_size, _ = clf_result_neg.shape

    # Ground truth labels: positive pairs are on the diagonal
    labels = torch.eye(batch_size, device=clf_result_neg.device).bool()
    probabilities = torch.softmax(clf_result_neg, dim=1)

    # Apply threshold to get binary predictions
    predictions = probabilities > threshold
    print(probabilities)

    # Calculate confusion matrix components
    true_positives = (predictions & labels).sum().item()
    false_negatives = (~predictions & labels).sum().item()
    true_negatives = (~predictions & ~labels).sum().item()
    false_positives = (predictions & ~labels).sum().item()

    # Calculate metrics
    total = batch_size * batch_size
    accuracy = (true_positives + true_negatives) / total

    # True Positive Rate (Sensitivity/Recall)
    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # True Negative Rate (Specificity)
    tnr = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0

    return accuracy, tpr, tnr


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    """
    Main training loop for the miRNA-mRNA interaction prediction model.
    
    This function:
    1. Initializes distributed training environment
    2. Loads pretrained model checkpoint
    3. Sets up data loaders with distributed sampling
    4. Runs the training loop with mixed precision
    5. Evaluates on validation set and saves best checkpoints
    """
    # -------------------------------------------------------------------------
    # Argument Parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train miRNA-mRNA interaction model")
    parser.add_argument(
        "--data_path",
        type=str,
        default='utr3_train_large.json',
        help="Name of the training data file (without .json extension)"
    )
    args = parser.parse_args()
    print(args)

    # -------------------------------------------------------------------------
    # Distributed Training Setup
    # -------------------------------------------------------------------------
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    setup_seed(42)
    
    # Initialize Weights & Biases logging (only on main process)
    if local_rank == 0:
        wandb.init(project="miRNA_cross-pi", dir='./', name='demo130k')
    
    # Initialize distributed process group
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = 'cuda'

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    DATA_PATH = f'./data/{args.data_path}.json'
    TOKENIZER_PATH = "./tokenizer_3mers/"
    TOKENIZER_PATH_1MER = "./tokenizer_3mers/"
    
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-5

    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------
    # Initialize encoder models
    encoder_mi = miRNAModel(
        num_attention_heads=2,
        num_hidden_layers=4,
        pad_token_id=3,
        hidden_size=128
    ).cuda()
    
    encoder_m = mRNA_encoder(
        num_attention_heads=2,
        num_hidden_layers=4,
        pad_token_id=3,
        hidden_size=128
    ).cuda()

    # Create contrastive learning model
    model = contrastive_mRNA(encoder_mi, encoder_m).cuda()
    
    # Load pretrained checkpoint
    #MODEL_PATH = './checkpoint/model_mm_3mer_130klong_4_wo_kl.pt'
    #ckpt = torch.load(MODEL_PATH, map_location='cpu')
    #model.load_state_dict(ckpt['state_dict'], strict=True)
    
    # Wrap model with DistributedDataParallel
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    # Training dataset
    train_db = RNA_Seq(
        data_path=DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        tokenizer_1mer=TOKENIZER_PATH_1MER,
        max_length=700,
        split='train'
    )
    
    # Validation dataset
    valid_db = RNA_Seq(
        data_path='./data/piRNA_validate_utr5_breast.json',
        tokenizer_path=TOKENIZER_PATH,
        tokenizer_1mer=TOKENIZER_PATH_1MER,
        max_length=700,
        split='test'
    )
    
    # Distributed sampler for training data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_db,
        shuffle=True,
        drop_last=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=BATCH_SIZE,
        num_workers=1,
        drop_last=False,
        sampler=train_sampler,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_db,
        batch_size=BATCH_SIZE,
        num_workers=1,
        drop_last=False,
        shuffle=True,
        pin_memory=False,
    )

    # -------------------------------------------------------------------------
    # Optimizer and Scheduler
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Cosine learning rate schedule with warmup
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_loader) * 5,
        num_training_steps=len(train_loader) * EPOCHS,
    )

    # -------------------------------------------------------------------------
    # Loss Functions
    # -------------------------------------------------------------------------
    criterion_cross = nn.CrossEntropyLoss()
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    best_acc = 0.0
    valid_loss_list = []
    global_step = 0
    
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch
        
        # Progress bar only on main process
        if local_rank == 0:
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0)
        else:
            loop = enumerate(train_loader)
        
        loss_list = []
 
        for batch_idx, batch in loop:
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                # Unpack batch data
                miRNA, miRNA_attention, target_site, target_site_attention, \
                    mRNA, mRNA_attention, labels = [x.to(device) for x in batch[:-1]]
                start_position = batch[-1]
                
                # Forward pass through model
                x_emd, y_emd, logits, logits_long, kl_loss = model(
                    miRNA,
                    miRNA_attention,
                    target_site,
                    target_site_attention,
                    mRNA,
                    mRNA_attention,
                    start_position,
                )

                # Prepare labels for binary classification
                labels = labels.view(-1, 1).float()
                bs = logits.size(0)
                
                # Compute losses
                loss_binary = criterion_binary(logits, labels)
                loss_binary_long = criterion_binary(logits_long, labels)
                loss_mse = info_nce_loss(x_emd, y_emd)
                
                # Combined loss (curriculum learning: add contrastive loss after epoch 5)
                if epoch > 5:
                    loss = 1 * loss_mse + loss_binary + 1 * loss_binary_long + 0.0 * kl_loss
                else:
                    loss = loss_binary + 0.0 * kl_loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                
                # Update EMA (Exponential Moving Average) model
                model.module.ema_step()
                
                # -------------------------------------------------------------
                # Logging (main process only)
                # -------------------------------------------------------------
                if local_rank == 0:
                    global_step += 1
                    
                    if global_step % 100 == 0:
                        # Calculate training accuracy for short-range predictions
                        probabilities = torch.sigmoid(logits)
                        predicted_labels = (probabilities >= 0.50).float()
                        correct_predictions = (predicted_labels == labels).float()
                        accuracy = correct_predictions.mean()
                        
                        # Calculate training accuracy for long-range predictions
                        probabilities_long = torch.sigmoid(logits_long)
                        predicted_labels_long = (probabilities_long >= 0.50).float()
                        correct_predictions_long = (predicted_labels_long == labels).float()
                        accuracy_long = correct_predictions_long.mean()
                
                        # Update progress bar
                        loop.set_postfix(Epoch=epoch, loss=loss.item())
                        loss_list.append(loss.item())
                        
                        # Log metrics to wandb
                        wandb.log({
                            "train/train loss": loss.item(),
                            'train/mse loss': loss_mse.item(),
                            'train/binary loss': loss_binary.item(),
                            'train/long binary loss': loss_binary_long.item(),
                            'train/accuracy': accuracy,
                            'train/accuracy_long': accuracy_long,
                            'metric/kl_loss': kl_loss.item(),
                        })
    
                # -------------------------------------------------------------
                # Validation (main process only)
                # -------------------------------------------------------------
                if local_rank == 0:
                    if global_step % 200 == 0 and epoch >= 0:
                        model.eval()
                        acc_test_list = []
                        test_num = 0
                        
                        for val_batch_idx, batch_test in enumerate(val_loader):
                            # Note: Using training batch for validation (potential bug in original)
                            miRNA, miRNA_attention, target_site, target_site_attention, \
                                mRNA, mRNA_attention, labels_test = [x.to(device) for x in batch[:-1]]
                            start_position = batch[-1]

                            with torch.no_grad():
                                x_emd, y_emd, logits, logits_long, kl_loss = model(
                                    miRNA,
                                    miRNA_attention,
                                    target_site,
                                    target_site_attention,
                                    mRNA,
                                    mRNA_attention,
                                    start_position,
                                )
                                
                                labels_test = labels_test.view(-1, 1).float()
                                probabilities = torch.sigmoid(logits_long)
                                predicted_labels = (probabilities >= 0.5).float()
                                correct_predictions = (predicted_labels == labels_test).float()
                                accuracy = correct_predictions.mean()
                                acc_test_list.append(accuracy.item())
                                
                                # Compute ROC curve
                                fpr, tpr, thresholds = roc_curve(
                                    labels_test.view(-1).detach().cpu().numpy(),
                                    probabilities.view(-1).detach().cpu().numpy()
                                )
                                
                                test_num += 1
                                if test_num > 1:
                                    break
    
                        # Log validation accuracy
                        wandb.log({"test/accuracy": np.mean(acc_test_list)})
                        
                        # Save best model checkpoint
                        if best_acc < np.mean(acc_test_list):
                            best_acc = np.mean(acc_test_list)
                            model.module.ema.model.eval()
                            
                            checkpoint = {
                                'state_dict': model.module.state_dict(),
                                'ema': model.module.ema.model.state_dict(),
                                'm_encoder': model.module.encoder_mRNA.state_dict()
                            }
                            torch.save(checkpoint, './checkpoint/model_mm.pt')

                        model.train()

    # -------------------------------------------------------------------------
    # Final Checkpoint and Visualization
    # -------------------------------------------------------------------------
    if local_rank == 0:
        # Save final model checkpoint
        model.module.ema.model.eval()
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'ema': model.module.ema.model.state_dict(),
            'm_encoder': model.module.encoder_mRNA.state_dict(),
        }
        torch.save(checkpoint, './checkpoint/model_mm_last.pt')
        
        # Plot and save ROC curve
        plt.figure()
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier baseline
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()
