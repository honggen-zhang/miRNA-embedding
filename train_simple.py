import pandas as pd
import torch.nn.functional as F
import transformers
import argparse
import tqdm
import matplotlib.pyplot as plt
from DataLoad_simple import RNA_Seq
from model_cmm import contrastive_mRNA,miRNAModel,mRNA_encoder
import os
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.metrics import confusion_matrix,roc_curve, auc
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'#1#,1,2,3'#,4,5,6,7'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'#,4,5,6,7'
os.environ['OMP_NUM_THREADS'] = '1'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

local_rank = int(os.environ.get('LOCAL_RANK', 0))
setup_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,default='data1.csv')
    #parser.add_argument("--cuda_device", type=str, default='0')
    args = parser.parse_args()
    print(args)
    if local_rank == 0:
        wandb.init(project="mmRNA", dir = './', name = 'train',)
    # Set CUDA device
    #python trainer.py --data_path "test_small_data" --cuda_device "3"
    #os.environ['OMP_NUM_THREADS'] = '1'
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = 'cuda'
    #torch.cuda.set_device(int(args.cuda_device))
    data_path = args.data_path
    DATA_PATH = f'./data/{data_path}.json'
    TOKENIZER_PATH = "./tokenizer_mixmers/"
    TOKENIZER_PATH_1mer = "./tokenizer_mixmers/"
    BATCH_SIZE = 128
    EPOCHS = 4
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5

    encoder_mi = miRNAModel(num_attention_heads=4, num_hidden_layers=4, pad_token_id=3, hidden_size=128).cuda()
    encoder_m = mRNA_encoder(num_attention_heads=4, num_hidden_layers=4, pad_token_id=3, hidden_size=128).cuda()


    model = contrastive_mRNA(encoder_mi,encoder_m).cuda()
    
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    #ckpt = torch.load(MODEL_PATH, map_location='cpu')
    #if args.load_model == 'True':
        #print('loading model')
        #encoder_state_dict = {k[8:]: v for k, v in ckpt['encoder'].items() if k.startswith('encoder.')}
        #model.encoder.load_state_dict(encoder_state_dict, strict=True)
    scaler = torch.cuda.amp.GradScaler()


    train_db = RNA_Seq(data_path=DATA_PATH, tokenizer_path=TOKENIZER_PATH,tokenizer_1mer = TOKENIZER_PATH_1mer, max_length=1024,split = 'train')
    #valid_db = RNA_Seq(data_path='./data/eval_data_test.json', tokenizer_path=TOKENIZER_PATH,tokenizer_1mer = TOKENIZER_PATH_1mer, max_length=1024,split = 'test')
    valid_db = RNA_Seq(data_path='./data/cds_test.json', tokenizer_path=TOKENIZER_PATH,tokenizer_1mer = TOKENIZER_PATH_1mer, max_length=1024,split = 'test')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_db, shuffle=True, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_db,
                                               batch_size=BATCH_SIZE,
                                               num_workers=1,
                                               drop_last=False,
                                               sampler=train_sampler,
                                               pin_memory=False,
                                               )

    val_loader = torch.utils.data.DataLoader(valid_db,
                                             batch_size=BATCH_SIZE,
                                             num_workers=1,
                                             drop_last=False,
                                             shuffle=True,
                                             pin_memory=False, )


    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_decay = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                             num_warmup_steps = 1000,
                                             num_training_steps = len(train_loader)*EPOCHS,
                                             )
    criterion_cross = nn.CrossEntropyLoss()
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    best_acc = 0.
    valid_loss_list = []
    kk = 0
    for e in range(EPOCHS):
        train_sampler.set_epoch(e)
        if local_rank == 0:
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0)
        else:
            loop = enumerate(train_loader)
        
        loss_list = []
 
        for no, batch in loop:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                miRNA,miRNA_attention,target_site, target_site_attention, mRNA, mRNA_attention,labels = [x.to(device) for x in batch]
                x_emd,y_emd, logits,logits_long= model(
                    miRNA,
                    miRNA_attention,
                    target_site, 
                    target_site_attention,
                    mRNA,
                    mRNA_attention,
                )

                labels = (labels).view(-1,1).float()
                #probabilities = torch.sigmoid(logits)
                loss_binary = criterion_binary(logits, labels)
                
                #probabilities = torch.sigmoid(logits)
                #probabilities_neg = torch.sigmoid(neg_logits)
                #print(probabilities.shape,probabilities_neg.shape)
                #print(labels * torch.log(probabilities))
                #loss_binary = -torch.log(probabilities+0.0001).mean() - (torch.log(1 -probabilities_neg+0.0001).mean())

                loss_mse = criterion_mse(x_emd, y_emd)
                #sim = torch.matmul(x_emd,y_emd.T)
                #sim_label = torch.zeros(sim.size(0), dtype=torch.long).to(device)
                #loss_mse = criterion_cross(sim, sim_label)
                if e>1:
                    loss =loss_mse+ loss_binary
                else:
                    loss =loss_binary
                scaler.scale(loss).backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                lr_decay.step()
                model.module.ema_step()
                if local_rank == 0:
                    #print('loss',loss)
                    kk = kk+1
                    if kk%50==0:
                    #print(x_emd.shape)
                        probabilities = torch.sigmoid(logits)
                        print(x_emd[0])
                        print(y_emd[0])
                        # print(probabilities.view(1,-1))
                        predicted_labels = (probabilities >= 0.50).float()  # Convert to 0 or 1
                        correct_predictions = (predicted_labels == labels).float()  # 1 if correct, 0 if incorrect
                        accuracy = correct_predictions.mean()  # Mean of correct predictions
                        ######################
                        probabilities_long = torch.sigmoid(logits_long)
                        predicted_labels_long = (probabilities_long >= 0.50).float()  # Convert to 0 or 1
                        correct_predictions_long = (predicted_labels_long == labels).float()  # 1 if correct, 0 if incorrect
                        accuracy_long = correct_predictions_long.mean() 
                        # tn, fp, fn, tp = confusion_matrix(labels.view(-1).detach().cpu().numpy(), predicted_labels.view(-1).detach().cpu().numpy()).ravel()
                        # tpr_ = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
                        # tnr_ = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate

                        # # Convert logits to probabilities using sigmoid
                        # probabilities =probabilities.view(-1).detach().cpu().numpy()  # Convert to 1D numpy array for sklearn

                        # # Use sklearn to compute FPR, TPR, and thresholds
                        # fpr, tpr, thresholds = roc_curve(labels.view(-1).detach().cpu().numpy(), probabilities)

                        # print(f"Prediction accuracy: {accuracy.item() * 100:.2f}%")
                        print(f"step {kk} train loss: {loss.item() * 100:.2f}%")
                        print(f"step {kk} mse loss: {loss_mse.item() * 100:.2f}%")
                        print(f"step {kk} binary loss: {loss_binary.item() * 100:.2f}%")
                
                        loop.set_postfix(Epoch=e,
                                         loss=loss.item(),
                                         )
                        loss_list.append(loss.item())
                        current_lr = optimizer.param_groups[0]['lr']
                        print('lr',current_lr)
                        #print(probabilities_neg)
                        wandb.log({"train loss": loss.item(),
                                   'mse loss':loss_mse.item(),
                                   'binary loss':loss_binary.item(),
                                   'accuracy':accuracy.item(),
                                   'accuracy_long':accuracy_long.item(),
                                   #'tnr':tnr_,
                                   #'tpr':tpr_,
                                   'lr':current_lr,})
    
                
                
                if local_rank == 0:
                    if kk%100==0 and e>=0:
                        model.eval()
                        predict_loss_list = []
                        acc_test_list = []
                        test_num = 0
                        for no, batch_test in enumerate(val_loader):
                            miRNA,miRNA_attention,target_site, target_site_attention, mRNA,mRNA_attention,labels_test = [x.to(device) for x in batch_test]

                    
                            with torch.no_grad():
                                x_emd,y_emd, logits,logits_long= model(
                                    miRNA,
                                    miRNA_attention,
                                    target_site, 
                                    target_site_attention,
                                    mRNA,
                                    mRNA_attention,
                                    #target_site,
                                    #target_site_attention,
                                )
                                labels_test = (labels_test).view(-1,1).float()
                                probabilities = torch.sigmoid(logits_long)
                                print(probabilities.view(1,-1))
                                print(labels_test.view(1,-1))
                                predicted_labels = (probabilities >= 0.5).float()  # Convert to 0 or 1
                                correct_predictions = (predicted_labels == labels_test).float()  # 1 if correct, 0 if incorrect
                                accuracy = correct_predictions.mean()  # Mean of correct predictions
                                acc_test_list.append(accuracy.item())
                                fpr, tpr, thresholds = roc_curve(labels_test.view(-1).detach().cpu().numpy(), probabilities.view(-1).detach().cpu().numpy())
                                test_num = test_num+1
                                if test_num>2:
                                    break
    
                        #print(f"Prediction test accuracy: {accuracy.item() * 100:.2f}%")
                        
                        wandb.log({"test accuracy": np.mean(acc_test_list),})
                        if best_acc < np.mean(acc_test_list):
                            best_acc = np.mean(acc_test_list)

                            model.module.ema.model.eval()
                            
                            checkpoint = {'state_dict': model.module.state_dict(),'ema':model.module.ema.model.state_dict(),'m_encoder': model.module.encoder_mRNA.state_dict()}
                            torch.save(checkpoint, './checkpoint/model_mm_mixmer_best.pt')
                            
                            

                    model.train()
    

    if local_rank == 0:
        model.module.ema.model.eval()
        checkpoint = {'state_dict': model.module.state_dict(),
                      'ema':model.module.ema.model.state_dict(),
                      'm_encoder': model.module.encoder_mRNA.state_dict(),
                     }
        torch.save(checkpoint, './checkpoint/model_mm_mixmer2.pt')
        plt.figure()
        roc_auc = auc(fpr, tpr)
    
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Dashed diagonal line for random classifier
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')

if __name__ == '__main__':
    main()