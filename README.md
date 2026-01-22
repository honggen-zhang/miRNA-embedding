# miRNA-mRNA Interaction Prediction

A deep learning framework for predicting microRNA (miRNA) and messenger RNA (mRNA) interactions using contrastive learning and cross-attention mechanisms.

## Overview

This project implements a neural network model that learns to predict binding interactions between miRNA and mRNA sequences. The model uses:

- **T5-based Transformer encoders** for both miRNA and mRNA sequence encoding
- **Cross-attention mechanism** to model the interaction between miRNA and mRNA
- **Contrastive learning** with InfoNCE loss to learn meaningful sequence representations
- **Exponential Moving Average (EMA)** for stable training as a teacher model
- **Distributed Data Parallel (DDP)** training for multi-GPU support

## Project Structure

```
miRNA/
├── train_simple.py          # Main training script with DDP support
├── model_cmm.py               # Model architectures (miRNAModel, mRNA_encoder, contrastive_mRNA)
├── DataLoad_simple.py         # Dataset classes for loading RNA sequences
├── EMA.py                     # Exponential Moving Average implementation
├── attention_score_binding_site.ipynb  # Jupyter notebook for visualization and analysis
├── tokenizer_3mers/           # Tokenizer files for 3-mer encoding
│   └── tokenizer.json
├── data/                      # Training and validation data
│   ├── utr3_train_large.json
│   ├── piRNA_validate_utr5_breast.json
│   └── ...
└── checkpoint/                # Model checkpoints
    └── model_mm.pt
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU training)

### Install Dependencies

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib tqdm wandb
```

## Data Format

Training data should be in JSON format with the following structure:

| Column | Description |
|--------|-------------|
| 0 | miRNA sequence |
| 1 | mRNA sequence (full length) |
| 2 | Target site sequence |
| 3 | Binding site start position |
| 4 | Binding site end position |
| 5 | Label (1 = interacting, 0 = non-interacting) |

**Note:** DNA sequences are automatically converted to RNA (T → U) during preprocessing.

## Usage

### Training

#### Single GPU Training

```bash
python train_simple.py --data_path your_data
```

#### Multi-GPU Training (Distributed)

```bash
torchrun --nproc_per_node=NUM_GPUS train_simple.py --data_path your_data
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | `utr3_train_large.json` | Name of training data file (without .json) |

### Configuration

Key hyperparameters can be modified in `train_simple-4.py`:

```python
BATCH_SIZE = 64          # Batch size per GPU
EPOCHS = 10              # Number of training epochs
LEARNING_RATE = 1e-6     # Learning rate
WEIGHT_DECAY = 1e-5      # Weight decay for AdamW
```

### Monitoring Training

Training metrics are logged to [Weights & Biases](https://wandb.ai/). Tracked metrics include:

- Training loss (total, binary, MSE/contrastive)
- Training accuracy (short-range and long-range)
- Validation accuracy

## Inference & Visualization

Use the Jupyter notebook `attention_score_binding_site.ipynb` to:

1. **Load a trained model** and run inference
2. **Visualize attention weights** between miRNA and mRNA positions
3. **Identify binding sites** based on attention scores
4. **Analyze predicted interactions** with sequence-level detail

### Example: Extracting Attention Weights

```python
# Load model
model = contrastive_mRNA(encoder_mi, encoder_m)
ckpt = torch.load('checkpoint/model_mm.pt', map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=True)

# Run inference
attention_weight, clf_result = model.forward_eval(miRNA, miRNA_attention, mRNA, mRNA_attention)

# Visualize
plt.matshow(attention_weight[0].cpu().numpy(), cmap='viridis')
plt.xlabel('mRNA Position')
plt.ylabel('miRNA Position')
plt.colorbar()
```

## Outputs

### Checkpoints

- `checkpoint/model_mm.pt` - Best model based on validation accuracy
- `checkpoint/model_mm_last.pt` - Final model after training

Each checkpoint contains:
- `state_dict`: Full model weights
- `ema`: EMA model weights
- `m_encoder`: mRNA encoder weights


## Key Features

### Curriculum Learning

The training uses a curriculum learning strategy:
- **Epochs 0-5**: Binary classification loss only
- **Epochs 6+**: Adds contrastive (InfoNCE) loss for improved representations

### Attention-Guided Binding Site Prediction

The cross-attention weights can be used to identify potential binding sites:
- High attention scores indicate likely interaction regions
- KL divergence loss guides attention to known binding sites during training

## Citation

If you use this code in your research, please cite:

```bibtex
@artical{title = {Contrastive Sequence Modeling Advances microRNA-mRNA Target Site Prediction},
  author = {Gao, Zitong; Zhang, Honggen; Zhang, Hanqiu ; Ma, Li; Feng, Zhuokun ; Zhang, June ; Deng, Youping; Wu, Lang},
  year = {2025},
  url = {https://sncrna-func.jabsom.hawaii.edu/}
}
```

## License
MIT




