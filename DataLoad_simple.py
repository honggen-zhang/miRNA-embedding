import os
import json
import torch
import pandas as pd
import torch.nn as nn
from torch.utils import data
from transformers import T5EncoderModel, T5Config
from transformers import PreTrainedTokenizerFast,PreTrainedTokenizer
import numpy as np
import random

def split_into_one_mers(sequence):
    one_mers = [sequence[i:i+3] for i in range(0, len(sequence), 1)]
    sequence_preprocessed = ' '.join(one_mers)
    return sequence_preprocessed
    
def split_into_six_mers(sequence):
    six_mers = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    sequence_preprocessed = ' '.join(six_mers)
    return sequence_preprocessed
class RNA_Seq(data.Dataset):
    def __init__(self, data_path=str,
                 tokenizer_path=str,
                 tokenizer_1mer=str,
                 max_length=32,
                 split = 'train',
                 ):
        self.max_length = max_length
        rna_data = pd.read_json(data_path)
        #if mode == 'train':
        #te_data = te_data[te_data['Sequence'].str.len() <= 1024]
        mirna_seqs = rna_data.iloc[1:, 0].values.tolist()
        dna_seqs = rna_data.iloc[1:, 1].values.tolist()
        target_seqs = rna_data.iloc[1:, 2].values.tolist()
        self.starts = rna_data.iloc[1:, 3].values.tolist()
        self.ends = rna_data.iloc[1:, 4].values.tolist()
        self.target = rna_data.iloc[1:, 5].values.tolist()
        self.split = split
        
        
        mirna_seqs = [x.replace('T', 'U') for x in mirna_seqs]
        dna_seqs = [x.replace('T', 'U') for x in dna_seqs]
        target_seqs = [x.replace('T', 'U') for x in target_seqs]

        #states = [x if x=='test' else 'train' for x in states ]
        self.data = [[x, y1,y2] for x, y1,y2 in zip(mirna_seqs, dna_seqs,target_seqs)]
        print(len(self.data))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"),
                                                 #model_max_length=10,
                                                 unk_token="[UNK]",
                                                 cls_token="[CLS]",
                                                 sep_token="[SEP]",
                                                 pad_token="[PAD]",
                                                 mask_token="[MASK]",
                                                 #padding_side = 'left',
                                                )
        #self.mask_token_id = self.tokenizer.mask_token_id
        #self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"
        self.mask_token_id = self.tokenizer.mask_token_id

        self.tokenizer_1mer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_1mer, "tokenizer.json"),
                                                 #model_max_length=10,
                                                 unk_token="[UNK]",
                                                 cls_token="[CLS]",
                                                 sep_token="[SEP]",
                                                 pad_token="[PAD]",
                                                 mask_token="[MASK]",
                                                 #padding_side = 'left',
                                                )
        #self.mask_token_id = self.tokenizer.mask_token_id
        #self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer_1mer.padding_side = "right"
        #self.mask_token_id = self.tokenizer.mask_token_id

    def __getitem__(self, index):
        rna_, dna_, target_site_ = self.data[index]
        starts_one = int((self.starts[index]))+1
        ends_one = int((self.ends[index]))+1
        target = int(self.target[index])
        #label = np.array(label, dtype=np.float32)
        rna_ = split_into_six_mers(rna_)
        self.tokenizer_1mer.model_max_length = 25#14
        rna = self.tokenizer_1mer(rna_,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors='pt',)
        #print('target',target_site)
        #print('dna_----',dna_[self.starts[index]:self.ends[index]])
        target_site_ = split_into_one_mers(target_site_)
        #self.tokenizer.model_max_length = 6
        target_site = self.tokenizer_1mer(target_site_,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors='pt',)
        

        dna_ = split_into_one_mers(dna_)
        self.tokenizer.model_max_length = self.max_length
        dna = self.tokenizer(dna_,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors='pt',)

        
        mirna_input = rna.input_ids[0].numpy()
        mirna_attention = rna.attention_mask[0].numpy()

        dna_input = dna.input_ids[0].numpy()
        dna_attention = dna.attention_mask[0].numpy()

        target_site_input = target_site.input_ids[0].numpy()
        target_site_attention = target_site.attention_mask[0].numpy()

        
        
        return mirna_input, mirna_attention,target_site_input,target_site_attention, dna_input,dna_attention,target,starts_one
        
    def __len__(self):
        return len(self.data)

class RNA_Seq_eval(data.Dataset):
    def __init__(self, data_path=str,
                 tokenizer_path=str,
                 tokenizer_1mer=str,
                 max_length=32,
                 ):
        self.max_length = max_length
        rna_data = pd.read_json(data_path)
        #if mode == 'train':
        #te_data = te_data[te_data['Sequence'].str.len() <= 1024]
        mirna_seqs = rna_data.iloc[:, 0].values.tolist()
        dna_seqs = rna_data.iloc[:, 1].values.tolist()
        target_seqs = rna_data.iloc[:, 2].values.tolist()
        self.starts = rna_data.iloc[:, 3].values.tolist()
        self.ends = rna_data.iloc[:, 4].values.tolist()
        self.target = rna_data.iloc[:, 5].values.tolist()
        
        
        mirna_seqs = [x.replace('T', 'U') for x in mirna_seqs]
        dna_seqs = [x.replace('T', 'U') for x in dna_seqs]
        target_seqs = [x.replace('T', 'U') for x in target_seqs]

        #states = [x if x=='test' else 'train' for x in states ]
        self.data = [[x, y1,y2] for x, y1,y2 in zip(mirna_seqs, dna_seqs,target_seqs)]
        print(len(self.data))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"),
                                                 #model_max_length=10,
                                                 unk_token="[UNK]",
                                                 cls_token="[CLS]",
                                                 sep_token="[SEP]",
                                                 pad_token="[PAD]",
                                                 mask_token="[MASK]",
                                                 #padding_side = 'left',
                                                )
        #self.mask_token_id = self.tokenizer.mask_token_id
        #self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"
        self.mask_token_id = self.tokenizer.mask_token_id

        self.tokenizer_1mer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_1mer, "tokenizer.json"),
                                                 #model_max_length=10,
                                                 unk_token="[UNK]",
                                                 cls_token="[CLS]",
                                                 sep_token="[SEP]",
                                                 pad_token="[PAD]",
                                                 mask_token="[MASK]",
                                                 #padding_side = 'left',
                                                )
        #self.mask_token_id = self.tokenizer.mask_token_id
        #self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer_1mer.padding_side = "right"
        #self.mask_token_id = self.tokenizer.mask_token_id

    def __getitem__(self, index):
        rna_raw, dna_raw, target_site_ = self.data[index]
        starts_one = int((self.starts[index])/1)
        ends_one = int((self.ends[index])/1)+1
        target = int(self.target[index])
        #label = np.array(label, dtype=np.float32)
        #print(target_site_)
        #print(dna_[starts_one:ends_one])
        rna_ = split_into_six_mers(rna_raw)
        self.tokenizer_1mer.model_max_length = 14
        rna = self.tokenizer_1mer(rna_,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors='pt',)
        
        dna_ = split_into_six_mers(dna_raw)
        self.tokenizer.model_max_length = self.max_length
        dna = self.tokenizer(dna_,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors='pt',)

        
        mirna_input = rna.input_ids[0].numpy()
        mirna_attention = rna.attention_mask[0].numpy()

        dna_input = dna.input_ids[0].numpy()
        dna_attention = dna.attention_mask[0].numpy()


        
        
        return mirna_input, mirna_attention, dna_input,dna_attention,starts_one,ends_one,target,target_site_,dna_raw,rna_raw
        
    def __len__(self):
        return len(self.data)

class RNA_Seq_analysis(data.Dataset):
    def __init__(self, data_path=str,
                 tokenizer_path=str,
                 tokenizer_1mer=str,
                 max_length=32,
                 ):
        self.max_length = max_length
        rna_data = pd.read_json(data_path)
        #if mode == 'train':
        #te_data = te_data[te_data['Sequence'].str.len() <= 1024]
        mirna_seqs = rna_data.iloc[:, 0].values.tolist()
        dna_seqs = rna_data.iloc[:, 1].values.tolist()
        target_seqs = rna_data.iloc[:, 2].values.tolist()
        self.starts = rna_data.iloc[:, 3].values.tolist()
        self.ends = rna_data.iloc[:, 4].values.tolist()
        self.target = rna_data.iloc[:, 5].values.tolist()
        self.ensg = rna_data.iloc[:, 6].values.tolist()
        self.enst = rna_data.iloc[:, 7].values.tolist()
        
        
        mirna_seqs = [x.replace('T', 'U') for x in mirna_seqs]
        dna_seqs = [x.replace('T', 'U') for x in dna_seqs]
        target_seqs = [x.replace('T', 'U') for x in target_seqs]

        #states = [x if x=='test' else 'train' for x in states ]
        self.data = [[x, y1,y2] for x, y1,y2 in zip(mirna_seqs, dna_seqs,target_seqs)]
        print(len(self.data))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"),
                                                 #model_max_length=10,
                                                 unk_token="[UNK]",
                                                 cls_token="[CLS]",
                                                 sep_token="[SEP]",
                                                 pad_token="[PAD]",
                                                 mask_token="[MASK]",
                                                 #padding_side = 'left',
                                                )
        #self.mask_token_id = self.tokenizer.mask_token_id
        #self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"
        self.mask_token_id = self.tokenizer.mask_token_id

        self.tokenizer_1mer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_1mer, "tokenizer.json"),
                                                 #model_max_length=10,
                                                 unk_token="[UNK]",
                                                 cls_token="[CLS]",
                                                 sep_token="[SEP]",
                                                 pad_token="[PAD]",
                                                 mask_token="[MASK]",
                                                 #padding_side = 'left',
                                                )
        #self.mask_token_id = self.tokenizer.mask_token_id
        #self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer_1mer.padding_side = "right"
        #self.mask_token_id = self.tokenizer.mask_token_id

    def __getitem__(self, index):
        rna_raw, dna_raw, target_site_ = self.data[index]
        starts_one = int((self.starts[index])/1)
        ends_one = int((self.ends[index])/1)+1
        target = int(self.target[index])

        ensg = self.ensg[index]
        enst = self.enst[index]
        #label = np.array(label, dtype=np.float32)
        #print(target_site_)
        #print(dna_[starts_one:ends_one])
        rna_ = split_into_six_mers(rna_raw)
        self.tokenizer_1mer.model_max_length = 14
        rna = self.tokenizer_1mer(rna_,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors='pt',)
        
        dna_ = split_into_six_mers(dna_raw)
        self.tokenizer.model_max_length = self.max_length
        dna = self.tokenizer(dna_,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors='pt',)

        
        mirna_input = rna.input_ids[0].numpy()
        mirna_attention = rna.attention_mask[0].numpy()

        dna_input = dna.input_ids[0].numpy()
        dna_attention = dna.attention_mask[0].numpy()


        
        
        return mirna_input, mirna_attention, dna_input,dna_attention,starts_one,ends_one,target,target_site_,dna_raw,rna_raw,ensg,enst
        
    def __len__(self):
        return len(self.data)

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

