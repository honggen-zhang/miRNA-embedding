import random
from EMA import EMA
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import transformers
from torch.utils import data
from transformers import DebertaTokenizerFast, T5EncoderModel, T5Config


class miRNAModel(nn.Module):
    def __init__(self,
                 vocab_size=4101,#261,#4101,#69,
                 hidden_size=512,
                 num_hidden_layers=4,
                 num_attention_heads=4,
                 pad_token_id=1
                 ):
        super().__init__()
        self.embed_dim = hidden_size
        model_cofig = T5Config()
        model_cofig.d_model = hidden_size
        model_cofig.num_attention_heads = num_attention_heads
        model_cofig.d_kv = hidden_size // num_attention_heads
        model_cofig.pad_token_id = pad_token_id
        model_cofig.num_layers = num_hidden_layers
        model_cofig.d_ff = hidden_size * 4
        model_cofig.vocab_size = vocab_size
        self.encoder = T5EncoderModel(config=model_cofig)

    def forward_logit(self, x, mask,return_attention=False):
        #with torch.no_grad():
        outputs = self.encoder(x,attention_mask=mask,output_hidden_states = True,return_dict=True,output_attentions=return_attention)
 
        hidden_states = outputs.hidden_states[-1]
        #hidden_states = hidden_states.reshape(hidden_states.size(0), -1)

        return hidden_states


    def forward(self, x, mask,return_attention=False):

        x = self.forward_logit(x,
                                mask,
                                return_attention,
                                )
        return x


class mRNA_encoder(nn.Module):
    def __init__(self,
                 vocab_size=4101,
                 hidden_size=256,
                 num_hidden_layers=4,
                 num_attention_heads=4,
                 pad_token_id=1
                 ):
        super().__init__()
        self.embed_dim = hidden_size
        model_cofig = T5Config()
        model_cofig.d_model = hidden_size
        model_cofig.num_attention_heads = num_attention_heads
        model_cofig.d_kv = hidden_size//num_attention_heads
        model_cofig.pad_token_id = pad_token_id
        model_cofig.num_layers = num_hidden_layers
        model_cofig.d_ff = hidden_size * 4
        model_cofig.vocab_size = vocab_size
        self.encoder = T5EncoderModel(config=model_cofig)

    def forward(self, x, attention_mask):
        outputs = self.encoder(x,
                               attention_mask=attention_mask,
                               return_dict=True,
                               output_attentions = True,
                               output_hidden_states = True,
                               )
        encoder_states =outputs.hidden_states[:]  # encoder layers outputs separately
        encoder_out = outputs.hidden_states[-1]
        attentions = outputs.attentions
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }


class contrastive_mRNA(nn.Module):
    def __init__(self, encoder_mi, encoder_m, **kwargs):
        super(contrastive_mRNA, self).__init__()
        self.embed_dim = 128
        self.encoder_mRNA = encoder_m
        self.encoder_miRNA = encoder_mi
        self.ema = EMA(self.encoder_mRNA)  # EMA acts as the teacher
        self.cross_attention = nn.MultiheadAttention(128, 4)
        self.ema_decay =  0.999
        self.ema_end_decay = 0.9999
        self.ema_anneal_end_step = 300000
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

        self.clf = nn.Sequential(nn.Linear(128, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 1),
                                )
        self.head_map = nn.Sequential(nn.Linear(128, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 128),
                                     )
    def ema_step(self):
        self.ema.step(self.encoder_mRNA)



    def forward(self,miRNA,miRNA_attention,
                target_site, target_site_attention = None,
                mRNA=None, mRNA_attention=None,start=None, **kwargs):
        # model forward in online mode (student)
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention)
        x_short = self.encoder_mRNA(target_site, target_site_attention)['encoder_out']
        #x_long = self.encoder_mRNA(mRNA, mRNA_attention)['encoder_out']

        with torch.no_grad():
            self.ema.model.eval()
            x_long = self.ema.model(mRNA, mRNA_attention)['encoder_out']
            #x_short = self.ema.model(target_site, target_site_attention)['encoder_out']


        cross_attend, _ = self.cross_attention(miRNA_emb.permute(1,0,2), x_short.permute(1,0,2), x_short.permute(1,0,2))

        cross_attend_long, attn_weights = self.cross_attention(miRNA_emb.permute(1,0,2), x_long.permute(1,0,2), x_long.permute(1,0,2))

        #print('attention',attn_weights.shape)
        #print('start',start.shape)

        target_distribution = torch.zeros_like(attn_weights)+0.001
        for i in range(len(start)):
            start_pos =int(start[i]/3)+1
            end_pos = min(start_pos + 14, target_distribution.size(-1))
            target_distribution[i, :, start_pos:end_pos] = 1.0 / (end_pos - start_pos)

        target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True)

        #print('target_distribution',target_distribution[i,0,start_pos-5:])
        #print('attn_weights',attn_weights[i,0,start_pos-5:])
        attn_weights = torch.clamp(attn_weights, min=1e-9)

        kl_div_loss = F.kl_div(attn_weights.log(), target_distribution, reduction='batchmean')

        #print('kl_div_loss',kl_div_loss)
        #print(hh)


        #cross_attend, _ = self.cross_attention(x_short.permute(1,0,2), miRNA_emb.permute(1,0,2), miRNA_emb.permute(1,0,2))
        #cross_attend_long, _ = self.cross_attention(x_long.permute(1,0,2), miRNA_emb.permute(1,0,2), miRNA_emb.permute(1,0,2))
        norm_cross_attend = F.normalize(cross_attend.mean(dim=0),p=2,dim=1)
        
        f_o = self.fc1(norm_cross_attend)
        #f_o = self.fc1(cross_attend)
        f_o = torch.relu(f_o)
        f_o = self.fc2(f_o)
        ffo = norm_cross_attend + f_o
        #ffo = f_o
        ffo_pool = ffo#.mean(dim=0)
        clf_result = self.clf(ffo_pool)
        #logits = torch.softmax(clf_result, dim=1)
        norm_cross_attend_long = F.normalize(cross_attend_long.mean(dim=0),p=2,dim=1)
        f_o_n = self.fc1(norm_cross_attend_long)
        f_o_n = torch.relu(f_o_n)
        f_o_n = self.fc2(f_o_n)
        ffo_n = norm_cross_attend_long + f_o_n
        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_long = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)

        #mapped_cross_attend = self.head_map(cross_attend)
        


        return cross_attend.mean(dim=0), cross_attend_long.mean(dim=0),clf_result,clf_result_long,kl_div_loss
    def forward1(self,miRNA,miRNA_attention,
                target_site, target_site_attention = None,
                mRNA=None, mRNA_attention=None, **kwargs):
        # model forward in online mode (student)
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention)
        x_short = self.encoder_mRNA(target_site, target_site_attention)['encoder_out']
        #x_long = self.encoder_mRNA(mRNA, mRNA_attention)['encoder_out']

        with torch.no_grad():
            self.ema.model.eval()
            x_long = self.ema.model(mRNA, mRNA_attention)['encoder_out']
            #x_short = self.ema.model(target_site, target_site_attention)['encoder_out']


        cross_attend, _ = self.cross_attention(miRNA_emb.permute(1,0,2), x_short.permute(1,0,2), x_short.permute(1,0,2))

        cross_attend_long, _ = self.cross_attention(miRNA_emb.permute(1,0,2), x_long.permute(1,0,2), x_long.permute(1,0,2))


        #cross_attend, _ = self.cross_attention(x_short.permute(1,0,2), miRNA_emb.permute(1,0,2), miRNA_emb.permute(1,0,2))
        #cross_attend_long, _ = self.cross_attention(x_long.permute(1,0,2), miRNA_emb.permute(1,0,2), miRNA_emb.permute(1,0,2))
        norm_cross_attend = F.normalize(cross_attend.mean(dim=0),p=2,dim=1)
        
        f_o = self.fc1(norm_cross_attend)
        #f_o = self.fc1(cross_attend)
        f_o = torch.relu(f_o)
        f_o = self.fc2(f_o)
        ffo = norm_cross_attend + f_o
        #ffo = f_o
        ffo_pool = ffo#.mean(dim=0)
        clf_result = self.clf(ffo_pool)
        #logits = torch.softmax(clf_result, dim=1)
        norm_cross_attend_long = F.normalize(cross_attend_long.mean(dim=0),p=2,dim=1)
        f_o_n = self.fc1(norm_cross_attend_long)
        f_o_n = torch.relu(f_o_n)
        f_o_n = self.fc2(f_o_n)
        ffo_n = norm_cross_attend_long + f_o_n
        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_long = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)

        #mapped_cross_attend = self.head_map(cross_attend)
        


        return cross_attend.mean(dim=0), cross_attend_long.mean(dim=0),clf_result,clf_result_long
    
    def forward_eval(self,miRNA,miRNA_attention,
        mRNA=None, mRNA_attention=None, **kwargs):
        # model forward in online mode (student)
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention)
        a,b,c = miRNA_emb.shape #batch size, seq length, emb
        #with torch.no_grad():
        #self.ema.model.eval()
        x_long = self.encoder_mRNA(mRNA, mRNA_attention)['encoder_out']
            
        cross_attend_long, attn_weights = self.cross_attention(miRNA_emb.permute(1,0,2), x_long.permute(1,0,2), x_long.permute(1,0,2))
        norm_cross_attend_long = F.normalize(cross_attend_long.mean(dim=0),p=2,dim=1)

        f_o_n = self.fc1(norm_cross_attend_long)
        f_o_n = torch.relu(f_o_n)
        f_o_n = self.fc2(f_o_n)
        ffo_n =  norm_cross_attend_long + f_o_n
        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_long = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)
        


        return attn_weights,clf_result_long


class contrastive_mRNA2(nn.Module):
    def __init__(self, encoder_mi, encoder_m, **kwargs):
        super(contrastive_mRNA2, self).__init__()
        self.embed_dim = 128
        self.encoder_mRNA = encoder_m
        self.encoder_miRNA = encoder_mi
        self.ema = EMA(self.encoder_mRNA)  # EMA acts as the teacher
        self.cross_attention = nn.MultiheadAttention(128, 4)
        self.ema_decay =  0.999
        self.ema_end_decay = 0.9999
        self.ema_anneal_end_step = 300000
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

        self.clf = nn.Sequential(nn.Linear(128, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 1),
                                )
        self.head_map = nn.Sequential(nn.Linear(128, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 128),
                                     )
    def ema_step(self):
        self.ema.step(self.encoder_mRNA)

    def cross_attention_clf(self,query,key,value,batch_size):
# Efficient negative sampling with vectorized attention and normalization
        expanded_query = query.unsqueeze(2).repeat(1, 1, batch_size, 1)  # [seq_len, batch_size, batch_size, emb_dim]

        neg_outputs = []
        for i in range(batch_size):
            query_i = query[:, i:i+1]  # Select i-th query

            # Expand query to match the negative key dimensions
            expanded_query = query_i.repeat(1, key.size(1), 1)

            # Perform cross-attention with negative keys and values
            neg_output, _ = self.cross_attention(expanded_query, key, value)
            neg_outputs.append(neg_output)

        # Concatenate negative attention outputs
        neg_cross_attend = torch.cat(neg_outputs, dim=1)

    
        # Compute attention scores (scaled dot-product attention)
        #attention_scores = torch.matmul(expanded_query, key.unsqueeze(0).transpose(-2, -1))  # [seq_len, batch_size, batch_size, seq_len]
        #attention_scores = attention_scores / (emb_dim ** 0.5)  # Scale scores
        #attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize scores
        
        # Apply attention weights to values
        #neg_output = torch.matmul(attention_weights, value.unsqueeze(0))  # [seq_len, batch_size, batch_size, emb_dim]
        
        # Extract diagonal for positive cross-attention
        #print(neg_cross_attend.shape)
        #print(batch_size,key.size(2))
        neg_cross_attend_ = neg_cross_attend.view(40,batch_size,batch_size,key.size(2))
        diagonal_indices = torch.arange(batch_size)  # Indices for the diagonal
        pos_cross_attend = neg_cross_attend_[:, diagonal_indices, diagonal_indices, :] 

        
        # Concatenate negative outputs along the batch dimension
        #neg_cross_attend = neg_output.view(seq_len, batch_size * batch_size, emb_dim).permute(1, 0, 2)  # [batch_size*batch_size, seq_len, emb_dim]
        
        # Pass features through fully connected layers
        f_1_neg = self.fc1(neg_cross_attend.mean(dim=0))  # [batch_size*batch_size, hidden_dim]
        f_1_neg = F.relu(f_1_neg)
        f_2_neg = self.fc2(f_1_neg)  # [batch_size*batch_size, output_dim]
        ff_neg = neg_cross_attend.mean(dim=0) + f_2_neg  # [batch_size*batch_size, output_dim]
        
        # Pool final feature and classify
        ff_neg_pool = ff_neg  # Apply additional pooling if required
        clf_result_neg = self.clf(ff_neg_pool)  # [batch_size*batch_size, num_classes]
        return pos_cross_attend,clf_result_neg.view(batch_size,batch_size)


    def forward(self,miRNA,miRNA_attention,
                target_site, target_site_attention = None,
                mRNA=None, mRNA_attention=None, **kwargs):
        # Encode miRNA and target site
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention)  # [batch_size, seq_len, emb_dim]
        x_short = self.encoder_mRNA(target_site, target_site_attention)['encoder_out']  # [batch_size, seq_len, emb_dim]
        
        # Encode mRNA using EMA model in evaluation mode
        with torch.no_grad():
            self.ema.model.eval()
            x_long = self.ema.model(mRNA, mRNA_attention)['encoder_out']  # [batch_size, seq_len, emb_dim]
        
        batch_size, seq_len, emb_dim = miRNA_emb.shape
        query = miRNA_emb.permute(1, 0, 2)  # [seq_len, batch_size, emb_dim]
        key = x_short.permute(1, 0, 2)      # [seq_len, batch_size, emb_dim]
        value = x_short.permute(1, 0, 2)    # [seq_len, batch_size, emb_dim]
        pos_cross_attend,clf_result_neg = self.cross_attention_clf(query,key, value,batch_size)
        key = x_long.permute(1, 0, 2)      # [seq_len, batch_size, emb_dim]
        value = x_long.permute(1, 0, 2)    # [seq_len, batch_size, emb_dim]
        pos_cross_attend_long,clf_result_neg_long = self.cross_attention_clf(query,key, value,batch_size)
    

        mapped_cross_attend = self.head_map(pos_cross_attend)
        


        return mapped_cross_attend, pos_cross_attend_long,clf_result_neg,clf_result_neg_long
    
    def forward_eval(self,miRNA,miRNA_attention,
        mRNA=None, mRNA_attention=None, **kwargs):
        # model forward in online mode (student)
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention)
        a,b,c = miRNA_emb.shape #batch size, seq length, emb
        #with torch.no_grad():
        #self.ema.model.eval()
        x_long = self.encoder_mRNA(mRNA, mRNA_attention)['encoder_out']
            
        cross_attend_long, attn_weights = self.cross_attention(miRNA_emb.permute(1,0,2), x_long.permute(1,0,2), x_long.permute(1,0,2))

        f_o_n = self.fc1(cross_attend_long.mean(dim=0))
        f_o_n = torch.relu(f_o_n)
        f_o_n = self.fc2(f_o_n)
        ffo_n =  cross_attend_long.mean(dim=0) + f_o_n
        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_long = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)
        
