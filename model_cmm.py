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

class mRNA_encoder_(nn.Module):
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
    def ema_step(self):
        self.ema.step(self.encoder_mRNA)


    def forward(self,miRNA,miRNA_attention,
                target_site, target_site_attention = None,
                mRNA=None, mRNA_attention=None, **kwargs):
        # model forward in online mode (student)
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention)
        x_short = self.encoder_mRNA(target_site, target_site_attention)['encoder_out']

        with torch.no_grad():
            self.ema.model.eval()
            x_long = self.ema.model(mRNA, mRNA_attention)['encoder_out']


        cross_attend, _ = self.cross_attention(miRNA_emb.permute(1,0,2), x_short.permute(1,0,2), x_short.permute(1,0,2))

        cross_attend_long, _ = self.cross_attention(miRNA_emb.permute(1,0,2), x_long.permute(1,0,2), x_long.permute(1,0,2))

        
        f_o = self.fc1(cross_attend.mean(dim=0))
        f_o = torch.relu(f_o)
        f_o = self.fc2(f_o)
        #ffo = cross_attend + f_o
        ffo = f_o
        ffo_pool = ffo#.mean(dim=0)
        clf_result = self.clf(ffo_pool)
        #logits = torch.softmax(clf_result, dim=1)

        f_o_n = self.fc1(cross_attend_long.mean(dim=0))
        f_o_n = torch.relu(f_o_n)
        f_o_n = self.fc2(f_o_n)
        ffo_n =  f_o_n
        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_long = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)
        


        return cross_attend, cross_attend_long,clf_result,clf_result_long
    
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
        ffo_n =  f_o_n
        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_long = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)
        


        return attn_weights,clf_result_long
class mRNA2vec_simple(nn.Module):
    def __init__(self, encoder_mi, encoder_m, **kwargs):
        super(mRNA2vec_simple, self).__init__()
        self.embed_dim = 128
        self.encoder_mRNA = encoder_m
        self.encoder_miRNA = encoder_mi
        self.cross_attention = nn.MultiheadAttention(128, 4)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

        self.clf = nn.Sequential(nn.Linear(128, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 1),
                                )


    def forward(self,miRNA,miRNA_attention_mask,
                mRNA_masked, mRNA_unmasked=None, 
                mRNA_attention_mask=None,label_mask = None, **kwargs):
        # model forward in online mode (student)
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention_mask)
        #miRNA_emb = torch.exp(miRNA_emb)
        #scale_factor = torch.sqrt(torch.tensor(miRNA_emb.size(-1), dtype=torch.float32))


        a,b,c = miRNA_emb.shape #batch size, seq length, emb
        
        outputs = self.encoder_mRNA(mRNA_masked, mRNA_attention_mask) # fetch the last layer outputs
        x_short = outputs['encoder_out']
        #x_attentions = outputs['attentions']


        cross_attend, _ = self.cross_attention(miRNA_emb.permute(1,0,2), x_short.permute(1,0,2), x_short.permute(1,0,2))


        neg_outputs = []
        query = miRNA_emb.permute(1,0,2)
        key = x_short.permute(1,0,2)
        value = x_short.permute(1,0,2)

        for i in range(a):
            # Build negative key/value by excluding the i-th sequence
            neg_key = torch.cat([key[:, :i], key[:, i+1:]], dim=1)    
            neg_value = torch.cat([value[:, :i], value[:, i+1:]], dim=1) 
            query_i = query[:, i:i+1] 
            expanded_query = query_i.repeat(1, neg_key.size(1), 1)

            #print(f"query_i shape: {query_i.shape}")
            #print(f"neg_key shape: {neg_key.shape}")
            #print(f"neg_value shape: {neg_value.shape}")

            # Perform attention between query i and negative key/value
            neg_output, _ = self.cross_attention(expanded_query, neg_key, neg_value)
            
            # Collect the negative outputs and weights for this query
            neg_outputs.append(neg_output)
        
        # Concatenate negative outputs and weights
        neg_cross_attend = torch.cat(neg_outputs, dim=1)  # Concatenate across the batch
        #print('neg_cross_attend',neg_cross_attend.shape)
        #print('pos_cross_attend',cross_attend.shape)
        

        #cross_attend_neg, _ = self.cross_attention(miRNA_emb.permute(1,0,2)[:,1:,:], x_short.permute(1,0,2)[:,:-1,:], x_long.permute(1,0,2)[:,:-1,:])
        
        f_o = self.fc1(cross_attend.mean(dim=0))
        f_o = torch.relu(f_o)
        f_o = self.fc2(f_o)
        #ffo = cross_attend + f_o
        ffo = f_o

        f_o_n = self.fc1(neg_cross_attend.mean(dim=0))
        f_o_n = torch.relu(f_o_n)
        f_o_n = self.fc2(f_o_n)
        ffo_n =  f_o_n

        ffo_pool = ffo#.mean(dim=0)
        clf_result = self.clf(ffo_pool)
        #logits = torch.softmax(clf_result, dim=1)

        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_n = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)

        return x_short, x_short,clf_result,clf_result_n

class mRNA2vec(nn.Module):
    def __init__(self, encoder_mi, encoder_m, **kwargs):
        super(mRNA2vec, self).__init__()
        self.embed_dim = 128
        self.encoder_mRNA = encoder_m
        self.encoder_miRNA = encoder_mi
        self.ema = EMA(self.encoder_mRNA)  # EMA acts as the teacher
        self.regression_head = self._build_regression_head()
        self.ema_decay =  0.999
        self.ema_end_decay = 0.9999
        self.ema_anneal_end_step = 300000
        self.cross_attention = nn.MultiheadAttention(128, 4)
        self.fc1 = nn.Linear(128*5, 256)
        self.fc2 = nn.Linear(256, 128)

        self.clf = nn.Sequential(nn.Linear(128, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 256),
                                 nn.ReLU (),
                                 nn.Linear(256, 1),
                                )



    def _build_regression_head(self):
        return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim))

    def ema_step(self):
        self.ema.step(self.encoder_mRNA)

    def forward(self,miRNA,miRNA_attention_mask,
                mRNA_masked, mRNA_unmasked=None, 
                mRNA_attention_mask=None,label_mask = None, **kwargs):
        # model forward in online mode (student)
        miRNA_emb = self.encoder_miRNA(miRNA, miRNA_attention_mask)
        #miRNA_emb = torch.exp(miRNA_emb)
        #scale_factor = torch.sqrt(torch.tensor(miRNA_emb.size(-1), dtype=torch.float32))


        a,b,c = miRNA_emb.shape #batch size, seq length, emb
        
        outputs = self.encoder_mRNA(mRNA_masked, mRNA_attention_mask) # fetch the last layer outputs
        x_short = outputs['encoder_out']
        #x_attentions = outputs['attentions']

        with torch.no_grad():
            self.ema.model.eval()
            x_long = self.ema.model(mRNA_unmasked, mRNA_attention_mask)['encoder_out']  
            #x_long = x_long[-3:]  # take the last k transformer layers
            #x_long = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in x_long]
            #x_long = sum(x_long) / len(x_long)
            #if self.cfg.model.normalize_targets:
            #x_long = F.layer_norm(x_long.float(), x_long.shape[-1:])
        
        #boolean_mask = label_mask == 0 
        #attn_mask = boolean_mask.unsqueeze(1).expand(a, 1024, 1024) 
        #attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
        #attn_mask = attn_mask.repeat_interleave(4, dim=0)

        cross_attend, attn_weights = self.cross_attention(miRNA_emb.permute(1,0,2), x_long.permute(1,0,2), x_short.permute(1,0,2))




        neg_outputs = []
        query = miRNA_emb.permute(1,0,2)
        key = x_long.permute(1,0,2)
        value = x_short.permute(1,0,2)

        for i in range(a):
            # Build negative key/value by excluding the i-th sequence
            neg_key = torch.cat([key[:, :i], key[:, i+1:]], dim=1)    
            neg_value = torch.cat([value[:, :i], value[:, i+1:]], dim=1) 
            query_i = query[:, i:i+1] 
            expanded_query = query_i.repeat(1, neg_key.size(1), 1)

            #print(f"query_i shape: {query_i.shape}")
            #print(f"neg_key shape: {neg_key.shape}")
            #print(f"neg_value shape: {neg_value.shape}")

            # Perform attention between query i and negative key/value
            neg_output, _ = self.cross_attention(expanded_query, neg_key, neg_value)
            
            # Collect the negative outputs and weights for this query
            neg_outputs.append(neg_output)
        
        # Concatenate negative outputs and weights
        neg_cross_attend = torch.cat(neg_outputs, dim=1)  # Concatenate across the batch
        #print('neg_cross_attend',neg_cross_attend.shape)
        #print('pos_cross_attend',cross_attend.shape)
        

        #cross_attend_neg, _ = self.cross_attention(miRNA_emb.permute(1,0,2)[:,1:,:], x_short.permute(1,0,2)[:,:-1,:], x_long.permute(1,0,2)[:,:-1,:])
        
        f_o = self.fc1(cross_attend.view(a, -1))
        f_o = torch.relu(f_o)
        f_o = self.fc2(f_o)
        #ffo = cross_attend + f_o
        ffo = f_o

        f_o_n = self.fc1(neg_cross_attend.view(a*(a-1), -1))
        f_o_n = torch.relu(f_o_n)
        f_o_n = self.fc2(f_o_n)
        ffo_n =  f_o_n

        ffo_pool = ffo#.mean(dim=0)
        clf_result = self.clf(ffo_pool)
        #logits = torch.softmax(clf_result, dim=1)

        ffo_n_pool = ffo_n#.mean(dim=0)
        clf_result_n = self.clf(ffo_n_pool)
        #logits_n = torch.softmax(clf_result_n, dim=1)
        

        #logits = torch.cat((clf_result, clf_result_n), dim=0) 
        #positive_labels = torch.ones(a, 1)  # Labels for positive class
        #negative_labels = torch.zeros(a-1, 1)  # Labels for negative class
        #labels = torch.cat((positive_labels, negative_labels), dim=0) 

        

        masked_indices = label_mask.eq(0)
        #print('label_mask',label_mask)
        #print('masked_indices',masked_indices)
        #x_short = x_short#[masked_indices]
        #x_long = x_long#[masked_indices]

        x_short = self.regression_head(x_short).mean(dim=1)
        x_long = self.regression_head(x_long).mean(dim=1)
        #print('x_short',x_short.shape)
        #print('x_long',x_long.shape)

        return x_short, x_long,clf_result,clf_result_n

