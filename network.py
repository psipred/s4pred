# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:44:17 2020

@author:
    Lewis Moffat
    Bioinformatics Group - Comp. Sci. Dep., University College London (UCL)
    Github: CraftyColossus

Inference Only Version of S4PRED - Single Sequence Secondary Structure Pred

This is culled down to exclude the various DropConnect/Dropout etc. from the 
training methods so that it is more clear.

If you'd like a training version of the model please raise an issue or submit a PR.
The AWD-GRU training script model was a tweak on the offical Salesforce 
AWD-LSTM (https://github.com/salesforce/awd-lstm-lm/). It needed to be adapted to 
take multiple layers of RNNs. 




"""

import torch
import torch.nn as nn
import torch.nn.functional as F




class ResidueEmbedding(nn.Embedding):
    def __init__(self, vocab_size=21, embed_size=128, padding_idx=None):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)

        
        
class GRUnet(nn.Module):
    def __init__(self,lstm_hdim=1024, embed_size=128, num_layers=3,bidirectional=True,lstm=False,outsize=3):
        super().__init__()
        """
            This version of the model has all the bells & whistles (e.g. 
            dropconnect) ripped out so its slimmed down for inference
            
        """
        
        self.lstm_hdim = lstm_hdim
        self.embed=ResidueEmbedding(vocab_size=22, embed_size=embed_size, padding_idx=21)
        self.lstm = nn.GRU(128, 1024, num_layers=3, bidirectional=True, batch_first=True,dropout=0.0)
        self.outlayer = nn.Linear(lstm_hdim*2, outsize)
        self.finalact=F.log_softmax

    
    def forward(self, x):
        """
            Assumes a batch size of one currently but can be changed
        """
        x=self.embed(x)
        x, _ = self.lstm(x)
        x=self.outlayer(x)
        x=self.finalact(x,dim=-1)
        return x.squeeze()        
        
        
class S4PRED(nn.Module):
    def __init__(self):
        super().__init__()
        """
            This loads the ensemble of models in a lazy way but its clear and 
            leaves the weight loading out of the run_model script. 
        """
                                            
        # Manually listing for clarity and hot swapping in future
        self.model_1=GRUnet()
        self.model_2=GRUnet()
        self.model_3=GRUnet()
        self.model_4=GRUnet()
        self.model_5=GRUnet()
        
    def forward(self, x):
        y_1=self.model_1(x)
        y_2=self.model_2(x)
        y_3=self.model_3(x)
        y_4=self.model_4(x)
        y_5=self.model_5(x)
        y_out=y_1*0.2+y_2*0.2+y_3*0.2+y_4*0.2+y_5*0.2
        return y_out        
        
        
        
        
        
        
        
        
