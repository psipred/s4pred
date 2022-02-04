# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:49:37 2020

@author:
    Lewis Moffat
    Bioinformatics Group - Comp. Sci. Dep., University College London (UCL)
    Github: limitloss

TO DO: Add option for sequential model loading and inference, and pull model loading out of module code (bad practice)
"""
from __future__ import print_function

import torch
import argparse
import os 

from network import S4PRED
from utilities import loadfasta


# =============================================================================
# Command Line Args
# =============================================================================
parser = argparse.ArgumentParser(description='Predict Secondary Structure with the S4PRED model', epilog='Takes a FASTA file for a single sequence and outputs to stdout')
parser.add_argument('input', metavar='input', type=str,
                    help='FASTA file with a single sequence')
parser.add_argument('--device', metavar='d', type=str, default='cpu',
                    help='Device to run on, Either: cpu or gpu (default; cpu)')
parser.add_argument('--outfmt', metavar='o', type=str, default='ss2',
                    help='Output Format, Either: ss2 or fas (default; ss2)')

args = parser.parse_args()
args_dict=vars(args)

# =============================================================================
# Model Initialization
# =============================================================================

# Load and freeze model
if args_dict['device']=='cpu': device = torch.device("cpu:0") 
if args_dict['device']=='gpu': device = torch.device("cuda:0") 

s4pred=S4PRED().to(device)
s4pred.eval()
# Setting requires_grad is redundant but pytorch has been weird in the past
s4pred.requires_grad=False


# Loading model parameters 

scriptdir = os.path.dirname(os.path.realpath(__file__))
                            
weight_files=['/weights/weights_1.pt',
              '/weights/weights_2.pt',
              '/weights/weights_3.pt',
              '/weights/weights_4.pt',
              '/weights/weights_5.pt']

# Manually listing for clarity and hot swapping in future
# Inelegant, ugly ugly, to be cleaned up in the future
s4pred.model_1.load_state_dict(torch.load(scriptdir + weight_files[0], map_location=lambda storage, loc: storage))
s4pred.model_2.load_state_dict(torch.load(scriptdir + weight_files[1], map_location=lambda storage, loc: storage))
s4pred.model_3.load_state_dict(torch.load(scriptdir + weight_files[2], map_location=lambda storage, loc: storage))
s4pred.model_4.load_state_dict(torch.load(scriptdir + weight_files[3], map_location=lambda storage, loc: storage))
s4pred.model_5.load_state_dict(torch.load(scriptdir + weight_files[4], map_location=lambda storage, loc: storage))




# =============================================================================
# Load Data and Run 
# =============================================================================

# Get the Data
data=loadfasta(args_dict['input'])

with torch.no_grad():
    ss_conf=s4pred(torch.tensor([data[1]]).to(device))
    ss=ss_conf.argmax(-1)
    # move the confidence scores out of log space
    ss_conf=ss_conf.exp()
    # renormalize to assuage any precision issues
    tsum=ss_conf.sum(-1)
    tsum=torch.cat((tsum.unsqueeze(1),tsum.unsqueeze(1),tsum.unsqueeze(1)),1)
    ss_conf/=tsum
    ss=ss.cpu().numpy()
    ss_conf=ss_conf.cpu().numpy()

# =============================================================================
# Sling the results to stdout
# =============================================================================

ind2char={0:"C",
          1:"H",
          2:"E"}
if args_dict['outfmt'] == 'ss2':
    print('# PSIPRED VFORMAT (S4PRED V1.0)\n')
    for i in range(len(ss)):
        print("%4d %c %c  %6.3f %6.3f %6.3f" % (i + 1, data[2][i], ind2char[ss[i]], ss_conf[i,0], ss_conf[i,1], ss_conf[i,2]))

if args_dict['outfmt'] == 'fas':
    print(data[0])
    print(data[2])
    print("".join([ind2char[s.item()] for s in ss]))  








