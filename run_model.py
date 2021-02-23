# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:49:37 2020

@author:
    Lewis Moffat
    Bioinformatics Group - Comp. Sci. Dep., University College London (UCL)
    Github: CraftyColossus

"""
from __future__ import print_function

import torch
import argparse

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








