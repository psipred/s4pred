# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:49:37 2020

@author:
    Lewis Moffat
    Github: limitloss

TO DO: Add option for sequential model loading and inference, and pull model loading out of module code (bad practice)
"""
from __future__ import print_function

import torch
import argparse
import os 

import numpy as np

from network import S4PRED
from utilities import loadfasta


# =============================================================================
# Command Line Args
# =============================================================================
parser = argparse.ArgumentParser(description='Predict Secondary Structure with the S4PRED model', epilog='Takes a FASTA file containing an arbitrary number of individual sequences and outputs a secondary structure prediction for each.')
parser.add_argument('input', metavar='input', type=str,
                    help='FASTA file with a single sequence.')
parser.add_argument('-d','--device', metavar='d', type=str, default='cpu',
                    help='Device to run on, Either: cpu or gpu (default; cpu).')
parser.add_argument('-t','--outfmt', metavar='m', type=str, default='ss2',
                    help='Output format, Either: ss2, fas, or horiz (default; ss2).')
parser.add_argument('-c','--fas-conf', default=False, action='store_true',
                    help='Include confidence scores if using .fas output.')
parser.add_argument('-s','--silent', default=False, action='store_true',
                    help='Suppress printing predictions to stdout.')
parser.add_argument('-z','--save-files', default=False, action='store_true',
                    help='Save each input sequence prediction in an individual file. Makes and saves to a directory called preds in the same dir as this script unless --outdir is specified.')
parser.add_argument('-o','--outdir', metavar='p', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                    help='Absolute file-path where files are to be saved, if --save-files is used.')
parser.add_argument('-x','--save-by-idx', default=False, action='store_true',
                    help='If saving with --save-files, use a counter to name files instead of sequence ID.')
parser.add_argument('-t2','--outfmt2', metavar='n', type=str, default='',
                    help='Save output with a 2nd format, Either: ss2, fas, or horiz (default; None).')
parser.add_argument('-p','--prefix', metavar='n', type=str, default=None,
                   help='Use prefix for output filenames, rather than stdout (default; None).')
parser.add_argument('-T','--threads', metavar='n', type=int, default=None,
                   help='Number of CPU threads to use for inference (default; Number of CPU cores).')
args = parser.parse_args()
args_dict=vars(args)

# =============================================================================
# Model Initialization
# =============================================================================

# set number of threads (so we can play nice with other parallel processes)
if args_dict['threads']:
    torch.set_num_threads(args_dict['threads'])

# Load and freeze model
if args_dict['device']=='cpu': device = torch.device('cpu:0') 
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
# Load Data
# =============================================================================

# Get the Data
seqs = loadfasta(args_dict['input'])

def predict_sequence(data):
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
    return ss, ss_conf

# =============================================================================
# Output helpers
# =============================================================================

ind2char={0:"C", 1:"H", 2:"E"}

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def format_ss2(data, ss, ss_conf):
    ''' Formats output for the PSIPRED VFORMAT .ss2 files. 
    ''' 
    lines = ['# PSIPRED VFORMAT (S4PRED V1.2.4)\n']
    for i in range(len(ss)):
        lines.append("%4d %c %c  %6.3f %6.3f %6.3f" % (i + 1, data[2][i], ind2char[ss[i]], ss_conf[i,0], ss_conf[i,1], ss_conf[i,2]))
    return lines

def format_fas(data, ss, ss_conf, include_conf=False):
    ''' Formats output as a pseudo-FASTA file
    ''' 
    lines=['>'+data[0]]
    lines.append(data[2])
    lines.append("".join([ind2char[s.item()] for s in ss]))
    
    if include_conf:
        lines.append(np.array2string(ss_conf[:,0],max_line_width=1e6, precision=3,formatter={'float_kind':lambda x: "%.3f" % x}).replace('[','').replace(']',''))
        lines.append(np.array2string(ss_conf[:,1],max_line_width=1e6, precision=3,formatter={'float_kind':lambda x: "%.3f" % x}).replace('[','').replace(']',''))
        lines.append(np.array2string(ss_conf[:,2],max_line_width=1e6, precision=3,formatter={'float_kind':lambda x: "%.3f" % x}).replace('[','').replace(']',''))
    
    return lines
    
def format_horiz(data, ss, ss_conf):
    ''' Formats output for the PSIPRED HFORMAT .horiz files. 
        Care must be taken as there is a fixed column width of 60 char
    '''    
    lines=['# PSIPRED HFORMAT (S4PRED V1.2.4)']
    sub_seqs = list(chunkstring(data[2],60))
    sub_ss   = list(chunkstring("".join([ind2char[s.item()] for s in ss]),60))
    
    num_len  =  int(np.floor(len(data[2])/10))
    num_seq  = ''.join(f'{str((i+1)*10):>10}' for i in range(num_len+1))
    num_seq  = list(chunkstring(num_seq,60))
        
    # get confidences then floor them and convert to string 
    conf_idxs = ss_conf.argmax(-1)
    confs = ss_conf[np.arange(len(conf_idxs)),conf_idxs[:]]
    confs = "".join([str(x) for x in np.floor(confs*10).astype(np.int32)])
    confs = list(chunkstring(confs,60))

    for idx, subsq in enumerate(sub_seqs):
        lines.append(f'\nConf: {confs[idx]}')
        lines.append(f'Pred: {sub_ss[idx]}')
        lines.append(f'  AA: {subsq}')
        lines.append(f'      {num_seq[idx]}\n')
        
    return lines

# =============================================================================
# Predict then print AND/OR save
# =============================================================================

# make sure last char is a '/'
output_dir = args_dict['outdir']
if output_dir[-1] != '/': output_dir+='/'

# if directory isnt specified and we're saving then we make a preds/ dir in 
# s4pred directory
scriptdir = os.path.dirname(os.path.realpath(__file__))+'/'
if output_dir == scriptdir and args_dict['save_files']:
    os.makedirs(scriptdir+'preds/', exist_ok=True)
    output_dir += 'preds/'
                                 

# Run the loop through each sequence and predict then save and/or print
for idx, data in enumerate(seqs):
    
    ss, ss_conf = predict_sequence(data)
    
    if args_dict['outfmt'] == 'ss2':
        lines=format_ss2(data, ss, ss_conf)
        suffix = '.ss2'
    elif args_dict['outfmt'] == 'fas':
        lines=format_fas(data, ss, ss_conf, include_conf=args_dict['fas_conf'])
        suffix = '.fas'
    elif args_dict['outfmt'] == 'horiz':
        lines=format_horiz(data, ss, ss_conf)
        suffix = '.horiz'
        
    if not args_dict['silent']:
        for line in lines: print(line)
    else:
        if not args_dict['save_files']:
            raise ValueError('Using --silent and not using --save-files will lead to no output.')
        
    if args_dict['save_files']:
            
        
        if args_dict['save_by_idx']:
            file_name = 's4_out_'+str(idx)+suffix
        else:
            if args_dict['prefix']:
                file_name = args_dict['prefix'] + suffix
            else:
                file_name = data[0]+suffix
        
        file_path = output_dir + file_name
        
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line+'\n')
        
    # repeat boolean logic cascade for if a secondary output format is provided
    if len(args_dict['outfmt2'])>2:
        if args_dict['outfmt2'] == 'ss2':
            lines=format_ss2(data, ss, ss_conf)
            suffix = '.ss2'
        elif args_dict['outfmt2'] == 'fas':
            lines=format_fas(data, ss, ss_conf, include_conf=args_dict['fas_conf'])
            suffix = '.fas'
        elif args_dict['outfmt2'] == 'horiz':
            lines=format_horiz(data, ss, ss_conf)
            suffix = '.horiz'
        else:
            raise ValueError('Invalid 2nd output file format provided. Use horiz, ss2, or fas.')
        
        if not args_dict['silent']:
            for line in lines: print(line)
        
        if args_dict['save_files']:
            
            if args_dict['save_by_idx']:
  
               file_name = 's4_out_'+str(idx)+suffix
            else:
                if args_dict['prefix']:
                    file_name = args_dict['prefix'] + suffix
                else:
                    file_name = data[0]+suffix
             
            file_path = output_dir + file_name
            
            with open(file_path, 'w') as f:
                for line in lines:
                    f.write(line+'\n')
        
        
        
        
        
        
        
        
        
        
        
        
        
