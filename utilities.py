# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:55:56 2020

@author:
    Lewis Moffat
    Github: limitloss

"""

from Bio import SeqIO

def aas2int(seq):
    aanumdict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 
             'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16,
             'W':17, 'Y':18, 'V':19}
    return [aanumdict.get(res, 20) for res in seq]

def LEGACY_loadfasta(fasta_file="test_seqs.fas"):
    '''
    Assumes a fasta file containing a single sequence
    '''
    
    seqs=[]
    with open(fasta_file,"r") as f:
        for line in f:
            line=line.rstrip("\n")
            if ">" in line:
                name=line
            else:
                seqs.append(line)
    seqs=''.join(seqs)
    iseqs=aas2int(seqs)
    data=[name,iseqs,seqs]
    return data 

def loadfasta(fasta_file="test_seqs.fas"):
    '''
    Takes a single FASTA file containing a variable number of sequences.
    Returns a list of each example, where each example is a list itself of the 
    Biopython-determined name string, the sequence as a list of integers, and the
    original sequence as a string.
    '''
    sequences = []
    
    records = list(SeqIO.parse(fasta_file, "fasta"))
    for record in records:
        name = record.name
        seq = str(record.seq)
        # Sanity checks for lowercase characters, gaps, etc.
        name = name.rstrip().upper().replace('-','')
        iseq = aas2int(seq)
        sequences.append([name,iseq,seq])
    
    return sequences