#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:49:23 2022

@author: smuradoglu
"""
import pandas as pd
import os
#%%#

lang = "tur"
os.chdir('/home/salkazzar/Documents/Active_learning_in_morphology/iterative/%s' %lang)
#%%##%% Open data files and clean lines
source = open('tst.%s_cycle1.input' %lang, 'r') 
source_lines= source.readlines()
output = open('tst.%s_cycle1.output' %lang, 'r')
output_lines= output.readlines()
pred = open('tst.%s_cycle1.guesses' %lang, 'r')
pred_lines= pred.readlines()
loglike = open('tst.%s_cycle1n2.guesses' %lang, 'r')
loglike_lines= loglike.readlines()


src_frm=[]
for i in range(len(source_lines)):
    src_frm.append(source_lines[i].rstrip('\n').split('\t'))
    
trg_frm=[]
for i in range(len(output_lines)):
    trg_frm.append(output_lines[i].rstrip('\n').split('\t'))
    
pred_frm=[]
for i in range(len(pred_lines)):
    pred_frm.append(pred_lines[i].rstrip('\n').split('\t'))
    
log_like=[]
for i in range(len(loglike_lines)):
    log_like.append(loglike_lines[i].rstrip('\n').split('\t'))

print(trg_frm[71])
print(pred_frm[71])

#%%# Create Truth/correctness table

correct = []

for i in range(len(trg_frm)):
    if trg_frm[i] == pred_frm[i]:
        correct.append('1')
    else:
        correct.append('0')

false = correct.count('0')
true = correct.count('1')

acc = true / (true + false)
print(acc)

#%%# split n1  n2 values
n1 = log_like[0::2]
n2 = log_like[1::2]

#%%#

from difflib import ndiff

def calculate_levenshtein_distance(str_1, str_2):
    """
        The Levenshtein distance is a string metric for measuring the difference between two sequences.
        It is calculated as the minimum number of single-character edits necessary to transform one string into another
    """
    distance = 0
    buffer_removed = buffer_added = 0
    for x in ndiff(str_1, str_2):
        code = x[0]
        # Code ? is ignored as it does not translate to any modification
        if code == ' ':
            distance += max(buffer_removed, buffer_added)
            buffer_removed = buffer_added = 0
        elif code == '-':
            buffer_removed += 1
        elif code == '+':
            buffer_added += 1
    distance += max(buffer_removed, buffer_added)
    return distance

#%%#
lev_dist = []

for k in range(len(trg_frm)):
    lev_dist.append(calculate_levenshtein_distance(trg_frm[k], pred_frm[k]))

print(lev_dist[19])

print(trg_frm[19], pred_frm[19])

#%%# Resample from 1st model test files

#Combine list

#Get rid of nested lists
clean_source = []
for sources in src_frm:
    for source in sources:
        clean_source.append(source)

clean_output = []
for outputs in trg_frm:
    for output in outputs:
        clean_output.append(output)
        
clean_guess = []
for guesses in pred_frm:
    for guess in guesses:
        clean_guess.append(guess)
        
clean_correct = []
for corrects in correct:
    for value in corrects:
        clean_correct.append(value)

clean_n1 = []
for n1s in n1:
    for l in n1s:
        clean_n1.append(float(l))

clean_n2 = []
for n2s in n2:
    for n in n2s:
        clean_n2.append(float(n))
        
#create dataframe from lists        
df = pd.DataFrame(list(zip(clean_source, clean_output, clean_guess, clean_correct, clean_n1, clean_n2, lev_dist)),
               columns =['input', 'output', 'guess', 'correct', 'n1','n2', 'Lev'])

df['n2 - n1'] = df['n2'] - df['n1']

df



#%%# Sample incorrect forms  
sample_size =107

if false == sample_size:
    INC_order = df.sort_values(['correct'], ascending=[True])
    INC_resamp = INC_order.head(sample_size)
    INC_remain_tst = INC_order.tail(df.shape[0] -sample_size)
if false < sample_size: 
    INC_order =  df.sort_values(['correct', 'n2 - n1'], ascending=[True, True]) 
    INC_resamp = INC_order.head(sample_size)
    INC_remain_tst = INC_order.tail(df.shape[0] -sample_size)
if false > sample_size:
    INC_order  =  df.sort_values(['correct', 'Lev'], ascending=[True, False])
    INC_resamp = INC_order.head(sample_size)
    INC_remain_tst = INC_order.tail(df.shape[0] -sample_size)

INC_resamp

#%%#Sample correct forms 
sample_size =107

if true == sample_size:
    CF_order = df.sort_values(['correct'], ascending=[False])
    CF_resamp = CF_order.head(sample_size)
    CF_remain_tst = CF_order.tail(df.shape[0] -sample_size)
if true < sample_size: 
    CF_order =  df.sort_values(['correct', 'n2 - n1'], ascending=[False, False]) 
    CF_resamp = CF_order.head(sample_size)
    CF_remain_tst = CF_order.tail(df.shape[0] -sample_size)
if true > sample_size:
    CF_order =  df.sort_values(['correct', 'Lev'], ascending=[False, True])
    CF_resamp = CF_order.head(sample_size)
    CF_remain_tst = CF_order.tail(df.shape[0] -sample_size)

CF_resamp

#%%#

#write to file input/output pairs for resampled INC training data
INC_resamp['input'].to_csv('%s.resampleINC.input' %lang, index=False, header= False)
INC_resamp['output'].to_csv('%s.resampleINC.output' %lang, index=False, header= False)

# Generate new test file
INC_remain_tst['input'].to_csv('%s.resampleINC_tst.input' %lang, index=False, header= False)
INC_remain_tst['output'].to_csv('%s.resampleINC_tst.output' %lang, index=False, header= False)

#write to file input/output pairs for resampled CF training data
CF_resamp['input'].to_csv('%s.resampleCF.input' %lang, index=False, header= False)
CF_resamp['output'].to_csv('%s.resampleCF.output' %lang, index=False, header= False)

# Generate new test file
CF_remain_tst['input'].to_csv('%s.resampleCF_tst.input' %lang, index=False, header= False)
CF_remain_tst['output'].to_csv('%s.resampleCF_tst.output' %lang, index=False, header= False)




#%%##Concatenate original training data with the corresponding files generated above


#train files:
filenames = ['train.%s_cycle1.output'%lang, '%s.resampleCF.output' %lang]
with open('train.%s_cycle1_resampleCF.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s_cycle1.input'%lang, '%s.resampleCF.input' %lang]
with open('train.%s_cycle1_resampleCF.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
filenames = ['train.%s_cycle1.output'%lang, '%s.resampleINC.output' %lang]
with open('train.%s_cycle1_resampleINC.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s_cycle1.input'%lang, '%s.resampleINC.input' %lang]
with open('train.%s_cycle1_resampleINC.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

#test files
            
filenames = ['resample.%s_cycle1.output'%lang, '%s.resampleCF_tst.output' %lang]
with open('tst.%s_cycle1_resampleCF.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s_cycle1.input'%lang, '%s.resampleCF_tst.input' %lang]
with open('tst.%s_cycle1_resampleCF.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
            
filenames = ['resample.%s_cycle1.output'%lang, '%s.resampleINC_tst.output' %lang]
with open('tst.%s_cycle1_resampleINC.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s_cycle1.input'%lang, '%s.resampleINC_tst.input' %lang]
with open('tst.%s_cycle1_resampleINC.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())