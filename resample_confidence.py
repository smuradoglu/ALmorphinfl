#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:20:29 2022

@author: smuradoglu
"""

import pandas as pd
import os


lang = "nqn"
os.chdir('/home/salkazzar/Documents/Active_learning_in_morphology/data/cycle3/%s' %lang)
#%%##%% Open data files and clean lines
source = open('tst.%s.input' %lang, 'r') 
source_lines= source.readlines()
output = open('tst.%s.output' %lang, 'r')
output_lines= output.readlines()
pred = open('tst.%s.guesses' %lang, 'r')
pred_lines= pred.readlines()
loglike = open('tst.%sn.guesses' %lang, 'r')
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

print(trg_frm[2])
print(pred_frm[2])

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


#%%# Resample from 1st model test files

#Combine lists
#Get rid of nested lists
clean_source = []
for sources in src_frm:
    for source in sources:
        clean_source.append(source)

clean_output = []
for outputs in trg_frm:
    for output in outputs:
        clean_output.append(output)

clean_log = []
for logs in log_like:
    for logli in logs:
        clean_log.append(logli)
        
#create dataframe from lists        
df = pd.DataFrame(list(zip(clean_source, clean_output, clean_log)),
               columns =['input', 'output', 'loglikelihood'])
#%%# Select based on loglikelihood values ##


sample_size =250
#LOW CONFIDENCE FORMS
#sort dataframe based on loglikelihood values (ascending false for low confidence forms)
low = df.sort_values('loglikelihood', ascending=False)


LC_resamp = low.head(sample_size)
LC_remain_tst = low.tail(df.shape[0] -sample_size)

#HIGH CONFIDENCE FORMS
#sort dataframe based on loglikelihood values (ascending True for low confidence forms)

high = df.sort_values('loglikelihood', ascending=True)

HC_resamp = high.head(sample_size)
HC_remain_tst = high.tail(df.shape[0] -sample_size)

HC_resamp
#%%#

#write to file input/output pairs for resampled INC training data
LC_resamp['input'].to_csv('%s.resampleLC.input' %lang, index=False, header= False)
LC_resamp['output'].to_csv('%s.resampleLC.output' %lang, index=False, header= False)

# Generate new test file
LC_remain_tst['input'].to_csv('%s.resampleLC_tst.input' %lang, index=False, header= False)
LC_remain_tst['output'].to_csv('%s.resampleLC_tst.output' %lang, index=False, header= False)

#write to file input/output pairs for resampled CF training data
HC_resamp['input'].to_csv('%s.resampleHC.input' %lang, index=False, header= False)
HC_resamp['output'].to_csv('%s.resampleHC.output' %lang, index=False, header= False)

# Generate new test file
HC_remain_tst['input'].to_csv('%s.resampleHC_tst.input' %lang, index=False, header= False)
HC_remain_tst['output'].to_csv('%s.resampleHC_tst.output' %lang, index=False, header= False)



#%%##Concatenate original training data with the corresponding files generated above


#train files:
filenames = ['train.%s.output'%lang, '%s.resampleLC.output' %lang]
with open('train.%s_resampleLC.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s.input'%lang, '%s.resampleLC.input' %lang]
with open('train.%s_resampleLC.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
filenames = ['train.%s.output'%lang, '%s.resampleHC.output' %lang]
with open('train.%s_resampleHC.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s.input'%lang, '%s.resampleHC.input' %lang]
with open('train.%s_resampleHC.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

#test files
            
filenames = ['resample.%s.output'%lang, '%s.resampleLC_tst.output' %lang]
with open('tst.%s_resampleLC.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s.input'%lang, '%s.resampleLC_tst.input' %lang]
with open('tst.%s_resampleLC.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
            
filenames = ['resample.%s.output'%lang, '%s.resampleHC_tst.output' %lang]
with open('tst.%s_resampleHC.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s.input'%lang, '%s.resampleHC_tst.input' %lang]
with open('tst.%s_resampleHC.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())