#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:50:03 2022

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

print(trg_frm[51])
print(pred_frm[51])

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

# Sample N randomly
sample_size = 107
rand_sample = df.sample(n = sample_size)
rand_remain = df.drop(rand_sample.index)

rand_remain
#%%#

#write to file input/output pairs for resampled training data
rand_sample['input'].to_csv('%s.resamplerand1.input' %lang, index=False, header= False)
rand_sample['output'].to_csv('%s.resamplerand1.output' %lang, index=False, header= False)

# Generate new test file
rand_remain['input'].to_csv('%s.resamplerand1_tst.input' %lang, index=False, header= False)
rand_remain['output'].to_csv('%s.resamplerand1_tst.output' %lang, index=False, header= False)


#%%##Concatenate original training data with the corresponding files generated above

#train files:

#rand1
filenames = ['train.%s.output'%lang, '%s.resamplerand1.output' %lang]
with open('train.%s_resamplerand1.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s.input'%lang, '%s.resamplerand1.input' %lang]
with open('train.%s_resamplerand1.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
            
#rand2
filenames = ['train.%s.output'%lang, '%s.resamplerand2.output' %lang]
with open('train.%s_resamplerand2.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s.input'%lang, '%s.resamplerand2.input' %lang]
with open('train.%s_resamplerand2.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
            
#rand3
filenames = ['train.%s.output'%lang, '%s.resamplerand3.output' %lang]
with open('train.%s_resamplerand3.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s.input'%lang, '%s.resamplerand3.input' %lang]
with open('train.%s_resamplerand3.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

#test files
            
#rand1
filenames = ['resample.%s.output'%lang, '%s.resamplerand1_tst.output' %lang]
with open('tst.%s_resamplerand1.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s.input'%lang, '%s.resamplerand1_tst.input' %lang]
with open('tst.%s_resamplerand1.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

#rand2           
filenames = ['resample.%s.output'%lang, '%s.resamplerand2_tst.output' %lang]
with open('tst.%s_resamplerand2.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s.input'%lang, '%s.resamplerand2_tst.input' %lang]
with open('tst.%s_resamplerand2.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

#rand3
filenames = ['resample.%s.output'%lang, '%s.resamplerand3_tst.output' %lang]
with open('tst.%s_resamplerand3.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s.input'%lang, '%s.resamplerand3_tst.input' %lang]
with open('tst.%s_resamplerand3.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())            
            
            
            
            
            