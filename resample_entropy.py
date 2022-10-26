#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:53:31 2022

@author: smuradoglu
"""
import pandas as pd
import os
import numpy as np
#%%#

lang = "tur"
os.chdir('/home/salkazzar/Documents/Active_learning_in_morphology/iterative/%s' %lang)
#%%##%% Open data files and clean lines
source = open('tst.%s_cycle10_resampleLH.input' %lang, 'r') 
source_lines= source.readlines()
output = open('tst.%s_cycle10_resampleLH.output' %lang, 'r')
output_lines= output.readlines()
pred = open('tst.%s_cycle10_resampleLH.guesses' %lang, 'r')
pred_lines= pred.readlines()
loglike = open('tst.%s_cycle10_resampleLHn5.guesses' %lang, 'r')
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
n1 = log_like[0::5]
n2 = log_like[1::5]
n3 = log_like[2::5]
n4 = log_like[3::5]
n5 = log_like[4::5]

clean_n1 = []
for n1s in n1:
    for l in n1s:
        clean_n1.append(float(l))

clean_n2 = []
for n2s in n2:
    for n in n2s:
        clean_n2.append(float(n))
        
clean_n3 = []
for n3s in n3:
    for m in n3s:
        clean_n3.append(float(m))
        
clean_n4 = []
for n4s in n4:
    for p in n4s:
        clean_n4.append(float(p))

clean_n5 = []
for n5s in n5:
    for q in n5s:
        clean_n5.append(float(q))
        
#%%#

#create dataframe from lists        
df = pd.DataFrame(list(zip(clean_n1, clean_n2, clean_n3, clean_n4, clean_n5)),
               columns =['n1', 'n2', 'n3', 'n4', 'n5'])
#take exponential of all ns
df=df.apply(np.exp)

#calculate sums for each instances and normalize to that sum
df['sums']= df.sum(axis=1)

df["n1"] = df["n1"]/df["sums"]
df["n2"] = df["n2"]/df["sums"]
df["n3"] = df["n3"]/df["sums"]
df["n4"] = df["n4"]/df["sums"]
df["n5"] = df["n5"]/df["sums"]

#Calculate each entropy
df["h1"] = -1*df["n1"]*df['n1'].apply(np.log)
df["h2"] = -1*df["n2"]*df['n2'].apply(np.log)
df["h3"] = -1*df["n3"]*df['n3'].apply(np.log)
df["h4"] = -1*df["n4"]*df['n4'].apply(np.log)
df["h5"] = -1*df["n5"]*df['n5'].apply(np.log)

df

#%%#
prob = df.to_numpy()

h=[]

for x in range(0,1000):
    ent = 0
    p_sum =0
    i=0
    j=6
    while p_sum <=0.95:
        p_sum += prob[x][i]
        ent += prob[x][j]
        i += 1
        j += 1
    h.append(ent)

print(len(h))

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
        
#create dataframe for sampling       
sf = pd.DataFrame(list(zip(clean_source, clean_output, clean_guess, h)),
               columns =['input', 'output', 'guess', 'entropy'])

sf
#%%# Select based on loglikelihood values ##


sample_size =250
#LOW ENTROPY FORMS
#sort dataframe based on H values (ascending true for lowest entropy values first)
lowH = sf.sort_values('entropy', ascending=True)


LH_resamp = lowH.head(sample_size)
LH_remain_tst = lowH.tail(df.shape[0] -sample_size)

lowH
#%%#
#HIGH ENTROPY FORMS
#sort dataframe based on H values (ascending false for highest entropy values first)
sample_size =250
highH = sf.sort_values('entropy', ascending=False)

HH_resamp = highH.head(sample_size)
HH_remain_tst = highH.tail(df.shape[0] -sample_size)

HH_resamp
#%%#

#write to file input/output pairs for resampled INC training data
LH_resamp['input'].to_csv('%s_cycle10_resampleLH.input' %lang, index=False, header= False)
LH_resamp['output'].to_csv('%s_cycle10_resampleLH.output' %lang, index=False, header= False)

# Generate new test file
LH_remain_tst['input'].to_csv('%s_cycle10_resampleLH_tst.input' %lang, index=False, header= False)
LH_remain_tst['output'].to_csv('%s_cycle10_resampleLH_tst.output' %lang, index=False, header= False)
#%%#
#write to file input/output pairs for resampled CF training data
HH_resamp['input'].to_csv('%s_cycle10_resampleHH.input' %lang, index=False, header= False)
HH_resamp['output'].to_csv('%s_cycle10_resampleHH.output' %lang, index=False, header= False)

# Generate new test file
HH_remain_tst['input'].to_csv('%s_cycle10_resampleHH_tst.input' %lang, index=False, header= False)
HH_remain_tst['output'].to_csv('%s_cycle10_resampleHH_tst.output' %lang, index=False, header= False)

#%%##Concatenate original training data with the corresponding files generated above


#train files:
filenames = ['train.%s_cycle9_resampleLH.output'%lang, '%s_cycle10_resampleLH.output' %lang]
with open('train.%s_cycle10_resampleLH.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s_cycle9_resampleLH.input'%lang, '%s_cycle10_resampleLH.input' %lang]
with open('train.%s_cycle10_resampleLH.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
#%%#
filenames = ['train.%s_cycle9_resampleHH.output'%lang, '%s_cycle10_resampleHH.output' %lang]
with open('train.%s_cycle10_resampleHH.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['train.%s_cycle9_resampleHH.input'%lang, '%s_cycle10_resampleHH.input' %lang]
with open('train.%s_cycle10_resampleHH.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
#%%#
#test files
            
filenames = ['resample.%s_cycle9.output'%lang, '%s_cycle10_resampleLH_tst.output' %lang]
with open('tst.%s_cycle10_resampleLH.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s_cycle9.input'%lang, '%s_cycle10_resampleLH_tst.input' %lang]
with open('tst.%s_cycle10_resampleLH.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
#%%#            
filenames = ['resample.%s_cycle9.output'%lang, '%s_cycle10_resampleHH_tst.output' %lang]
with open('tst.%s_cycle10_resampleHH.output' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())

filenames = ['resample.%s_cycle9.input'%lang, '%s_cycle10_resampleHH_tst.input' %lang]
with open('tst.%s_cycle10_resampleHH.input' %lang, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())