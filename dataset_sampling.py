#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:13:23 2022

@author: smuradoglu
"""
import random
import os
#%%##%% Open data files and clean lines

# Using readlines()

lang = "cly"
os.chdir('/home/salkazzar/Documents/Active_learning_in_morphology/data/cycle3/%s' %lang)
pairs = open('raw/%s' %lang, 'r')
lines = pairs.readlines()

data=[]
for i in range(len(lines)):
    data.append(lines[i].rstrip('\n').split('\t'))


print(data[1])

#%%#collate all unique lexemes from data
lex =[]

for j in range(len(data)):
    triplet = data[j]
    lex.append(triplet[0])
print(len(lex))


unique_lex = set(lex)
lexeme =list(unique_lex)
print(len(lexeme))
#%%# sample n lexemes and the corresonding tags and target forms

n=185
random.shuffle(lexeme)
lexeme_sub=lexeme[0:n]
triplet_datauc = []

#Sample from full data set according to chosen n lexemes
for k in range(len(data)):
    triplet = data[k]
    if triplet[0] in lexeme_sub:
        triplet_datauc.append(data[k])

triplet_data = [list(x) for x in set(tuple(x) for x in triplet_datauc)]

print(len(triplet_data))
#%%#
#randomly split triplet data into source form, tags and target form (in respective order)    
        
random.shuffle(triplet_data)
tags =[]
trg_frm =[]
src_frm =[]

for n in range(len(triplet_data)):
    tri = triplet_data[n]
    src_frm.append(tri[0])
    trg_frm.append(tri[1])
    tags.append(tri[2])
        
print(len(tags))
print(len(trg_frm))
print(len(src_frm))

print(src_frm[1] + "  " + tags[1] + " " + trg_frm [1])
print(triplet_data[1])
#%%# Format according to fairseq requirements (introduce space between characters and '#')


src_frm_sp =[]
trg_frm_sp =[]

for i in range(len(trg_frm)):
    trg_frm_sp.append(" ".join(trg_frm[i]))
                      
for i in range(len(src_frm)):
    src_frm_sp.append(" ".join(src_frm[i]) + " # ")
                 
input_source =  [i + j for i, j in zip(src_frm_sp, tags)]

print(len(src_frm))                                 
                         

#%%# Split data into train, test and dev sets
train_input = input_source[0:1800]
train_output = trg_frm_sp[0:1800]
test_input = input_source[1800:2314]
test_output =  trg_frm_sp[1800:2314]
dev_input =  input_source[2314:2571]
dev_output = trg_frm_sp[2314:2571]

resample_input = input_source[2571:2700]
resample_output = trg_frm_sp[2571:2700]


print(len(train_input))
print(len(test_input))
print(len(dev_input))
print(len(resample_input))
#%%##%% Save input and output files

output=open('train.%s.input' %lang,'w')

for element in train_input:
     output.write(element)
     output.write('\n')
output.close()


output=open('train.%s.output' %lang,'w')

for element in train_output:
     output.write(element)
     output.write('\n')
output.close()

output=open('tst.%s.input'%lang,'w')

for element in test_input:
     output.write(element)
     output.write('\n')
output.close()


output=open('tst.%s.output'%lang,'w')

for element in test_output:
     output.write(element)
     output.write('\n')
output.close()
output=open('dev.%s.input'%lang,'w')

for element in dev_input:
     output.write(element)
     output.write('\n')
output.close()


output=open('dev.%s.output'%lang,'w')

for element in dev_output:
     output.write(element)
     output.write('\n')
output.close()

output=open('resample.%s.input'%lang,'w')

for element in resample_input:
     output.write(element)
     output.write('\n')
output.close()


output=open('resample.%s.output'%lang,'w')

for element in resample_output:
     output.write(element)
     output.write('\n')
output.close()

