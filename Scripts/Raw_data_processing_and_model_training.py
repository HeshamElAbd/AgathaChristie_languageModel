#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:47:25 2019

@author: Hesham El Abd

@Description: This script starts by reading the raw data and then prepare it 
as a long string, next, the text is mapped into integers and two maps are 
created the first is char2idx which maps characters into integers and the second
is idx2char which maps back integers two characters. Next, the data is prepared 
using TF.Data API and finally the model is created and trained on the generated 
datasets. 
"""
# load modules: 
import numpy as np
import pickle
from utility_functions import( EncodeText, PrepareTrainingTensors, 
                              CreateModels,loss)
import argparse
import tensorflow as tf
### define the user specific parameters:
parser=argparse.ArgumentParser()

parser.add_argument("--embedding_dimension", 
                    help="The embedding dimentionalility for building"+
                    "the embedding layer of the model",
                    type=int)

parser.add_argument("--num_of_lstm_units", 
                    help="The number of units to be used to create the"+
                    "LSTM",
                    type=int)

parser.add_argument("--recurrent_dropout",
                    help="Recurrent dropout between the LSTM time steps",
                    type=float)

parser.add_argument("--input_dropout",
                    help="The dropout applied to the inpt of the LSTM unit",
                    type=float)

parser.add_argument("--cond_str_len",
                help="length of conditional string to train the model on",
                type=int)

parser.add_argument("--batch_size",
                    help="the batch size used for training the model",
                    type=int
                   )

parser.add_argument("--output",
                help="The output directory to save the model after training")

parser.add_argument("--epochs",
                    help="number of training epoches",
                    type=int)

# Parsing the user inputs
user_inputs=parser.parse_args()
seq_len=user_inputs.cond_str_len
emb_dim=user_inputs.embedding_dimension
num_units=user_inputs.num_of_lstm_units
recurrent_dropout=user_inputs.recurrent_dropout
input_dropout=user_inputs.input_dropout
batch_size=user_inputs.batch_size
num_epoches=user_inputs.epochs
output_dir=user_inputs.output

# Prepare the data as a long string: 
raw_text=""
with open("../data/raw.txt","r") as openfile:
    for line in openfile: 
        raw_text+=" \n "+line
raw_text=raw_text.strip()

## mapping: 
text_vocab=sorted(set(raw_text))
char2idx={char:idx for idx, char in enumerate(text_vocab)}
idx2char= np.array(text_vocab)

## save the maps for subsequent uses : 
with open("../Resources/char2idx.pickle","wb") as outputfile:
    pickle.dump(char2idx,outputfile)

with open("../Resources/idx2char.pickle","wb") as outputfile:
    pickle.dump(idx2char,outputfile)

## Encoding the text numerically: 
text_encoded=EncodeText(text=raw_text, encoding_scheme=char2idx)

## prepare the data as a TF DataSet
text_as_chuncks=tf.data.Dataset.from_tensor_slices(text_encoded).batch(
        seq_len+1,drop_remainder=True)
## preapre the TF data for training 
train_dataset=text_as_chuncks.map(PrepareTrainingTensors).shuffle(
        10000).batch(batch_size,drop_remainder=True)

## Create the model: 
model=CreateModels(vocab_size=len(text_vocab), emb_dim=emb_dim,
                   num_lstm_units= num_units, 
                   recurrent_dropout=recurrent_dropout,
                   input_dropout=input_dropout,batch_size=batch_size)
# def loss function
    
model.compile(optimizer="Adam",
              loss=loss) 
print(model.summary())

# train the model 
model.fit(train_dataset, epochs=num_epoches)
# save the model
model.save(output_dir)
print("The Model has been trained for {} and saved at {}".format(num_epoches,output_dir))
