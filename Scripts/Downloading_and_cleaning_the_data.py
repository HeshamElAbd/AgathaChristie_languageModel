#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 05:49:16 2019
@author: Hesham El Abd
@description: The scripts downloads five books from gutenberg project, 
clean them and then write a one raw text file that will be used to train an 
RNN based language model for Agatha christie. 
"""
# loading the modules:
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import re
import os
## making a directory to save the data: 
os.mkdir("data")
# define the etextno list 
book_list=[58866,1155,863,22820]

# define the number of line to remove from the headers
cleanup_list=[980,1300,485, 2242]

# a list to store the results 
text_corpa =[]
## doing the initial cleaning by:
# 1- Downloading the book
# 2- split them into lines
# 3- clean the lines from anything that is not a character, number of
# punctuation marks
# 4- transfear the text from unicode into ASCII
for bk, cl in zip(book_list,cleanup_list): 
    dum_text=strip_headers(load_etext(bk)).strip().split("\n")[cl:]
    dum_list=[]
    for line in dum_text:
        if not(line ==""):
            dum_list.append(re.sub(
                    '[^A-Za-z0-9.,:;!?]+',' ', line).encode("ascii").strip())
    text_corpa.extend(dum_list)

# writing the text corpa as a raw file data: 
with open("data/raw.txt","w") as raw_writer:
    for line in text_corpa: 
        raw_writer.write(line+"\n")
    
    

