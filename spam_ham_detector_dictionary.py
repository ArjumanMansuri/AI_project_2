#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import re
import sys
import matplotlib
import os


# In[3]:


directory = "C:/Users/emnsrjm/Desktop/Courses/AI/Project/Assignment_2/sample_data"
vocabulary = set()


# In[4]:


hamDictionary = {}
spamDictionary = {}
vocabulary = set()
def word_count_directory(directory):
    filelist=[os.path.join(directory,f) for f in os.listdir(directory)]
    
    for file_path in filelist:
        with open(file_path) as infile:
            # to store type of file 'spam' or 'ham'
            file_type = ''
            if 'spam' in file_path:
                file_type = 'spam'
            elif 'ham' in file_path:
                file_type = 'ham'
            # Loop through each line of the file 
            for line in infile:
                # Remove the leading spaces and newline character 
                line = line.strip()
                # Convert the characters in line to lowercase to avoid case mismatch 
                lowerLine = str.lower(line)
                validWords = re.split('[^a-zA-Z]',lowerLine)
                # Iterate over each word in line 
                for word in validWords:
                    # if the word is not an empty space
                    if len(word) > 0:
                        
                        if file_type=='ham':
                            # Check if the word is already in dictionary
                            if word in hamDictionary:
                                hamDictionary[word] += 1
                            else:
                                # add word to dictionary with count 1
                                hamDictionary[word] = 1
                                # add word to vocabulary set
                                vocabulary.add(word)
                                # if this word is not present in spamDictionary, add it with count 0
                                if word not in spamDictionary:
                                    spamDictionary[word] = 0
                                
                        elif file_type=='spam':
                            # Check if the word is already in dictionary
                            if word in spamDictionary:
                                spamDictionary[word] += 1
                            else:
                                # add word to dictionary with count 1
                                spamDictionary[word] = 1
                                # add word to vocabulary set
                                vocabulary.add(word)
                                # if this word is not present in hamDictionary, add it with count 0
                                if word not in hamDictionary:
                                    hamDictionary[word] = 0
    
word_count_directory(directory)


# In[15]:


def create_model(vocabulary,hamDictionary,spamDictionary):
    # sorting the vocabulary to maintain order in model.txt
    vocabulary = sorted(vocabulary)
    # creating file that would store the model
    f= open("model.txt","w+")
    # getting size of vocabulary
    N = len(vocabulary)
    # smoothing value
    delta = 0.5
    smoothed_N = (delta * N)
    # calculating smoothed denominator while calculating condinational probability of ham words
    ham_denominator = sum(hamDictionary.values()) + smoothed_N
    # calculating smoothed denominator while calculating condinational probability of spam words
    spam_denominator = sum(spamDictionary.values()) + smoothed_N
    
    for i,word in enumerate(vocabulary):
        
        # frequency of word in ham dictionary
        freq_in_ham = hamDictionary[word]
        # conditional probabiltiy of word in ham
        c_p_in_ham = (freq_in_ham + delta) / ham_denominator

        # frequency of word in spam dictionary
        freq_in_spam = spamDictionary[word]
        # conditional probabiltiy of word in spam
        c_p_in_spam = (freq_in_spam + delta) / spam_denominator
        
        # writing all the data to model.txt
        f.write(str(i+1)+'  '+word+'  '+str(freq_in_ham)+'  '+str(c_p_in_ham)+'  '+str(freq_in_spam)+'  '+str(c_p_in_spam)+'\n')
        
    # closing the file
    f.close()
create_model(vocabulary,hamDictionary,spamDictionary)


# In[ ]:





# In[ ]:




