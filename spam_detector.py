#!/usr/bin/env python
# coding: utf-8

# In[278]:


import os;
import numpy as np;
import re;


# In[279]:


vocab = []
freq_ham=[]
freq_spam=[]
sizeOfVocab=0
conditionalProbOfHam=[]
conditionalProbOfSpam=[]


# Number of ham files
pattern = re.compile(r'train-ham')
filesInDir = [f for f in os.listdir('/Users/apple/Desktop/AI-COMP 6721/AI_Proj/Project2/train/') if re.match(pattern, f)]
noOfHamFiles = len(filesInDir)

# Number of ham files
pattern = re.compile(r'train-spam')
filesInDir = [f for f in os.listdir('/Users/apple/Desktop/AI-COMP 6721/AI_Proj/Project2/train/') if re.match(pattern, f)]
noOfSpamFiles = len(filesInDir)

# Number of total files
noOfTotalFiles = noOfHamFiles+noOfSpamFiles

#prior probability of ham p(ham)
priorProbOfHam=noOfHamFiles/noOfTotalFiles

#prior probability of spam p(spam)
priorProbOfSpam=noOfSpamFiles/noOfTotalFiles


# In[280]:


def word_count_directory(directory):
    filelist=[os.path.join(directory,f) for f in os.listdir(directory)]
    filelist.sort()
    sizeOfHamWords=0
    sizeOfSpamWords=0
    for i in filelist:
        print(i)
        filetext = open(i, "rb") 
        for line in filetext:
         #   print(line)
            line=str(line)
            line = line.strip() 
            line = line.lower()
            words=re.split('[^a-zA-Z]',line)
            words=list(filter(None, words))
     #       print(line)
     #       print(words)
  
            for word in words: 
        
                if word in vocab:
                
                    indexOfElement = vocab.index(word)
                    
                    if("train-ham" in i):
                        sizeOfHamWords=sizeOfHamWords+1
                        freq_ham[indexOfElement] +=1
                        freq_spam[indexOfElement] +=0
                    else:
                        sizeOfSpamWords=sizeOfSpamWords+1
                        freq_ham[indexOfElement] +=0
                        freq_spam[indexOfElement] +=1
                
                else:
                    
                    vocab.append(word)
                    
                    if("train-ham" in i):
                        sizeOfHamWords=sizeOfHamWords+1
                        freq_ham.append(1)
                        freq_spam.append(0)
                        
                    else:
                        sizeOfSpamWords=sizeOfSpamWords+1
                        freq_ham.append(0)
                        freq_spam.append(1)
                    
    #            print(word)
    #            print(vocab)
    #            print(freq_spam)
    #            print(freq_ham)
              
     #           print("\n")
    
    sizeOfVocab = len(vocab)


# In[284]:


word_count_directory("/Users/apple/Desktop/AI-COMP 6721/AI_Proj/Project2/train/")


# In[285]:


sizeOfVocab=len(vocab)
frequency_ham=np.array(freq_ham)
frequency_spam=np.array(freq_spam)
print(frequency_spam)

sizeOfHamWords2=np.sum(frequency_ham)
sizeOfSpamWords2=np.sum(frequency_spam)
print(frequency_ham.shape)
print(frequency_spam.shape)
print(sizeOfHamWords)
print(sizeOfSpamWords)
print(sizeOfHamWords2)
print(sizeOfSpamWords2)

# add Smoothing  δ = 0.5
frequency_ham =frequency_ham+0.5
frequency_spam=frequency_spam+0.5


# In[286]:


sizeOfVocab


# In[287]:


print(frequency_ham)


# In[288]:


print(frequency_spam)


# In[289]:


print(sizeOfVocab)
print(priorProbOfHam)
print(priorProbOfSpam)


# In[290]:


conditionalProbOfHam=[]
conditionalProbOfSpam=[]

for i in range(len(vocab)):
    p_wj_ham= frequency_ham[i]/(sizeOfHamWords2+sizeOfVocab)
    p_wj_spam=frequency_spam[i]/(sizeOfSpamWords2+sizeOfVocab)
  #  p_wj_ham = np.round(p_wj_ham, decimals=4)
  #  p_wj_spam = np.round(p_wj_spam, decimals=4)
    conditionalProbOfHam.append(p_wj_ham)
    conditionalProbOfSpam.append(p_wj_spam)
    
    
print(conditionalProbOfHam)


# In[291]:


print(conditionalProbOfSpam)


# In[309]:


sortedVocab=vocab.copy()
sortedVocab.sort()
f = open("model.txt", "w")   # 'r' for reading and 'w' for writing
f.truncate(0)
  
for counter in range(len(sortedVocab)):
    indexOfWord=vocab.index(sortedVocab[counter])
    modelText= str(counter)
    spaceTwo=str("  ")
    word=str(sortedVocab[counter])
    freqInham=str(frequency_ham[counter]-0.5)    #  freqInham=str((frequency_ham[counter]-0.5)) 
    probInham=str(conditionalProbOfHam[counter])
    freqInspam=str(frequency_spam[counter]-0.5)    #  freqInham=str((frequency_ham[counter]-0.5)) 
    probInspam=str(conditionalProbOfSpam[counter])
    f.write(str(modelText+spaceTwo+word+spaceTwo+freqInham+spaceTwo+probInham+spaceTwo+freqInspam+spaceTwo+probInspam+"\n"))
    
    
f.close() 

