#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os;


# In[15]:


def word_count_directory(directory):
    filelist=[os.path.join(directory,f) for f in os.listdir(directory)]
    print(type(filelist))
    filelist.sort()
    for i in filelist:
        print(i)     
        
    # Open the file in read mode 
    text = open("sample.txt", "r") 
  
    # Create an empty dictionary 
    d = dict()
  
    # Loop through each line of the file 
    for line in text: 
        # Remove the leading spaces and newline character 
        line = line.strip() 
  
        # Convert the characters in line to  
        # lowercase to avoid case mismatch 
        line = line.lower() 
  
        # Split the line into words 
        words = line.split(" ") 
  
        # Iterate over each word in line 
        for word in words: 
            # Check if the word is already in dictionary 
            if word in d: 
                # Increment count of word by 1 
                ham[word] = d[word] + 1
                
            else: 
                # Add the word to dictionary with count 1 
                d[word] = 1
  
        # Print the contents of dictionary 
    
    for key in list(d.keys()): 
        print(key, ":", d[key]) 
   

    tokenizer = RegexpTokenizer('\s+', gaps=True)
    x=re.split('[^a-zA-Z]',aString)
    s = "one two 3.4 5,6 seven.eight nine,ten"


# In[16]:


word_count_directory("/Users/apple/Desktop/AI-COMP 6721/AI_Proj/Project2/train/")


# In[ ]:




