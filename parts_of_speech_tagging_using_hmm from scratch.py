# -*- coding: utf-8 -*-
"""parts of speech tagging using HMM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13gKezoaqC1p2YHl0IszwSfS80sPLIijf
"""

# Importing libraries
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time

#download the treebank corpus from nltk
nltk.download('treebank')
 
#download the universal tagset from nltk
nltk.download('universal_tagset')
 
# reading the Treebank tagged sentences
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
 
#printing the first two sentences along with tags
print(nltk_data[:2])


for sent in nltk_data[:2]:
  for tuple in sent:
    print(tuple)

# split data into training and validation set in the ratio 80:20
train_set,test_set =train_test_split(nltk_data,train_size=0.80,test_size=0.20,random_state = 101)

# creating a list of test tagged words & trains
trainTags = [ tup for sent in train_set for tup in sent ]
testTags = [ tup for sent in test_set for tup in sent ]
print(len(trainTags))
print(len(testTags))
# check some of the tagged words.
trainTags[:5]

#using this datatype to check how many unique tags are present in training data
tags = {tag for word,tag in trainTags}
print(len(tags))
print(tags)
 
# check total words in vocabulary
vocab = {word for word,tag in trainTags}

#building the HMM model
# compute Emission Probability
def word_given_tag(word, tag, train_bag = trainTags):
    tagList = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tagList)#total number of times the passed tag occurred in train_bag
    w_given_tagList = [pair[0] for pair in tagList if pair[0]==word]
#now we are calculating the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tagList)
    return (count_w_given_tag, count_tag)

# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag = trainTags):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# creating t x t transition matrix of tags where t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
 
tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
 
print(tags_matrix)

#same as the transition table
tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
display(tags_df)


def Viterbi(words, train_bag = trainTags):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
     
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                 
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
             
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))

#testing model
random.seed(22) 
 
# choose random 10 numbers
rndom = [random.randint(1,len(test_set)) for x in range(10)]
 
# list of sentences for training
test_run = [test_set[i] for i in rndom]
 
# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]
 
# list of untagged words
testTags = [tup[0] for sent in test_run for tup in sent]


#Here We will only test 10 sentences to check the accuracy
#as testing the whole training set takes huge amount of time
start = time.time()
tagged_seq = Viterbi(testTags)
end = time.time()
difference = end-start
 
print("Time taken in seconds: ", difference)
 
# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
 
accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)