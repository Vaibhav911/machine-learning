#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:52:53 2020

@author: vaibhav
"""

import string
import numpy as np
import pandas as pd
import random

# Text preprocessing part
with open("./../data/a1_data/a1_d3.txt") as file:
    data = pd.Series(file)

# Sentence and class of data is seperated by "\t"
data = data.str.split("\t")

# Convert a string to lowercase, remove all punctuations 
# in a sentence and split string by spaces and convert 
# it into a list of words.
data = data.apply(lambda row: [row[0]
                                .lower()
                                .translate(str.maketrans('', '', string.punctuation))
                                .split(" "), int(row[1][0])])

# Split data into approx 80% and 20% for testing and training.
msk  = np.random.rand(len(data)) < 0.8

test_data = data[~msk]
train_data = data[msk]

# Reset the index to sequence 0,1,2...
test_data.reset_index(drop=True, inplace=True)
train_data.reset_index(drop=True, inplace=True)

def create_dict(data):
    """
    For each class(i.e. class1 & class2), it creates a dictionary
    with key as words and value as their count. This further helps 
    calculating probability using naive bayes theorem.
    """
    class1_dict = {}
    class2_dict = {}
    class1_len, class2_len = 0, 0
    for row in data:
        words = row[0]
        if row[1] == 0:
            class1_len += 1
            for word in words:
                if class1_dict.get(word):
                    class1_dict[word] += 1
                else:
                    class1_dict[word] = 1
        else:
            class2_len += 1
            for word in words:
                if class2_dict.get(word):
                    class2_dict[word] += 1
                else:
                    class2_dict[word] = 1
    return class1_dict, class2_dict, class1_len, class2_len

def predict_class(sentence, class1_dict, class2_dict, class1_len, class2_len):
    """
    Given a sentence (as list of words) and dictionary of words for     
    both classes, it calculates probability of sentence belonging 
    to each class using naive bayes, and predicts the class of 
    given sentence.
    """
    # Prior probability of both classes.
    prob1 = class1_len / (class1_len + class2_len)
    prob2 = class1_len / (class1_len + class2_len)
    
    
    for word in sentence:
        # Calculates probability that word belongs to class 1
        if class1_dict.get(word):
            prob1 *= class1_dict[word] / class1_len
        else:
            # If word doesn't belong to this class, instead of 
            # multiplying with 0, penalize probability with 
            # below value.
            prob1 *= 1 / (class1_len)**2
            
        # Calculates probability that word belongs to class 2
        if class2_dict.get(word):
            prob2 *= class2_dict[word] / class2_len
        else:
            prob2 *= 1 / (class2_len)**2
    
    # Below line can be used to print details about predictions
    # of each sentence.
    #print(prob1, prob2, len(sentence), int(prob1 < prob2))
    if (prob1 >= prob2):
        return 0
    return 1

def calc_accur(data, class1_dict, class2_dict, class1_len, class2_len):
    """
    Iterates over test dataset, and calculates number of correct 
    predictions and returns ratio of correct predictions upon 
    total number of predictions.
    """
    correct_predictions = 0
    for row in data:
        if predict_class(row[0], class1_dict, class2_dict, class1_len, class2_len) == row[1]:
            correct_predictions += 1

    return correct_predictions / len(data)

def k_fold_cross_valid(data, k):
    """
    Apply k-fold validation to a given data set, and
    return accuracies of all k iterations.
    """
    # Shuffle data randomly before validation.
    random.shuffle(data)
    
    # Test set size for each iteration
    set_size = int(len(data) / k)
    
    # Output to be returned
    accuracies = []
    
    for iteration in range(k):
        #Prepare mask to seperate test and training data set
        mask = [False for i in range(len(data))]
        for index in range(set_size):
            mask[index + (set_size * iteration)] = True
            
        test_data = data[mask]
        train_data = data[~np.array(mask)]
        
        # Prepare the dictionaries to be used for predicting class
        class1_dict, class2_dict, class1_len, class2_len = create_dict(train_data)
        accuracy = calc_accur(test_data, class1_dict, class2_dict, class1_len, class2_len)
        accuracies.append(accuracy)
        
    return accuracies
    

def main():
    # Uncomment below three lines, if you don't want to use
    # k-fold validation method
    # class1_dict, class2_dict, class1_len, class2_len = create_dict(train_data)
    # accuracy = calc_accur(test_data, class1_dict, class2_dict, class1_len, class2_len)
    # print(accuracy)
    
    k = 5
    accuracies = k_fold_cross_valid(data, k)
    mean = np.array(accuracies).mean()
    std = np.array(accuracies).std()
    print(u"%s \u00B1 %s" % (round(mean, 3), round(std, 3)))
    return 0
            
if __name__ == "__main__":
    main()
            
    