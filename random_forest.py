"""
Blake Masters
bemaster@calpoly.edu
David Cho
dcho08@calpoly.edu
"""


from c45 import c45, c45Node
import numpy as np
import pandas as pd
import random
from collections import Counter
#from modified_csv import csv_handler
class random_forest:
    def __init__(self, num_attributes = 0, num_data_points = 0, num_trees = 10, split_metric="Gain", threshold=0.1, attribute_types=None):
        self.forest = []
        self.num_data_points = num_data_points #k in notes
        self.num_attributes = num_attributes #m in notes
        self.num_trees = num_trees #n in notes
        self.threshold = threshold #default to 0.1
        self.split_metric = split_metric  #"Gain" or "Ratio" from my c45. defaulting to Gain
        
        self.attribute_types = attribute_types if attribute_types is not None else {} #it needs this for numeric vs categorical

    

    def create_decision_tree(self, training, truth, chosen_attr_types):
        tree = c45(split_metric=self.split_metric, threshold=self.threshold, attribute_types=chosen_attr_types)
        tree.fit(training, truth)
        return tree
    
    def fit(self, training_set, truth): #X, Y #mine uses the label of the truth
        self.forest = []
        all_attributes = [col for col in training_set.columns if col != truth]
        if self.num_attributes > len(all_attributes):
            raise ValueError("num_attributes requested is greater than the number of available attributes")
        for i in range(self.num_trees):
            if self.num_data_points < 1:
                num_points = int(self.num_data_points * len(training_set))
            else:
                num_points = self.num_data_points
            data_sample = training_set.sample(n=num_points, replace=True)
            #randomly sample data/attributes w/replacement np.random
            chosen_features = np.random.choice(all_attributes, size=self.num_attributes, replace=False)
            chosen_features = list(chosen_features)
            chosen_attr_types = {feature: self.attribute_types[feature] for feature in chosen_features}
            
            subset = data_sample[chosen_features + [truth]]
            tree = self.create_decision_tree(subset, truth, chosen_attr_types)
            self.forest.append(tree)
        
        
        """The .fit() method. The .fit() method takes as input two parameters, X- the training set, and Y- the ground truth. It creates a random forest 
        consisting of NumTree decision trees, each tree created by randomly sampling the data with replacement, and randomly sampling the attributes 
        without replacement (using the NumDataPoints and NumAttributes values for guidance), and built by a call to C45 with the inputs created. 
        In implementing .fit() you may build helper functions that perform individual tree creation. Your instance of the RandomForest class shall 
        have a class variable or variables that store the built model (a forest of decision tree) for further use"""
    
    def majority_vote(self, votes):
        """return majority, else in case of a tie, return the smallest label (lexicographically or numerically)."""
        counts = Counter(votes)
        max_votes = max(counts.values())
        candidates = [vote for vote, count in counts.items() if count == max_votes]
        return sorted(candidates)[0]
    
    def predict(self, x_test):
        """
        For each sample in x_test, each tree in the forest produces a prediction.
        The final prediction is determined by majority vote across trees.
        x_test is expected to have all the original features. Each tree will
        only use the features it was trained on.
        """
        all_tree_preds = [tree.predict(x_test) for tree in self.forest]
        num_samples = len(x_test)
        combined_preds = []
        for i in range(num_samples):
            votes = [preds[i] for preds in all_tree_preds]
            combined_pred = self.majority_vote(votes)
            combined_preds.append(combined_pred)
        return combined_preds
            
        """The .predict() method. The .predict() method takes as input one parameter, X- the test set, and outputs a vector of predictions- 
        one prediction per observation/row in X. The .predict() method essentially acts as a wrapper around the calls to the c45.predict() on each decision tree 
        that forms your random forest. After each decision tree reports its prediction on a given data point (or all its predictions on all data points from X), 
        the RandomForest.predict() shall combine them and form a single prediction for each input row of data. Any ties shall be resolved in an arbitrary but consistent way 
        (e.g., by selecting the smaller (numerically, or lexicographically) label)."""
        


            
        
        
        
        