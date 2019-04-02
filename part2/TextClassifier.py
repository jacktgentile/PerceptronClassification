import math

# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

# Subclass representing a category we can classify our text as
class TextCategory(object):
    laplace_const = 0.022
    def __init__(self):
        self.word_freqs = {}
        self.total_words = 0
        self.total_instances = 0

    def add_document(self, document):
        for word in document:
            if word not in self.word_freqs.keys():
                self.word_freqs[word] = 1
            else:
                self.word_freqs[word] += 1

        self.total_words += len(document)
        self.total_instances += 1

    def predict_class(self, document, total_trained):
        curr_prob = 0
        for word in document:
            if word not in self.word_freqs.keys():
                curr_prob += math.log(TextCategory.laplace_const / self.total_words)
            else:
                curr_prob += math.log(self.word_freqs[word] / self.total_words)

        return curr_prob + math.log(self.total_instances / total_trained)

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0
        self.categories = {}
        self.num_trained = 0

    # Trains our model
    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """
        # TODO: Write your code here
        self.classes = set(train_label)
        self.num_trained += len(train_set)

        for categ in self.classes:
            self.categories[categ] = TextCategory()

        for i in range(len(train_set)):
            self.categories[train_label[i]].add_document(train_set[i])

    # Classifies a single document
    def classify(self, document):

        # Iterate through each class and the class with the maximum probability
        max_prob = None
        max_class = None

        for categ in self.classes:
            class_prob = self.categories[categ].predict_class(document, self.num_trained)

            if max_prob is None or max_prob < class_prob:
                max_prob = class_prob
                max_class = categ

        return max_class


    def predict(self, x_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """
        accuracy = 0.0
        result = []

        num_correct = 0


        for x_idx in range(len(x_set)):
            predicted_class = self.classify(x_set[x_idx])

            if dev_label[x_idx] == predicted_class:
                num_correct += 1

            result.append(predicted_class)

        accuracy = num_correct / len(x_set)

        return accuracy,result
