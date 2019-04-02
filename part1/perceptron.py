import numpy as np


def extend_array(curr_arr, end_term):
	aug_arr = np.zeros((curr_arr.shape[0] + 1))
	aug_arr[:-1] = curr_arr
	aug_arr[curr_arr.shape[0]] = end_term

	return aug_arr

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model.

		This function will initialize a feature_dim weight vector,
		for each class.

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS]
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example
		"""
		bias = 1
		self.w = np.zeros((feature_dim+1,num_class))
		self.w[self.w.shape[0] - 1] = bias


	def classify(self, feats):
		feats_aug = extend_array(feats, 1)

		scores = self.w.T @ feats_aug
		return np.argmax(scores)


	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset.

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE
		# print(np.amax(train_label))
		# print(np.amin(train_label))
		total_epochs = 4
		epochs_passed = 0

		while epochs_passed < total_epochs:
			for train_idx in range(train_set.shape[0]):
				# feat_aug = extend_array(train_set[train_idx], 1)
				feats = train_set[train_idx]
				predicted_class = self.classify(feats)
				actual_class = train_label[train_idx]

				if actual_class != predicted_class:
					learn_rate = 1 / (train_idx + 1)
					feat_update = extend_array(feats, 0)
					self.w[:, actual_class] += learn_rate * feat_update
					self.w[:, predicted_class] -= learn_rate * feat_update

			epochs_passed += 1

		# print(self.w)
		# self.save_model("epoch4")


	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset.
			The accuracy is computed as the average of correctness
			by comparing between predicted label and true label.

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE
		accuracy = 0
		pred_label = np.zeros((len(test_set)))

		for pred_idx in range(test_set.shape[0]):

			feats = test_set[pred_idx]
			predicted_class = self.classify(feats)
			pred_label[pred_idx] = predicted_class

			if predicted_class == test_label[pred_idx]:
				accuracy += 1

		accuracy /= test_set.shape[0]

		print(accuracy)

		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters
		"""

		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters
		"""

		self.w = np.load(weight_file)
