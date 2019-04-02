import numpy as np
import matplotlib.pyplot as plt

def plot_visualization(images, classes, cmap):
    """Plot the visualizations
    """
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax[i%2, i//2].imshow(images[:, i].reshape((28, 28)), cmap=cmap)
        ax[i%2, i//2].set_xticks([])
        ax[i%2, i//2].set_yticks([])
        ax[i%2, i//2].set_title(classes[i])
    plt.show()

class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""Initialize a naive bayes model.

		This function will initialize prior and likelihood, where
		prior is P(class) with a dimension of (# of class,)
			that estimates the empirical frequencies of different classes in the training set.
		likelihood is P(F_i = f | class) with a dimension of
			(# of features/pixels per image, # of possible values per pixel, # of class),
			that computes the probability of every pixel location i being value f for every class label.

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example
		    num_value(int): number of possible values for each pixel
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim

		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

	def train(self,train_set,train_label):
		""" Train naive bayes model (self.prior and self.likelihood) with training dataset.
			self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
			self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of
				(# of features/pixels per image, # of possible values per pixel, # of class).
			You should apply Laplace smoothing to compute the likelihood.

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""
		count = np.zeros((self.num_class))
		for label in train_label:
			count[label] += 1.0
		self.prior = count / len(train_label)

		# calculate likelihood using Laplace smoothing
		k = 0.01
		kV = k * self.num_value
		for i in range(self.num_class):
			count[i] += kV
			for r in range(self.feature_dim):
				for s in range(self.num_value):
					self.likelihood[r][s][i] = k / count[i]

		for i in range(len(train_set)):
			cur_label = train_label[i]
			cur_img = train_set[i]
			for j in range(self.feature_dim):
				self.likelihood[j][cur_img[j]][cur_label] += 1.0 / (count[cur_label])

	def test(self,test_set,test_label):
		""" Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
			by performing maximum a posteriori (MAP) classification.
			The accuracy is computed as the average of correctness
			by comparing between predicted label and true label.

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""
		pred_label = np.zeros((len(test_set)))
		highest_probs = np.zeros((self.num_class, 2))
		lowest_probs = np.zeros((self.num_class, 2))
		for i in range(len(test_set)):
			cur_img = test_set[i]
			prob_arr = np.log(self.prior)
			for j in range(self.num_class):
				for k in range(self.feature_dim):
					prob_arr[j] += np.log(self.likelihood[k][cur_img[k]][j])
			maxdex = 0
			for j in range(1,self.num_class):
				if prob_arr[j] > prob_arr[maxdex]:
					maxdex = j
			pred_label[i] = maxdex
			# find indices of highest and lowest posterior probabilities
			if pred_label[i] == test_label[i]:
				if highest_probs[maxdex][1] == 0 or prob_arr[maxdex] > highest_probs[maxdex][1]:
					highest_probs[maxdex][1] = prob_arr[maxdex]
					highest_probs[maxdex][0] = i
				elif lowest_probs[maxdex][1] == 0 or prob_arr[maxdex] < lowest_probs[maxdex][1]:
					lowest_probs[maxdex][1] = prob_arr[maxdex]
					lowest_probs[maxdex][0] = i
        # display images with highest and lowest posterior probability
		class_names = np.array(["T-shirt/top","Trouser","Pullover","Dress",
	        "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
		best_imgs = np.zeros((self.feature_dim, self.num_class))
		worst_imgs = np.zeros((self.feature_dim, self.num_class))
		for i in range(self.num_class):
			best_imgs[:,i] = test_set[highest_probs[i][0]]
			worst_imgs[:,i] = test_set[lowest_probs[i][0]]
		# plot_visualization(best_imgs, class_names, "Greys")
		# plot_visualization(worst_imgs, class_names, "Greys")

		accurate_count = 0.0
		for i in range(len(pred_label)):
			if pred_label[i] == test_label[i]:
				accurate_count += 1.0
		accuracy = accurate_count / len(pred_label)
		return accuracy, pred_label


	def save_model(self, prior, likelihood):
		""" Save the trained model parameters
		"""

		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		""" Load the trained model parameters
		"""

		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
		"""
	    Get the feature likelihoods for high intensity pixels for each of the classes,
	        by sum the probabilities of the top 128 intensities at each pixel location,
	        sum k<-128:255 P(F_i = k | c).
	        This helps generate visualization of trained likelihood images.

	    Args:
	        likelihood(numpy.ndarray): likelihood (in log) with a dimension of
	            (# of features/pixels per image, # of possible values per pixel, # of class)
	    Returns:
	        feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
	            (# of features/pixels per image, # of class)
	    """
		feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2]))
		for i in range(likelihood.shape[0]):
			for c in range(likelihood.shape[2]):
				for k in range(128,256):
					feature_likelihoods[i][c] += likelihood[i][k][c]
		return feature_likelihoods
