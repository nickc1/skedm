"""
Metrics for scoring predictions and also some more specialized
math needed for skNLF
"""

import numpy as np
from scipy import stats as stats
from numba import jit

def corrCoef(preds,actual):
	"""
	Correlation Coefficient of between predicted values and actual values

	Parameters
	----------

	preds : array shape (num samples,num targets)

	test : array of shape (num samples, num targets)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
	"""

	cc = np.corrcoef(preds,actual)[0,1]

	return cc


def classCompare(preds,actual):
	"""
	Percent correct between predicted values and actual values

	Parameters
	----------

	preds : array shape (num samples,num targets)

	test : array of shape (num samples, num targets)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
	"""


	cc = np.mean( preds == actual )

	return cc

def classificationError(preds,actual):
	"""
	Percent correct between predicted values and actual values scaled
	to the most common prediction of the space

	Parameters
	----------

	preds : array shape (num samples,)

	test : array of shape (num samples,)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
	"""

	most_common,_=stats.mode(actual,axis=None)

	num = np.mean(preds == actual)
	denom = np.mean(actual == most_common)

	cc = num/denom.astype('float')

	return cc

def kleckas_tau(preds,actual):
	"""
	Calculates kleckas tau

	Parameters
	----------
	preds : array shape (num samples,)
	test : array of shape (num samples,)
		actual values from the testing set

	Returns
	-------
	cc : float
		Returns the correlation coefficient
	"""

	ncorr = np.sum(preds == actual) #number correctly classified
	cats_unique = np.unique(actual)

	sum_t = 0
	for cat in cats_unique:
		ni = np.sum(cat==actual)
		pi = float(ni)/len(preds)
		sum_t += ni*pi


	tau = (ncorr - sum_t) / (len(preds) - sum_t)

	return tau

def cohens_kappa(preds,actual):
	"""
	Calculates cohens kappa

	Parameters
	----------
	preds : array shape (num samples,)
	test : array of shape (num samples,)
		actual values from the testing set

	Returns
	-------
	cc : float
		Returns the correlation coefficient
	"""

	c = cohen_kappa_score(preds,actual)

	return c


def varianceExplained(preds,actual):
	"""
	Explained variance between predicted values and actual values scaled
	to the most common prediction of the space

	Parameters
	----------

	preds : array shape (num samples,num targets)

	actual : array of shape (num samples, num targets)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
	"""


	cc = np.var(preds - actual) / np.var(actual)

	return cc




def score(preds,actual):
	"""
	The coefficient R^2 is defined as (1 - u/v), where u is the regression
	sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
	sum of squares ((y_true - y_true.mean()) ** 2).sum(). Best possible
	score is 1.0, lower values are worse.

	Parameters
	----------

	preds : array shape (num samples,num targets)

	test : array of shape (num samples, num targets)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
	"""

	u = np.square(actual - preds ).sum()
	v = np.square(actual - actual.mean()).sum()
	r2 = 1 - u/v

	return r2


def weighted_mode(a, w, axis=0):
	"""This function is borrowed from sci-kit learn's extmath.py

	Returns an array of the weighted modal (most common) value in a

	If there is more than one such value, only the first is returned.
	The bin-count for the modal bins is also returned.

	This is an extension of the algorithm in scipy.stats.mode.

	Parameters
	----------
	a : array_like
	n-dimensional array of which to find mode(s).
	w : array_like
	n-dimensional array of weights for each value
	axis : int, optional
	Axis along which to operate. Default is 0, i.e. the first axis.

	Returns
	-------
	vals : ndarray
	Array of modal values.
	score : ndarray
	Array of weighted counts for each mode.

	Examples
	--------
	>>> from sklearn.utils.extmath import weighted_mode
	>>> x = [4, 1, 4, 2, 4, 2]
	>>> weights = [1, 1, 1, 1, 1, 1]
	>>> weighted_mode(x, weights)
	(array([ 4.]), array([ 3.]))

	The value 4 appears three times: with uniform weights, the result is
	simply the mode of the distribution.

	>>> weights = [1, 3, 0.5, 1.5, 1, 2] # deweight the 4's
	>>> weighted_mode(x, weights)
	(array([ 2.]), array([ 3.5]))

	The value 2 has the highest score: it appears twice with weights of
	1.5 and 2: the sum of these is 3.

	See Also
	--------
	scipy.stats.mode
	"""
	if axis is None:
		a = np.ravel(a)
		w = np.ravel(w)
		axis = 0
	else:
		a = np.asarray(a)
		w = np.asarray(w)
		axis = axis

	if a.shape != w.shape:
		print('both weights')
		w = np.zeros(a.shape, dtype=w.dtype) + w

	scores = np.unique(np.ravel(a))       # get ALL unique values
	testshape = list(a.shape)
	testshape[axis] = 1
	oldmostfreq = np.zeros(testshape)
	oldcounts = np.zeros(testshape)

	for score in scores:
		template = np.zeros(a.shape)
		ind = (a == score)
		template[ind] = w[ind]
		counts = np.expand_dims(np.sum(template, axis), axis)
		mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
		oldcounts = np.maximum(counts, oldcounts)
		oldmostfreq = mostfrequent

	return mostfrequent, oldcounts

def deterministic_metric(CC,lag):
	"""
	Calculate the deterministic metric. Calculates the difference in forecast
	skill between low numbers of near neighbors and high numbers of near neighbors

	Parameters
	----------

	CC : 2-D array
		array of forecast skill for a given system
	lag : int
		lag value for the system which is calculated as the first minimum of the
		mutual information

	Returns
	-------
	d_metric : float
		evaluated deterministic metric

	"""

	nn_iters = CC.shape[0]
	half = int(nn_iters/2.)
	d_range = int(lag/2.)
	d_metric=0
	for ii in range(d_range):
		left = CC[0:half,ii]
		right = CC[half::,ii]

		s_max = np.max(left)
		s_min = np.min(right)
		d_metric += (s_max-s_min)
	return d_metric


@jit
def quick_mode_axis1(X):
	X = X.astype(int)
	len_x = len(X)
	mode = np.zeros(len_x)
	for i in range(len_x):
		mode[i] = np.bincount(X[i,:]).argmax()
	return mode


def auto_correlation(X,shift):
	"""
	Parameters
	----------
	X : 2d array to be shifted
	frac_shift : what percent of length to shift the array
	"""

	r,c = X.shape
	rshift,cshift = shift

	rsum = np.zeros(rshift)
	csum = np.zeros(cshift)
	#shift horizontally
	for i in range(rshift):

		shifted = np.roll(X,i,axis=0)
		rsum[i] = np.sum(shifted == X)

	for i in range(cshift):
		shifted = np.roll(X,i,axis=1)
		csum[i] = np.sum(shifted == X)

	return rsum,csum

def keep_diversity(X,thresh=1.):
	"""
	Throws out rows of only one class.
	X : 2d array of ints

	Returns
	keep : 1d boolean

	ex:
	[1 1 1 1]
	[2 1 2 3]
	[2 2 2 2]
	[3 2 1 4]

	returns:
	[F]
	[T]
	[F]
	[T]
	"""

	X = X.astype(int)
	mode = quick_mode_axis1(X).reshape(-1,1)

	compare = np.repeat(mode,X.shape[1],axis=1)
	thresh = int(thresh*X.shape[1])
	keep = np.sum(compare==X, axis=1) < X.shape[1]

	return keep
