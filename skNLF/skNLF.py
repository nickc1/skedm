"""
Scikit learn implementation of nonlinear forecasting.

By Nick Cortale
"""

import skNLF.metrics as mets
from scipy import stats as stats
from sklearn import neighbors
import numpy as np
from sklearn import metrics as skmetrics

class NonLin:
	"""
	Nonlinear forecasting
	"""

	def __init__(self,max_nn,weights):
		"""
		Parameters
		----------
		max_nn : int
			Maximum number of near neighbors to use
		weights : string
			-'uniform' : uniform weighting
			-'distance' : weighted as 1/distance
		"""
		self.max_nn = max_nn
		self.weights = weights

		self.knn = neighbors.KNeighborsRegressor(int(max_nn),
			weights=weights,metric='minkowski')


	def fit(self, Xtrain, ytrain):
		"""
		Xtrain : features (nsamples,nfeatures)
		ytrain : targets (nsamples,ntargets)
		"""
		self.Xtrain = Xtrain
		self.ytrain = ytrain

		self.knn.fit(Xtrain,ytrain)

	def dist_calc(self,Xtest):
		"""
		Calculates the distance from the testing set to the training
		set.

		Parameters
		----------
		Xtest : test features (nsamples,nfeatures)

		Returns
		-------
		self.dist : distance to each of the training samples
		self.ind : indices of the sample that are closest
		"""
		d,i = self.knn.kneighbors(Xtest)

		self.dist = d
		self.ind = i

		self.Xtest = Xtest
		print("distance calculated")

	def predict(self,nn):
		"""
		Make a prediction for a certain value of near neighbors

		Parameters
		----------
		nn : int
			How many near neighbors to use
		"""

		#check to see if distances have been calculated already

		if self.weights== 'uniform':

			neigh_ind = self.ind[:,0:nn]


			y_pred = np.mean(self.ytrain[neigh_ind], axis=1)


		elif self.weights =='distance':
			dist = self.dist[:,0:nn]
			neigh_ind = self.ind[:,0:nn]
			W = 1./dist

			y_pred = np.empty((dist.shape[0], self.ytrain.shape[1]), dtype=np.float)
			denom = np.sum(W, axis=1)

			for j in range(self.ytrain.shape[1]):
				num = np.sum(self.ytrain[neigh_ind, j] * W, axis=1)
				y_pred[:, j] = num / denom

		self.y_pred = y_pred

		return y_pred



	def predict_range(self,nn_range):
		"""
		predict over a range of near neighbors
		Parameters
		----------
		nn_range : 1d array
			The number of near neighbors to test

		Example :
		nn_range = np.arange(0,100,10) would calculate the
		predictions at 0,10,20,...,90 near neighbors
		"""

		xsize = self.dist.shape[0]
		ysize = self.ytrain.shape[1]
		zsize = len(nn_range)
		y_pred_range = np.empty((xsize,ysize,zsize))

		for ii,nn in enumerate(nn_range):

			y_p = self.predict(nn)

			y_pred_range[:,:,ii] = y_p

		self.y_pred_range = y_pred_range

		return y_pred_range

	def score(self, ytest, how='score'):
		"""
		Evalulate the predictions
		Parameters
		----------
		ytest : 2d array containing the targets
		how : string
			how to score the predictions
			-'score' : see scikit-learn's score function
			-'corrcoef' : correlation coefficient
		"""

		num_preds = ytest.shape[1]

		sc = np.empty((1,num_preds))

		for ii in range(num_preds):

			p = self.y_pred[:,ii]

			if how == 'score':
				sc[0,ii] = mets.score(p,ytest[:,ii])

			if how == 'corrcoef':
				sc[0,ii] = mets.corrcoef(p,ytest[:,ii])

		return sc

	def score_range(self,ytest,how='score'):
		"""
		scores the predictions if they were calculated for numerous
		values of near neighbors.

		Parameters
		----------
		ytest : 2d array containing the targets
		how : string
			how to score the predictions
			-'score' : see scikit-learn's score function
			-'corrcoef' : correlation coefficient
		"""


		num_preds = self.y_pred_range.shape[1]
		nn_range_len = self.y_pred_range.shape[2]

		sc = np.empty((nn_range_len,num_preds))

		for ii in range(nn_range_len):

			self.y_pred = self.y_pred_range[:,:,ii]

			sc[ii,:] = self.score(ytest,how=how)

		return sc



class NonLinDiscrete:
	"""
	Nonlinear forecasting
	"""

	def __init__(self,max_nn,weights='uniform'):
		"""
		Parameters
		----------
		max_nn : int
			Maximum number of near neighbors to use
		weights : string
			-'uniform' : uniform weighting
			-'distance' : weighted as 1/distance

		"""
		self.max_nn = max_nn
		self.weights = weights

		self.knn = neighbors.KNeighborsRegressor(int(max_nn),
			weights=weights,metric='hamming')


	def fit(self, Xtrain, ytrain):
		"""
		Xtrain : features (nsamples,nfeatures)
		ytrain : targets (nsamples,ntargets)
		"""
		self.Xtrain = Xtrain
		self.ytrain = ytrain

		self.knn.fit(Xtrain,ytrain)

	def dist_calc(self,Xtest):
		"""
		Calculates the distance from the testing set to the training
		set.

		Parameters
		----------
		Xtest : test features (nsamples,nfeatures)

		Returns
		-------
		self.dist : distance to each of the training samples
		self.ind : indices of the sample that are closest
		"""
		d,i = self.knn.kneighbors(Xtest)

		self.dist = d
		self.ind = i

		self.Xtest = Xtest
		print("distance calculated")

	def predict(self,nn):
		"""
		Make a prediction for a certain value of near neighbors

		Parameters
		----------
		nn : int
			How many near neighbors to use
		"""

		xsize = self.dist.shape[0]
		ysize = self.ytrain.shape[1]

		y_pred = np.empty((xsize,ysize))

		if self.weights =='uniform':

			neigh_ind = self.ind[:,0:nn]

			for j in range(self.ytrain.shape[1]):
				#mode, _ = stats.mode(self.ytrain[neigh_ind,j], axis=1)
				mode = mets.quick_mode_axis1(self.ytrain[neigh_ind,j].astype(int))
				y_pred[:,j] = mode# .ravel()


		elif self.weights=='distance':
			dist = self.dist[:,0:nn]
			neigh_ind = self.ind[:,0:nn]
			W = 1./dist

			for j in range(self.ytrain.shape[1]):
				mode, _ = mets.weighted_mode(self.ytrain[neigh_ind,j], W, axis=1)

				mode = np.asarray(mode.ravel(), dtype=np.intp)

				y_pred[:, j] = mode

		self.y_pred = y_pred

		return y_pred



	def predict_range(self,nn_range):
		"""
		predict over a range of near neighbors
		Parameters
		----------
		nn_range : 1d array
			The number of near neighbors to test

		Example :
		nn_range = np.arange(0,100,10) would calculate the
		predictions at 0,10,20,...,90 near neighbors
		"""

		xsize = self.dist.shape[0]
		ysize = self.ytrain.shape[1]
		zsize = len(nn_range)
		y_pred_range = np.empty((xsize,ysize,zsize))

		for ii,nn in enumerate(nn_range):

			y_p = self.predict(nn)

			y_pred_range[:,:,ii] = y_p

		self.y_pred_range = y_pred_range

		return y_pred_range

	def score_individual(self,ytest,how='tau'):
		"""
		Scores each individual near neighbor.

		Returns
		----------
		dist : 1d array of the average distances for each near neighbor
		score : 2d (num of NN, prediction distance) array of the score values
		"""

		num_neighbors = self.dist.shape[1]
		num_preds = self.ytrain.shape[1]
		score = np.zeros((num_neighbors, num_preds))

		for i in range(num_neighbors):

			ypred = self.ytrain[self.ind[:,i]] # grab all the 1st NN, then 2nd, etc...

			for j in range(num_preds):


				if how == 'classCompare':
					score[i,j] = mets.classCompare(ypred[:,j], ytest[:,j])

				elif how == 'classError':
					score[i,j] = mets.classificationError(ypred[:,j], ytest[:,j])

				elif how == 'tau':
					score[i,j] = mets.kleckas_tau(ypred[:,j], ytest[:,j])

		avg_dist = np.mean(self.dist,axis=0)
		return avg_dist, score



	def score(self, ytest, how='classCompare'):
		"""
		Evalulate the predictions
		Parameters
		----------
		ytest : 2d array containing the targets
		how : string
			how to score the predictions
			-'classCompare' : percent correctly predicted
			-'classError' : Dont use this
			-'tau' : kleckas tau
		"""

		num_preds = ytest.shape[1]

		sc = np.empty((1,num_preds))

		for ii in range(num_preds):

			p = self.y_pred[:,ii]

			if how == 'classCompare':
				sc[0,ii] = mets.classCompare(p,ytest[:,ii])

			elif how == 'classError':
				sc[0,ii] = mets.classificationError(p,ytest[:,ii])

			elif how == 'tau':
				sc[0,ii] = mets.kleckas_tau(p,ytest[:,ii])



		return sc

	def score_range(self,ytest,how='classError'):
		"""
		Evalulate the predictions that were made for numerous nn values
		Parameters
		----------
		ytest : 2d array containing the targets
		how : string
			how to score the predictions
			-'classCompare' : percent correctly predicted
			-'classError' : Dont use this
			'tau' : kleckas tau
		"""


		num_preds = self.y_pred_range.shape[1]
		nn_range_len = self.y_pred_range.shape[2]

		sc = np.empty((nn_range_len,num_preds))

		for ii in range(nn_range_len):

			self.y_pred = self.y_pred_range[:,:,ii]

			sc[ii,:] = self.score(ytest,how=how)

		return sc




class embed:

	def __init__(self,X):
		"""
		Parameters
		----------
		X : series, 2d array, or 3d array to be embedded
		"""

		self.X = X

	def mutual_information(self,max_lag):
		"""
		Calculates the mutual information between the an unshifted time series
		and a shifted time series. Utilizes scikit-learn's implementation of
		the mutual information found in sklearn.metrics.

		Parameters
		----------

		X : 1-D array
			time series that is to be shifted over

		max_lag : integer
			maximum amount to shift the time series

		Returns
		-------
		m_score : 1-D array
			mutual information at between the unshifted time series and the
			shifted time series
		"""

		#number of bins - say ~ 20 pts / bin for joint distribution
		#and that at least 4 bins are required
		N = max(self.X.shape)
		num_bins = max(4.,np.floor(np.sqrt(N/20)))
		num_bins = int(num_bins)

		m_score = np.zeros((max_lag))

		for jj in range(max_lag):
			lag = jj+1

			ts = self.X[0:-lag]
			ts_shift = self.X[lag:]

			min_ts = np.min(self.X)
			max_ts = np.max(self.X)+.0001 #needed to bin them up

			bins = np.linspace(min_ts,max_ts,num_bins+1)

			bin_tracker = np.zeros_like(ts)
			bin_tracker_shift = np.zeros_like(ts_shift)

			for ii in range(num_bins):

				locs = np.logical_and( ts>=bins[ii], ts<bins[ii+1] )
				bin_tracker[locs] = ii

				locs_shift = np.logical_and( ts_shift>=bins[ii], ts_shift<bins[ii+1] )
				bin_tracker_shift[locs_shift]=ii


			m_score[jj] = skmetrics.mutual_info_score(bin_tracker,bin_tracker_shift)
		return m_score


	def mutual_information_spatial(self,max_lag,percent_calc=.5):
		"""
		Calculates the mutual information along the rows and columns at a
		certain number of indices (percent_calc) and returns
		the sum of the mutual informaiton along the columns and along the rows.

		Parameters
		----------

		M : 2-D array
			input two-dimensional image

		max_lag : integer
			maximum amount to shift the space

		percent_calc : float
			How many rows and columns to use to calculate the mutual information

		Returns
		-------

		R_mut : 1-D array
			the mutual inforation averaged down the rows (vertical)

		C_mut : 1-D array
			the mutual information averaged across the columns (horizontal)

		r_mi : 2-D array
			the mutual information down each row (vertical)

		c_mi : 2-D array
			the mutual information across the columns (horizontal)


		"""

		M = self.X.copy()
		rs, cs = np.shape(M)

		rs_iters = int(rs*percent_calc)
		cs_iters = int(cs*percent_calc)

		r_picks = np.random.choice(np.arange(rs),size=rs_iters,replace=False)
		c_picks = np.random.choice(np.arange(cs),size=cs_iters,replace=False)


		# The r_picks are used to calculate the MI in the columns
		# and the c_picks are used to calculate the MI in the rows

		c_mi = np.zeros((max_lag,rs_iters))
		r_mi = np.zeros((max_lag,cs_iters))

		for ii in range(rs_iters):

			m_slice = M[r_picks[ii],:]
			self.X = m_slice
			c_mi[:,ii] = self.mutual_information(max_lag)

		for ii in range(cs_iters):

			m_slice = M[:,c_picks[ii]]
			self.X = m_slice
			r_mi[:,ii] = self.mutual_information(max_lag)

		r_mut = np.sum(r_mi,axis=1)
		c_mut = np.sum(c_mi,axis=1)

		self.X = M
		return r_mut, c_mut, r_mi, c_mi


	def embed_vectors_1d(self,lag,embed,predict):
		"""
		Embeds vectors from a two dimensional image in m-dimensional space.

		Parameters
		----------
		X : array
			A 1-D array representing the training or testing set.

		lag : int
			lag values as calculated from the first minimum of the mutual info.

		embed : int
			embedding dimension, how many lag values to take

		predict : int
			distance to forecast (see example)


		Returns
		-------
		features : array of shape [num_vectors,embed]
			A 2-D array containing all of the embedded vectors

		targets : array of shape [num_vectors,predict]
			A 2-D array containing the evolution of the embedded vectors

		Example
		-------
		X = [0,1,2,3,4,5,6,7,8,9,10]

		em = 3
		lag = 2
		predict=3

		returns:
		features = [[0,2,4], [1,3,5], [2,4,6], [3,5,7]]
		targets = [[5,6,7], [6,7,8], [7,8,9], [8,9,10]]
		"""

		tsize = self.X.shape[0]
		t_iter = tsize-predict-(lag*(embed-1))

		features = np.zeros((t_iter,embed))
		targets = np.zeros((t_iter,predict))

		for ii in range(t_iter):

			end_val = ii+lag*(embed-1)+1

			part = self.X[ii : end_val]

			features[ii,:] = part[::(lag)]
			targets[ii,:] = self.X[end_val:end_val+predict]
		return features, targets




	def embed_vectors_2d(self,lag,embed,predict,percent=0.1):
		"""
		Embeds vectors from a two dimensional image in m-dimensional space.

		Parameters
		----------
		X : array
			A 2-D array representing the training set or testing set.

		lag : tuple of ints (r,c)
			row and column lag values (r,c) can think of as (height,width).

		embed : tuple of ints (r,c)
			row and column embedding shape (r,c) can think of as (height,width).
			c must be odd

		predict : int
			distance in the space to forecast (see example)

		percent : float (default = None)
			percent of the space to embed. Used for computation efficiency

		Returns
		-------
		features : array of shape [num_vectors,r*c]
			A 2-D array containing all of the embedded vectors

		targets : array of shape [num_vectors,predict]
			A 2-D array containing the evolution of the embedded vectors


		Example:
		lag = (3,4)
		embed = (2,5)
		predict = 2


		[f] _ _ _ [f] _ _ _ [f] _ _ _ [f] _ _ _ [f]
		 |         |         |         |         |
		 |         |         |         |         |
		[f] _ _ _ [f] _ _ _ [f] _ _ _ [f] _ _ _ [f]
							[t]
							[t]
		"""

		rsize = self.X.shape[0]
		csize = self.X.shape[1]

		r_lag,c_lag = lag
		rem,cem = embed


		# determine how many iterations we will have and
		# the empty feature and target matrices

		c_iter = csize - c_lag*(cem-1)
		r_iter = rsize  - predict - r_lag*(rem-1)

		#randomly pick spots to be embedded
		ix = np.random.rand(r_iter,c_iter)<=percent
		r_inds,c_inds = np.where(ix)

		targets = np.zeros((len(r_inds),predict))
		features = np.zeros((len(r_inds),rem*cem))


		print("targets before loop:", targets.shape)

		for ii in range(features.shape[0]):

			rs = r_inds[ii]
			cs = c_inds[ii]


			r_end_val = rs+r_lag*(rem-1)+1
			c_end_val = cs+c_lag*(cem-1)+1

			part = self.X[rs : r_end_val, cs : c_end_val ]

			features[ii,:] = part[::r_lag,::c_lag].ravel()
			targets[ii,:] = self.X[r_end_val:r_end_val+predict,cs + int(c_lag*(cem-1)/2)]


		return features,targets



	def embed_vectors_3d(self,lag,embed,predict,percent=0.1):
		"""
		Embeds vectors from a three-dimensional matrix in m-dimensional space.
		The third dimension is assumed to be time.

		Parameters
		----------
		X : array
			A 3-D array representing the training set or testing set.

		lag : tuple of ints (r,c)
			row and column lag values (r,c) can think of as (height,width).

		embed : tuple of ints (r,c,t)
			row and column, and time embedding shape (r,c,t) can think of as
			(height,width,time). c must be odd

		predict : int
			distance in the space to forecast (see example)

		percent : float (default = None)
			percent of the space to embed. Used for computation efficiency

		Returns
		-------
		features : array of shape [num_vectors,r*c]
			A 2-D array containing all of the embedded vectors

		targets : array of shape [num_vectors,predict]
			A 2-D array containing the evolution of the embedded vectors


		Example:
		lag = (3,4,2) #height,width,time
		embed = (3,3)
		predict = 2


		[f] _ _ _ [f] _ _ _ [f]
		 |         |         |
		 |         |         |
		[f] _ _ _ [f] _ _ _ [f]
		 |         |         |
		 |         |         |
		[f] _ _ _ [f] _ _ _ [f]

		The targets would be directly below the center [f]

		"""

		rsize = X.shape[0]
		csize = X.shape[1]
		tsize = X.shape[2]

		r_lag,c_lag,t_lag = lag
		rem,cem,tem = embed


		# determine how many iterations we will have and
		# the empty feature and target matrices

		c_iter = csize - c_lag*(cem-1)
		r_iter = rsize  - r_lag*(rem-1)
		t_iter = tsize - t_lag*(tem-1) - predict

		#create tuples of all the possible x,y,t values for the image
		# creates a bunch of tuples
		ix = np.random.rand(r_iter,c_iter,t_iter)<=percent

		r_inds,c_inds,t_inds = np.where(ix)

		#choose only a percent of them if percent is defined

		percent_tot = len(r_inds)

		targets = np.zeros((percent_tot,predict))
		features = np.zeros((percent_tot,rem*cem*tem))



		print('targets before loop:', targets.shape)

		for ii in range(features.shape[0]):

			rs = r_inds[ii]
			cs = c_inds[ii]
			ts = t_inds[ii]


			r_end_val = rs + r_lag * (rem-1) + 1
			c_end_val = cs + c_lag * (cem-1) + 1
			t_end_val = ts + t_lag * (tem-1) + 1

			part = X[rs : r_end_val, cs : c_end_val, ts : t_end_val ]

			features[ii,:] = part[::r_lag,::c_lag,::t_lag].ravel()


			rs_target = rs + r_lag
			cs_target = cs + c_lag
			targets[ii,:] = X[rs + r_lag*(rem-1)/2,
				cs+ c_lag*(cem-1)/2, t_end_val:t_end_val+predict].ravel()


		return features,targets
