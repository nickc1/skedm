"""
Generates data for skNLF (SciKit Nonlinear Forecasting)

Available data:

  1D
------
logistic_map : logistic equation
noisyPeriodic : sine and cosine wave added with noise
noisyPeriodic_complicated : more complicated sine and cosine wave
noise : randomly generated numbers
lorenz : lorenz equations

  2D
------
chaos2D : 2D logistic map diffused in space
periodic : sine and cosine addition
periodicBrown : sine and cosine with brown noise added
brownNoise : brown noise generator
noise : randomly generated numbers
chaos3D : logistic map diffused in the third dimension
randomCircles : randomly placed circles
circleInCircle : circles sorrouned by larger circle
circlesWithStuff : larger circles with smaller circles around
randomSizedCircles : randomly sized circles spread around
voronoiMatrix : voronoi polygons

"""



import numpy as np
from numpy import genfromtxt
from scipy import integrate
from sklearn import neighbors


def logistic_map(sz=256,A=3.99,seed=36, noise=0):
	"""
	Logistic map.

	X(t+1) = rX(t)(1 - X(t)) + random

	Parameters
	----------

	sz : int
		size of the time series to be generated

	A : float
		Parameter for the logistic map. Values values beyond 3.56995
		exhibit chaotic behaviour.

	seed : int
		sets the random seed for the logistic map. Allows results to be
		easily reproduced.

	noise : float
		Amplitude of noise to add to the logistic map

	Returns
	-------

	X : 1D array
		Logistic map of size (sz)

	"""

	# set random seed
	np.random.seed(seed=seed)

	X = np.zeros((sz,))
	X[0] = np.random.rand(1,)


	for tt in range(sz-1):

		# logistic equation in space
		X[tt+1] = A*X[tt]*(1-X[tt])

	X += noise*np.random.rand(sz)

	return X

def noisyPeriodic(sz=256,noise=.5,freq=52,seed=36):
	"""
	A simple periodic equation with a a specified amplitude of noise.

	X = sin(x) + .5cos(x) + random

	Parameters
	----------

	sz : int
		Length of the time series

	noise : float
		Amplitude of the noise

	freq : int
		Frequency of the periodic equation

	seed : int
		Sets the random seed for reproducible results

	Returns
	-------

	X : 1D array
		Returns a 1D periodic equaiton of size (sz) with values between 0 and 1
	"""

	np.random.seed(seed=seed)

	x = np.linspace(0,freq*np.pi,sz)  #prep range for averages

	X = np.sin(x) + .5*np.cos(x) + noise*np.random.rand(sz)

	#all positive and between 0 and 1
	X = X + np.abs(np.min(X))
	X = X/np.max(X)
	return X

def noisyPeriodic_complicated(sz=256,noise=.5,freq=52,seed=36):
	"""
	A more complicated periodic equation with a a specified amplitude of noise.

	X = sin(x) + .5cos(.5x) + .25sin(.25x) + random

	Parameters
	----------

	sz : int
		Length of the time series

	noise : float
		Amplitude of the noise

	freq : int
		Frequency of the periodic equation

	seed : int
		Sets the random seed for reproducible results

	Returns
	-------

	X : 1D array
		Returns a 1D periodic equaiton of size (sz) with values between 0 and 1
	"""

	np.random.seed(seed=seed)

	x = np.linspace(0,freq*np.pi,sz)  #prep range for averages

	X = np.sin(x) + .5*np.cos(.5*x) +.25*np.sin(.25*x) + noise*np.random.rand(sz)

	#all positive and between 0 and 1
	X = X + np.abs(np.min(X))
	X = X/np.max(X)

	return X

def noise(sz=256, seed=36):
	"""
	A random distribution of numbers.

	X = random

	Parameters
	----------

	sz : int
		Length of the time series

	seed : int
		Sets the random seed for reproducible results

	Returns
	-------

	X : 1D array
		Returns a 1D periodic equaiton of size (sz) with values between 0 and 1
	"""

	np.random.seed(seed=seed)

	X = np.random.rand(sz)

	return X


def lorenz(sz=10000,noise=0,max_t=100.):
	"""
	Integrates the lorenz equation defined in lorenz_deriv

	Parameters
	----------

	sz : int
		Length of the time series to be integrated

	noise : float
		Amplitude of noise to be added to the lorenz equation

	max_t : float
		Length of time to solve the lorenz equation over

	Returns
	-------

	X : 1D array
		Returns a 1D periodic equaiton of size (sz) with values between 0 and 1
	"""

	x0 = [1, 1, 1]  # starting vector
	t = np.linspace(0, max_t, sz)  # one thousand time steps
	X = integrate.odeint(lorenz_deriv, x0, t) + noise*np.random.rand(sz,3)

	return X


def lorenz_deriv(xyz, t0, sigma=10., beta=8./3, rho=28.0):
	"""
	Lorenz equations to be integrated in the function lorenz

	dx/dt = sigma(y - x)

	dy/dt = x(rho - z) - y

	dz/dt = xy - Bz


	"""
	x,y,z = xyz
	return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]



"""
***************
TWO DIMENSIONAL
***************
"""


def chaos2D(sz=128, A=3.99, eps=1., seed=36, noise=0):
	"""
	Logistic map diffused in space. Refer to thesis for specifics.


	Parameters
	----------

	sz : int
		row and column size of the spatio-temporal series to be generated

	A : float
		Parameter for the logistic map. Values beyond 3.56995
		exhibit chaotic behaviour.

	seed : int
		sets the random seed for the logistic map. Allows results to be
		easily reproduced.

	noise : float
		Amplitude of noise to add to the logistic map

	Returns
	-------

	X : 2D array
		Spatiotemporal logistic map of size (sz)

	"""

	# set random seed
	np.random.seed(seed=seed)

	X = np.zeros((sz,sz))
	X[0,:] = np.random.rand(1,sz)


	for tt in range(sz-1):
		left_X = np.roll(X,1)  # shift it around for diffusion
		right_X = np.roll(X,-1)

		left2_X = np.roll(X,2)
		right2_X = np.roll(X,-2)

		# logistic equation in space
		X[tt+1,:] = ( (A/(1+4*eps))*
		(X[tt,:]*(1-X[tt,:])
		+ eps*left_X[tt,:]*(1-left_X[tt,:])
		+ eps*right_X[tt,:]*(1-right_X[tt,:])
		+ eps*right2_X[tt,:]*(1-right2_X[tt,:])
		+ eps*left2_X[tt,:]*(1-left2_X[tt,:])) )

	X += noise*np.random.rand(sz,sz)
	return X

def periodic(sz=128,noise=0.5,freq=36):
	"""
	A simple 2D periodic equation with a a specified amplitude of noise.

	X = sin(y) + .5cos(x) + random

	Parameters
	----------

	sz : int
		Length of the time series

	noise : float
		Amplitude of the noise

	freq : int
		Frequency of the periodic equation

	seed : int
		Sets the random seed for reproducible results

	Returns
	-------

	X : 2D array
		Returns a 2D periodic equaiton of size (sz) with values between 0 and 1
	"""

	# set random seed
	np.random.seed(seed=seed)

	x = np.linspace(0,freq*np.pi,sz)  #prep range for averages
	y = np.linspace(0,freq*np.pi,sz)
	xx,yy = np.meshgrid(x,y)

	X = np.sin(yy) + .5*np.cos(xx) #+ np.cos(.5*yy)  + np.cos(.25*xx)

	X += noise*np.random.rand(sz,sz)


	#normalize
	X += np.abs(X.min())
	X /= X.max()

	return X


def periodicBrown(sz=128,noise=1.5,freq=36):
	"""
	A periodic equation with a specified amplitude of brown noise.
	Calls the function brownNoise

	X = sin(y + noise*2pi)

	Parameters
	----------

	sz : int
		Length of the spatiotemporal series

	noise : float
		Amplitude of the noise

	freq : int
		Frequency of the periodic equation

	seed : int
		Sets the random seed for reproducible results

	Returns
	-------

	X : 2D array
		Returns a 2D periodic equaiton of size (sz) with values between 0 and 1
	"""

	# set random seed
	np.random.seed(seed=seed)

	x = np.linspace(0,freq*np.pi,sz)  #prep range for averages
	y = np.linspace(0,freq*np.pi,sz)
	xx,yy = np.meshgrid(x,y)

	noise = brownNoise(sz=sz)

	X = np.sin(yy+noise*2*np.pi) #+ np.cos(xx+noise*2*np.pi)

	#normalize
	X += np.abs(X.min())
	X /= X.max()

	return X

def brownNoise(sz=128,num_walks=500,walk_sz=100000,spread=1000,seed=3):
	'''
	Creates brown noise with a bunch of random walks.
	Subsamples to generate sizes: 128, 256, or 512.
	512 is the full size

	Parameters
	----------

	sz : int
		row and column size of the array to be returned

	num_walks : int
		number of random walks to taken

	walk_sz : int
		Length of the random walk to take

	spread : int
		normal distribution of walks sizes randn*spread

	seed : int
		sets the random seed for reproducible results

	'''

	#set random seed
	np.random.seed(seed=seed)

	store_x = np.empty((walk_sz,2,num_walks))

	for ii in range(num_walks):

		x1 = np.random.randn(walk_sz,2)
		x1[0,:] = np.random.rand(2,)*spread  # start them at random locations
		x = np.cumsum(x1,axis=0)

		# store x, but shift it so all the values are positive integers
		store_x[:,:,ii] = x

	# now we want to fill in a grid with the walks
	store_x = store_x.astype(int) + 1000

	grid = np.zeros((3500,3500))

	for ii in range(num_walks):
		for jj in range(walk_sz):

			grid[store_x[jj,0,ii],store_x[jj,1,ii]]+=1

	# now just return the middle subsample
	# and scale it between 0 and 1
	X = grid[1500:2012,1500:2012] / np.max( grid[1500:2012,1500:2012])

	if sz==128:

		X = X[0::4,0::4]

	elif sz == 256:

		X = X[0::2,0::2]

	return X

def noise(sz=128,seed=36):
	"""
	A 2D random distribution of numbers.

	X = random

	Parameters
	----------

	sz : int
		row and column size of array

	seed : int
		Sets the random seed for reproducible results

	Returns
	-------

	X : 2D array
		Returns a 2D array of size (sz,sz) with values between 0 and 1
	"""

	# set random seed
	np.random.seed(seed=seed)

	X = np.random.rand(sz,sz)

	return X



def chaos3D(sz=128,A=3.99,eps=1.,steps=100):
	"""
	Logistic map diffused in space and then taken through time.
	Chaos evolves in 3rd dimension.

	Parameters
	----------

	sz : int
		row and column size of the spatio-temporal series to be generated

	A : float
		Parameter for the logistic map. Values beyond 3.56995
		exhibit chaotic behaviour.

	eps : float
		Amount of coupling/diffusion between adjecent cells

	seed : int
		sets the random seed for the logistic map. Allows results to be
		easily reproduced.

	Returns
	-------

	X : 2D array
		Spatiotemporal logistic map of size (sz)

	"""

	X = np.random.rand(sz,sz)


	for tt in range(steps-1):
		left_X = np.roll(X,1)  # shift it around for diffusion
		right_X = np.roll(X,-1)

		left2_X = np.roll(X,2)
		right2_X = np.roll(X,-2)

		# logistic equation in space

		X = (1/(1+4*eps))*(A*X*(1-X)
		+ A*left_X*(1-left_X)
		+ A*right_X*(1-right_X)
		+ A*right2_X*(1-right2_X)
		+ A*left2_X*(1-left2_X))

	return X


def randomCircles(sz=256,rad=20.,sigma=1,num_circles = 1000):
	"""
	Randomly places down gaussian circles and the sum is taken.
	Calls circleCreate to make the circles

	Parameters
	----------

	sz : int
		Row and column size of the space

	rad : float
		Radius of the circles

	sigma : float
		Constant to create gaussian circles. Changes the distribution of values

	num_circles : int
		Number of circles to place down randomly

	Returns
	-------

	X : 2D array
		Returns the summed gaussian circles
	"""

	circ_store = np.empty((sz,sz,num_circles))


	for ii in range(num_circles):


		r = np.floor(np.random.rand()*sz)
		c = np.floor(np.random.rand()*sz)

		circ_store[:,:,ii] = self.circleCreate(r,c,sz,rad,sigma)

	X = circ_store.sum(axis=2)

	X = X/np.max(X)

	return X

def circleCreate(r,c,sz,rad,sigma,gauss=True):
	"""
	To be used inside randomCircles

	Parameters
	----------
	r,c : int
		Defines the center of the circle

	sz : int
		Size of the space where the circle will be placed

	rad : int
		Radius of the circle

	sigma : float
		Distribution of the values within the circle

	gauss : boolean
		Controls whether to apply a gaussian filter to a cirlce of ones

	Returns
	-------

	X : 2D array
		Array containing a single circle

	"""

	rad = np.around(np.random.rand()*rad)
	y,x = np.ogrid[-r:sz-r, -c:sz-c]
	mask = x*x + y*y <= rad*rad

	array = np.zeros((sz, sz))
	array[mask] = 1

	if gauss==True:
		array = filt.gaussian_filter(array,sigma)


	return array


def circleInCircle(sz=256,rad1 = 5, rad2 = 8,num_blobs=1000):
	'''
	Create circles inside larger circles. These circles cannot overlap.
	Calls the function blobber

	Parameters
	----------

	sz : int
		row and column size of the returned space

	rad1 : float
		radius of the interior circle

	rad2 : float
		radius of the exterior circle

	num_blobs : int
		number of circles to create

	Returns
	-------

	blobs : 2D array of shape (sz,sz)
		2D array of the circles within circles

	'''


	blobs = np.zeros((sz,sz))
	blob_count = 0

	for ii in range(num_blobs):

		r = np.around(np.random.rand()*sz)
		c = np.around(np.random.rand()*sz)

		new_blob = blobber(r,c,rad1,rad2,sz)

		#check to see if there is any overlap

		occupied1 = new_blob>0

		occupied2 = blobs>0

		overlap = np.logical_and(occupied1,occupied2)

		if np.any(overlap)==False:

			blobs+=new_blob

			blob_count+=1

	print('Blobs Generated:',blob_count)

	return blobs


def blobber(r,c,rad1,rad2,sz):
	"""
	Creates a circle sorrounded by a larger circle.
	To be used within circleInCircle.

	Parameters
	----------

	r,c : float
		Center of the circles

	rad1 : float
		Radius of the interior circle

	rad2 : float
		Radius of the exterior circle

	sz : int
		Size of the space to generated

	Returns
	-------

	X : 2D array of size (sz,sz)
		Array containing a single circle within another circle


	"""
	y,x = np.ogrid[-r:sz-r, -c:sz-c]
	mask1 = x*x + y*y <= rad1*rad1
	mask2 = x*x + y*y <= rad2*rad2

	X = np.zeros((sz, sz))
	X[mask2] = 1
	X[mask1] = 2

	return X

def blobber2(r,c,rad1,sz):
	"""
	Creates a single circle

	Parameters
	----------

	r,c : int
		Center of the circle

	rad : int
		Radius of the circle

	sz : int
		row and column size of the space in which the circle is placed

	Returns
	-------

	X : 2D array
		Array containing a single circle
	"""

	y,x = np.ogrid[-r:sz-r, -c:sz-c]
	mask1 = x*x + y*y <= rad1*rad1


	X = np.zeros((sz, sz))
	X[mask1] = 1


	return X

def circlesWithStuff(sz=256,rad1 = 5, rad2 = 8,num_blobs=100):
	"""
	Create circles with random smaller circles spread around randomly.
	Same number of large circles and smaller circles.
	Calls blobber2

	Parameters
	----------

	sz : int
		row and column size of the space in which the circle is placed

	rad1 : int
		Radius of the smaller circle

	rad2 : int
		Radius of the larger circle

	num_blobs : number of large and small circles to create

	Returns
	-------

	blobs : 2D array
		Array containing a all circles
	"""

	blobs = np.zeros((sz,sz))
	blob_count = 0

	for ii in range(num_blobs):

		r1 = np.around(np.random.rand()*sz)
		c1 = np.around(np.random.rand()*sz)

		r2 = np.around(np.random.rand()*sz)
		c2 = np.around(np.random.rand()*sz)


		new_blob1 = blobber2(r1,c1,rad1,sz)

		new_blob2 = blobber2(r2,c2,rad2,sz)*2

		#check to see if there is any overlap

		occupied1 = new_blob1 > 0

		occupied2 = new_blob2 > 0

		old_occupied = blobs > 0

		overlap1 = np.logical_and(occupied1,occupied2).any()
		overlap2 = np.logical_and(occupied1,old_occupied).any()
		overlap3 = np.logical_and(occupied2,old_occupied).any()

		if ( (overlap1 == False) and (overlap2==False) and (overlap3==False)):

			blobs+=new_blob1
			blobs+=new_blob2

			blob_count+=1

	print('Number of blobs:',blob_count)

	return blobs

def randomSizedCircles(sz=1024,rad_max = 28 ,num_blobs=3000):
	"""
	Create random sized circles spread around randomly and assign them
	to classes: 1:27

	Parameters
	----------

	sz : int
		row and column size of the space in which the circle is placed

	rad_max : int
		Radius of the largest circle

	num_blobs : total number of circles to create

	Returns
	-------

	blobs : 2D array
		Array containing a all circles
	"""

	blobs = np.zeros((sz,sz))
	blob_count = 0
	rad_store = np.zeros((num_blobs,))

	for ii in range(num_blobs):

		r = np.around(np.random.rand()*sz)
		c = np.around(np.random.rand()*sz)

		rad = np.random.randint(2,high=rad_max+1)

		new_blob = blobber2(r,c,rad,sz)*rad

		#check to see if there is any overlap

		occupied = new_blob > 0

		old_occupied = blobs > 0

		overlap = np.logical_and(occupied,old_occupied).any()


		if  (overlap == False):

			blobs+=new_blob

			rad_store[blob_count] = rad
			blob_count+=1

	blobs -= 1
	blobs[blobs<0] = 0

	print(blob_count)

	return blobs


def voronoiMatrix(sz=512,percent=0.1,num_classes=27):
	"""
	Create voronoi polygons.

	Parameters
	----------

	sz : int
		row and column size of the space in which the circle is placed

	percent : float
		Percent of the space to place down centers of the voronoi polygons.
		Smaller percent makes the polygons larger

	num_classes : int
		Number of classes to assign to each of the voronoi polygons

	Returns
	-------

	X : 2D array
		Array containing all voronoi polygons
	"""


	X = np.zeros((sz,sz))

	#fill in percentage of the space
	locs = np.random.rand(sz,sz)<=percent
	vals = np.random.randint(1,num_classes,size=(sz,sz))
	X[locs]=vals[locs]

	#get all the indices of the matrix
	cc,rr = np.meshgrid(np.arange(0,sz),np.arange(0,sz))

	f = np.zeros((sz**2,2))
	f[:,0]=rr.ravel()
	f[:,1]=cc.ravel()

	t = X.ravel()

	train_ind = t>0

	f_train = f[train_ind]
	t_train = t[train_ind]

	clf = neighbors.KNeighborsClassifier(n_neighbors=1)
	clf.fit(f_train, t_train)

	preds = clf.predict(f)

	locs = f.astype(int)
	X[locs[:,0],locs[:,1]] = preds

	return X
