#
# Generates data for skedm (SciKit Emperical Dynamic Modeling)
#
# Available data:
#
#   1D
# ------
# logistic_map : logistic equation
# noisyPeriodic : sine and cosine wave added with noise
# noisyPeriodic_complicated : more complicated sine and cosine wave
# noise : randomly generated numbers
# lorenz : lorenz equations
#
#   2D
# ------
# chaos2D : 2D logistic map diffused in space
# periodic : sine and cosine addition
# periodicBrown : sine and cosine with brown noise added
# brownNoise : brown noise generator
# noise : randomly generated numbers
# chaos3D : logistic map diffused in the third dimension
# randomCircles : randomly placed circles
# circleInCircle : circles sorrouned by larger circle
# circlesWithStuff : larger circles with smaller circles around
# randomSizedCircles : randomly sized circles spread around
# voronoiMatrix : voronoi polygons
#
#



import numpy as np
from numpy import genfromtxt
from scipy import integrate
from sklearn import neighbors


def logistic_map(sz=256, A=3.99, seed=36, noise=0):
    """Logistic map.

    X(t+1) = rX(t)(1 - X(t)) + random

    Parameters
    ----------
    sz : int
        Length of the time series.
    A : float
        Parameter for the logistic map. Values beyond 3.56995 exhibit
        chaotic behaviour.
    seed : int
        Sets the random seed for the logistic map.
    noise : float
        Amplitude of noise to add to the logistic map.

    Returns
    -------
    X : 1D array
        Logistic map of size (sz).

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

def noisyPeriodic(sz=256,freq=52,seed=36,noise=.5):
    """ A simple periodic equation with a a specified amplitude of noise.

    X = sin(x) + .5cos(x) + random

    Parameters
    ----------
    sz : int
        Length of the time series.
    freq : int
        Frequency of the periodic equation.
    seed : int
        Sets the random seed for reproducible results.
    noise : float
        Amplitude of the noise.

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

def noisyPeriodic_complicated(sz=256, freq=52, seed=36, noise=.5):
    """A complicated periodic equation with a a specified amplitude of noise.

    X = sin(x) + .5cos(.5x) + .25sin(.25x) + random

    Parameters
    ----------
    sz : int
        Length of the time series.
    freq : int
        Frequency of the periodic equation.
    seed : int
        Sets the random seed for reproducible results.
    noise : float
        Amplitude of the noise.

    Returns
    -------
    X : 1D array
        Returns a 1D periodic equaiton of size (sz) with values between 0 and 1.

    """

    np.random.seed(seed=seed)

    x = np.linspace(0,freq*np.pi,sz)  #prep range for averages

    X = np.sin(x) + .5*np.cos(.5*x) +.25*np.sin(.25*x) + noise*np.random.rand(sz)

    #all positive and between 0 and 1
    X = X + np.abs(np.min(X))
    X = X/np.max(X)

    return X

def noise(sz=256, seed=36):
    """ A random distribution of numbers.

    Parameters
    ----------
    sz : int
        Length of the time series.
    seed : int
        Sets the random seed for reproducible results.

    Returns
    -------
    X : 1D array
        Returns a 1D periodic equaiton of size (sz) with values between 0 and 1.
    """

    np.random.seed(seed=seed)

    X = np.random.rand(sz)

    return X


def lorenz(sz=10000, max_t=100., noise=0):
    """ Integrates the lorenz equations.

    dx/dt = sigma(y - x)
    dy/dt = x(rho - z) - y
    dz/dt = xy - Bz

    sigma=10, beta=8/3, rho=28

    Parameters
    ----------
    sz : int
        Length of the time series to be integrated.
    max_t : float
        Length of time to solve the lorenz equation over,
    noise : float
        Amplitude of noise to be added to the lorenz equation.

    Returns
    -------
    X : 2D array
        X solutions in the first column, Y in the second, and Z in the third.
    """

    def lorenz_deriv(xyz, t0, sigma=10., beta=8./3, rho=28.0):
        x,y,z = xyz
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    x0 = [1, 1, 1]  # starting vector
    t = np.linspace(0, max_t, sz)  # one thousand time steps
    X = integrate.odeint(lorenz_deriv, x0, t) + noise*np.random.rand(sz,3)

    return X


"""
***************
TWO DIMENSIONAL
***************
"""


def chaos2D(sz=128, A=3.99, eps=1., seed=36, noise=None):
    """Logistic map diffused in space.

    Parameters
    ----------
    sz : int
        Row and column size of the spatio-temporal series to be generated.
    A : float
        Parameter for the logistic map. Values beyond 3.56995 exhibit chaotic
        behaviour.
    seed : int
        Sets the random seed for the logistic map. Allows results to be
        easily reproduced.
    noise : float
        Amplitude of noise to add to the logistic map.

    Returns
    -------
    X : 2D array
        Spatio-temporal logistic map of size (sz,sz).

    """

    # set random seed
    np.random.seed(seed=seed)

    X = np.zeros((sz,sz))
    X[0,:] = np.random.rand(1,sz)


    for tt in range(sz-1):
        left_X = np.roll(X,1)  # shift it around for diffusion
        right_X = np.roll(X,-1)

        # logistic equation in space
        reg = X[tt,:] * (1 - X[tt,:])
        left = eps * left_X[tt,:] * ( 1 - left_X[tt,:])
        right = eps * right_X[tt,:] * (1 - right_X[tt,:])

        X[tt+1,:] = (A/(1+2*eps)) * (reg + left + right)

    if noise:
        X += noise * np.random.rand(sz,sz)

    return X

def periodic(sz=128, freq=36, seed=36, noise=0.5):
    """A simple 2D periodic equation with a specified amplitude of noise.

    X = sin(y) + .5cos(x) + random

    Parameters
    ----------
    sz : int
        Length of the time series.
    freq : int
        Frequency of the periodic equation.
    seed : int
        Sets the random seed for reproducible results.
    noise : float
        Amplitude of the noise.

    Returns
    -------
    X : 2D array
        Returns a 2D periodic equaiton of size (sz,sz). Values between 0 and 1.
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


def periodic_brown(sz=128, freq=36, seed=15, noise=1.5):
    """ A periodic equation with a specified amplitude of brown noise.
    Calls the function brownNoise.

    X = sin(y + noise*2pi)

    Parameters
    ----------
    sz : int
        Length of the spatiotemporal series.
    freq : int
        Frequency of the periodic equation.
    noise : float
        Amplitude of the noise.
    seed : int
        Sets the random seed for reproducible results.

    Returns
    -------
    X : 2D array
    	Returns a 2D periodic equaiton of size (sz,sz).
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

def brown_noise(sz=128, num_walks=500, walk_sz=100000, spread=1000, seed=3):
    """Creates brown noise with a bunch of random walks.

    Subsamples to generate sizes: 128, 256, or 512. 512 is the full size.

    Parameters
    ----------
    sz : int
        Row and column size of the array to be returned.
    num_walks : int
        Number of random walks.
    walk_sz : int
        Length of the random walk to take.
    spread : int
        Normal distribution of walks. Sizes randn*spread.
    seed : int
        Sets the random seed for reproducible results.

    Returns
    -------
    X : 2D array
    	Returns a 2D brown noise array size (sz,sz).

    """

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

def noise_2d(sz=128,seed=36):
    """A 2D random distribution of numbers.

    Parameters
    ----------
    sz : int
        row and column size of array.
    seed : int
        Sets the random seed for reproducible results.

    Returns
    -------
    X : 2D array
        Returns a 2D array of size (sz,sz) with values between 0 and 1.

    """

    # set random seed
    np.random.seed(seed=seed)

    X = np.random.rand(sz,sz)

    return X


def chaos_3d(sz=128,A=3.99,eps=1.,steps=100,tstart = 50):
    """Logistic map diffused in space and then taken through time.

    Chaos evolves in 3rd dimension.

    Parameters
    ----------

    sz : int
        Row and column size of the spatio-temporal series to be generated.
    A : float
        Parameter for the logistic map. Values beyond 3.56995 exhibit chaotic
        behaviour.
    eps : float
        Amount of coupling/diffusion between adjecent cells.
    seed : int
        sets the random seed for the logistic map. Allows results to be
        easily reproduced.
    tstart : int
        When to start collecting the data. This allows the chaos to be
        fully developed before collection.

    Returns
    -------
    X : 2D array
        Spatiotemporal logistic map of size (sz,sz,steps)

    """

    X = np.random.rand(sz,sz)
    storeX = []

    for tt in range(tstart + steps + 1):
        left_X = np.roll(X,1,axis=1)  # shift it around for diffusion
        right_X = np.roll(X,-1,axis=1)
        top_X = np.roll(X,1,axis=0)
        bot_X = np.roll(X,1,axis=0)

        # logistic equation in space
        reg = X * (1 - X)
        left = eps * left_X * (1 - left_X)
        right = eps * right_X * (1 - right_X)
        top = eps * top_X * (1 - top_X)
        bot = eps * bot_X * (1 - bot_X)

        X = (A/(1+4*eps)) * (reg + left + right + top + bot)

        if tt > tstart:
            storeX.append(X)

    return np.dstack(storeX)


def randomCircles(sz=256, rad=20., sigma=1, num_circles = 1000):
    """Randomly places down gaussian circles and the sum is taken.

    Calls circleCreate to make the circles

    Parameters
    ----------
    sz : int
        Row and column size of the space.
    rad : float
        Radius of the circles.
    sigma : float
        Constant to create gaussian circles. Changes the distribution of values.
    num_circles : int
        Number of circles to place down randomly.

    Returns
    -------
    X : 2D array
        Returns the summed gaussian circles. Size (sz,sz).

    """

    circ_store = np.empty((sz,sz,num_circles))

    for ii in range(num_circles):

        r = np.floor(np.random.rand()*sz)
        c = np.floor(np.random.rand()*sz)

        circ_store[:,:,ii] = self.circleCreate(r,c,sz,rad,sigma)

    X = circ_store.sum(axis=2)
    X = X/np.max(X)

    return X

def circle_create(r, c, sz, rad, sigma, gauss=True):
    """Places down a single circle in a 2d array.

    Parameters
    ----------
    r,c : int
        Defines the center of the circle.
    sz : int
        Size of the space where the circle will be placed.
    rad : int
        Radius of the circle.
    sigma : float
        Distribution of the values within the circle.
    gauss : bool
        Controls whether to apply a gaussian filter to a cirlce of ones.

    Returns
    -------
    X : 2D array
        Array of size (sz,sz) containing a single circle of size pi*rad^2.

    """

    rad = np.around(np.random.rand()*rad)
    y,x = np.ogrid[-r:sz-r, -c:sz-c]
    mask = x*x + y*y <= rad*rad

    array = np.zeros((sz, sz))
    array[mask] = 1

    if gauss==True:
        array = filt.gaussian_filter(array,sigma)

    return array


def circle_in_circle(rad_list, sz=256, num_blobs=1000):
    """Create circles inside larger circles. These circles cannot overlap.
    Calls the function blobber.

    Parameters
    ----------
    rad_list : list of ints
        Radii of the circles.
    sz : int
        Row and column size of the returned space.
    num_blobs : int
        Number of circles to create.

    Returns
    -------
    blobs : 2D array of shape (sz,sz)
        2D array of the circles within circles.

    """

    blobs = np.zeros((sz,sz))
    blob_count = 0

    for ii in range(num_blobs):

        r = int(np.around(np.random.rand()*sz))
        c = int(np.around(np.random.rand()*sz))

        new_blob = blobber(r,c,rad_list,sz)

        #check to see if there is any overlap

        occupied1 = new_blob>0

        occupied2 = blobs>0

        overlap = np.logical_and(occupied1,occupied2)

        if np.any(overlap)==False:

            blobs+=new_blob

            blob_count+=1

    print('Blobs Generated:',blob_count)

    return blobs


def blobber(r, c, rad_list, sz):
    """Creates a single circle sorrounded by a larger circle.

    To be used within circleInCircle.

    Parameters
    ----------
    r,c : int
        Center of the circles.
    rad_list : list of ints
        Radius of the interior circle.
    sz : int
        Size of the space to generated.

    Returns
    -------
    X : 2D array
    	Array containing a single circle within another circle. Size (sz,sz).

    """

    y,x = np.ogrid[-r:sz-r, -c:sz-c]
    masks = []
    for rad in rad_list:
        mask = x*x + y*y <= rad*rad
        masks.append(mask)

    X = np.zeros((sz, sz))

    # need to flip them so it fills from outside to inside
    for i, mask in enumerate(masks[::-1]):
        X[mask] = i +1

    return X

def blobber2(r, c, rad1, sz):
    """Creates a single circle.

    Parameters
    ----------
    r,c : int
        Center of the circle.
    rad : int
        Radius of the circle.
    sz : int
        Row and column size of the space in which the circle is placed.

    Returns
    -------
    X : 2D array
    	Array containing a single circle. Size (sz,sz).
    """

    y,x = np.ogrid[-r:sz-r, -c:sz-c]
    mask1 = x*x + y*y <= rad1*rad1

    X = np.zeros((sz, sz))
    X[mask1] = 1

    return X

def circlesWithStuff(sz=256, rad1=5, rad2=8, num_blobs=100):
    """Create circles with random smaller circles spread around randomly.

    Same number of large circles and smaller circles. Calls blobber2.

    Parameters
    ----------
    sz : int
        Row and column size of the space in which the circle is placed.
    rad1 : int
        Radius of the smaller circle.
    rad2 : int
        Radius of the larger circle.
    num_blobs : int
        Number of large and small circles to create.

    Returns
    -------
    blobs : 2D array
        Array of size (sz,sz) containing all the circles.

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

def randomSizedCircles(rad_list, val_list, sz=512, num_blobs=3000):
    """
    Create random sized circles spread around randomly and assign them
    to classes: 1:27

    Parameters
    ----------
    sz : int
        Row and column size of the space in which the circle is placed.
    rad_list : list of ints
        List of radii.
    val_list : list ints
        List of values associated with the radii.
    num_blobs : int
        Total number of circles to create.

    Returns
    -------
    blobs : 2D array
        Array containing a all circles.

    """

    blobs = np.zeros((sz,sz))
    blob_count = 0
    rad_store = np.zeros((num_blobs,))
    r_iter = 0
    for ii in range(num_blobs):

        r = np.around(np.random.rand()*sz)
        c = np.around(np.random.rand()*sz)

        rad = rad_list[r_iter]

        new_blob = blobber2(r,c,rad,sz)*val_list[r_iter]

        #check to see if there is any overlap

        occupied = new_blob > 0

        old_occupied = blobs > 0

        overlap = np.logical_and(occupied,old_occupied).any()

        if  (overlap == False):

            blobs+=new_blob
            blob_count+=1

        #update r_iter
        r_iter += 1
        if r_iter == len(rad_list):
            r_iter=0

    print(blob_count)

    return blobs


def voronoiMatrix(sz=512, percent=0.1, num_classes=27):
    """Create voronoi polygons.

    Parameters
    ----------
    sz : int
        Row and column size of the space in which the circle is placed.
    percent : float
        Percent of the space to place down centers of the voronoi polygons.
        Smaller percent makes the polygons larger.
    num_classes : int
        Number of classes to assign to each of the voronoi polygons.

    Returns
    -------
    X : 2D array
        2D array of size (sz,sz) containing the voronoi polygons.
    """


    X = np.zeros((sz,sz))

    #fill in percentage of the space
    locs = np.random.rand(sz,sz)<=percent
    vals = np.random.randint(0,num_classes,size=(sz,sz))
    X[locs]=vals[locs]

    #get all the indices of the matrix
    cc,rr = np.meshgrid(np.arange(0,sz),np.arange(0,sz))

    f = np.zeros((sz**2,2))
    f[:,0]=rr.ravel() #feature1
    f[:,1]=cc.ravel() #feature2

    t = X.ravel() #target

    train_ind = locs.ravel()

    f_train = f[train_ind]
    t_train = t[train_ind]

    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(f_train, t_train)

    preds = clf.predict(f)

    locs = f.astype(int)
    X[locs[:,0],locs[:,1]] = preds

    return X
