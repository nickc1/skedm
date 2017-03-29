#
# Generates data for skedm (SciKit Emperical Dynamic Modeling)
#
# By Nick Cortale

import numpy as np
from numpy import genfromtxt
from sklearn import neighbors
import scipy.ndimage


def logistic_map(sz=256, A=3.99, seed=36, noise=0):
    """Solutions to the `logistic map`_ for a given amount of time steps.

    .. math::

      X_{t+1} = AX_t(1 - X_t) + \\alpha\\eta

    where :math:`A` is the parameter that controls chaos, :math:`\eta` is
    uncorrelated white noise and :math:`\\alpha` is the amplitude of the noise.

    Parameters
    ----------
    sz : int
        Length of the time series.
    A : float
        Parameter for the logistic map. Values beyond 3.56995 exhibit
        chaotic behaviour.
    seed : int
        Sets the random seed for numpy's random number generator.
    noise : float
        Amplitude of the noise.

    Returns
    -------
    X : 1D array
        Logistic map of size (sz).


    .. _logistic map : https://en.wikipedia.org/wiki/Logistic_map

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

def noisy_periodic(sz=256, freq=52, noise=.5, seed=36):
    """A simple periodic equation with a a specified amplitude of noise.

    .. math::

      X(t) = sin( 2\pi ft ) + 0.5cos( 2\pi ft ) + \\alpha\\eta

    Where :math:`f` is the frequency :math:`\eta` is uncorrelated white noise
    and :math:`\\alpha` is the amplitude of the noise.

    Parameters
    ----------
    sz : int
        Length of the time series.
    freq : int
        Frequency of the periodic equation.
    noise : float
        Amplitude of the noise.
    seed : int
        Sets the random seed for numpy's random number generator.

    Returns
    -------
    X : 1D array
        Periodic array of size (sz) with values between 0 and 1.

    """

    np.random.seed(seed=seed)

    t = np.linspace(0, freq, sz)  #prep range for averages

    X = np.sin(2*np.pi*t) + .5*np.cos(2*np.pi*t) + noise * np.random.rand(sz)

    #all positive and between 0 and 1
    X = X + np.abs(np.min(X))
    X = X/np.max(X)
    return X

def noise_1d(sz=256, seed=36):
    """White noise with values between 0 and 1. Uses numpy's random number
    generator.

    Parameters
    ----------
    sz : int
        Length of the time series.
    seed : int
        Sets the random seed for numpy's random number generator.

    Returns
    -------
    X : 1D array
        Random array of size (sz) with values between 0 and 1.
    """

    np.random.seed(seed=seed)

    X = np.random.rand(sz)

    return X


def lorenz(sz=10000, max_t=100., noise=0, parameters=(10,8./3,28.0)):
    """Integrates the `lorenz equations`_. Which are defined as:

    .. math::

      \\frac{dx}{dt} = \\sigma (y - x)

      \\frac{dy}{dt} = x(\\rho - z) - y

      \\frac{dz}{dt} = xy - \\beta z

    Where :math:`\\sigma=10`, :math:`\\beta=8/3`, and :math:`rho=28` lead to
    chaotic behavior.

    Parameters
    ----------
    sz : int
        Length of the time series to be integrated.
    max_t : float
        Length of time to solve the lorenz equation over,
    noise : float
        Amplitude of noise to be added to the lorenz equation.
    parameters : tuple
        Sigma, beta, and rho parameters for the lorenz equations.

    Returns
    -------
    X : 2D array
        X solutions in the first column, Y in the second, and Z in the third.


    .. _lorenz equations: https://en.wikipedia.org/wiki/Lorenz_system
    """

    sigma, beta, rho = parameters

    def lorenz_deriv(xyz, t0, sigma=sigma, beta=beta, rho=rho):
        x,y,z = xyz
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    x0 = [1, 1, 1]  # starting vector
    t = np.linspace(0, max_t, sz)  # one thousand time steps
    X = scipy.integrate.odeint(lorenz_deriv, x0, t) + noise*np.random.rand(sz,3)

    return X


"""
***************
TWO DIMENSIONAL
***************
"""


def chaos_2d(sz=128, A=3.99, eps=1., noise=None, seed=36):
    """`Logistic map`_ diffused in space. It takes the following form:

    .. math:: x_{t+1} = A x_t(1-x_t) \\equiv f(x_t)

    .. math:: x_{t+1,s} = \\frac{1}{1+3\\epsilon}[f(x_{t,s})+ \\\
      \\epsilon f(x_{t,s \\pm 1})] +\\alpha\\eta

    Where :math:`A` is the parameter that controls chaos, :math:`\eta` is
    uncorrelated white noise, :math:`\\alpha` is the amplitude of the noise and
    :math:`\\epsilon` is the strength of the spatial coupling.

    Parameters
    ----------
    sz : int
        Row and column size of the spatio-temporal series to be generated.
    A : float
        Parameter for the logistic map. Values beyond 3.56995 exhibit chaotic
        behaviour.
    eps : float
        Spatial coupling strength.
    seed : int
        Sets the random seed for numpy's random number generator.
    noise : float
        Amplitude of the noise.

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

def periodic_2d(sz=128, freq=36, noise=0.5, seed=36):
    """A simple 2D periodic equation with a specified amplitude of noise. This
    is a sine wave down the rows, added to a cosine wave across the columns.

    .. math:: X(r,c) = sin(2\pi fr) + 0.5cos(2\pi fc) + \\alpha\\eta

    Where :math:`r` and :math:`c` are the row and column values, :math:`f`
    is the frequency, :math:`\eta` is uncorrelated white noise and
    :math:`\\alpha` is the amplitude of the noise.

    Parameters
    ----------
    sz : int
        Length of the time series.
    freq : int
        Frequency of the periodic equation.
    noise : float
        Amplitude of the noise.
    seed : int
        Sets the random seed for numpy's random number generator.

    Returns
    -------
    X : 2D array
        2D periodic equaiton of size (sz,sz). Values between 0 and 1.
    """

    # set random seed
    np.random.seed(seed=seed)

    x = np.linspace(0,freq*np.pi,sz)  #prep range for averages
    y = np.linspace(0,freq*np.pi,sz)
    xx,yy = np.meshgrid(x,y)

    X = np.sin(yy) + .5*np.cos(xx)

    X += noise*np.random.rand(sz,sz)


    #normalize
    X += np.abs(X.min())
    X /= X.max()

    return X

def brown_noise(sz=128, num_walks=500, walk_sz=100000, spread=1000, seed=3):
    """Creates `brown noise`_ with a bunch of random walks.

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
        Sets the random seed for numpy's random number generator.

    Returns
    -------
    X : 2D array
        2D brown noise array size (sz,sz).


    .. _brown noise: https://en.wikipedia.org/wiki/Brownian_noise

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


def periodic_brown(sz=128, freq=36, seed=36):
    """A periodic equation with a specified amplitude of `brown noise`_.
    Calls the function brown_noise.

    .. math:: X(r,c) = sin(2\pi fr + \\eta)

    Where :math:`r` and :math:`c` are the row and column values, :math:`f`
    is the frequency, and :math:`\eta` is the brown noise.

    Parameters
    ----------
    sz : int
        Length of the spatiotemporal series.
    freq : int
        Frequency of the periodic equation.
    seed : int
        Sets the random seed for numpy's random number generator.

    Returns
    -------
    X : 2D array
    	Array containing the periodic equation with brown noise added.
    """

    # set random seed
    np.random.seed(seed=seed)

    x = np.linspace(0,freq*np.pi,sz)  #prep range for averages
    y = np.linspace(0,freq*np.pi,sz)
    xx,yy = np.meshgrid(x,y)

    noise = brown_noise(sz=sz)

    X = np.sin(yy+noise*2*np.pi) #+ np.cos(xx+noise*2*np.pi)

    #normalize
    X += np.abs(X.min())
    X /= X.max()

    return X


def noise_2d(sz=128,seed=36):
    """2D array of white noise values between 0 and 1. Uses numpy's random
    number generator.

    Parameters
    ----------
    sz : int
        row and column size of array.
    seed : int
        Sets the random seed for numpy's random number generator.

    Returns
    -------
    X : 2D array
        Array of size (sz,sz) with values between 0 and 1.

    """

    # set random seed
    np.random.seed(seed=seed)

    X = np.random.rand(sz,sz)

    return X

def _gauss_circle_create(r, c, sz, rad, sigma=1, gauss=True):
    """Places down a single circle in a 2d array and then applies a `gaussian
    blur filter` to the single circle. This effictively diffuses the circle.

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


    .. _gaussian blur filter: https://en.wikipedia.org/wiki/Gaussian_blur

    """

    rad = np.around(np.random.rand()*rad)
    y,x = np.ogrid[-r:sz-r, -c:sz-c]
    mask = x*x + y*y <= rad*rad

    array = np.zeros((sz, sz))
    array[mask] = 1

    if gauss==True:
        array = scipy.ndimage.filters.gaussian_filter(array,sigma)

    return array

def overlapping_circles(sz=256, rad=20., sigma=1, num_circles = 1000):
    """Randomly places down circles that have been `gaussian blurred`.
    Overlapping circles are summed together. Uses scipy's gaussian filter.

    Calls _gauss_circle_create to make the circles.

    Parameters
    ----------
    sz : int
        Row and column size of the space.
    rad : float
        Radius of the circles.
    sigma : float
        Constant that controlls the strength of the filter.
    num_circles : int
        Number of circles to place down randomly.

    Returns
    -------
    X : 2D array
        Summed circles. Size (sz,sz).


    .. _gaussian blur filter: https://en.wikipedia.org/wiki/Gaussian_blur

    """

    circ_store = np.empty((sz,sz,num_circles))

    for ii in range(num_circles):

        r = np.floor(np.random.rand()*sz)
        c = np.floor(np.random.rand()*sz)

        circ_store[:,:,ii] = _gauss_circle_create(r,c,sz,rad,sigma)

    X = circ_store.sum(axis=2)
    X = X/np.max(X)

    return X

def _concentric_circle(r, c, rad_list, sz):
    """Creates a single circle sorrounded by a larger circle.

    To be used within concentric_circles. For example, if r=2, c=10,
    rad_list=[2,4,6], and sz=256; _concentric_circle will generate a series of
    concentric circles centered at (2,10) in a 256x256 matrix with the innermost
    radius equal to 2 and the outermost radius equal to six. The values of the
    circles will be 1, 2, and 3 respectively.

    Parameters
    ----------
    r,c : int
        Center of the circles.
    rad_list : list of ints
        Radius of the interior circles. The first value is the innermost and the
        last value is the outermost. So these must be sorted from smallest to
        largest.
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
        X[mask] = i + 1

    return X


def concentric_circles(rad_list, sz=256, num_circs=1000):
    """Create circles inside larger circles. These circles cannot overlap.
    Calls the function _concentric_circle.

    For example, if rad_list=[2,4,6], and sz=256; _concentric_circle will
    generate concentric circles in a 256x256 matrix with the innermost radius
    equal to 2 and the outermost radius equal to six. The values of the circles
    will be 1, 2, and 3 respectively.

    Parameters
    ----------
    rad_list : list of ints
        Radii of the cocentric circles circles. Must be increasing.
    sz : int
        Row and column size of the returned space.
    num_circs : int
        Number of circles to create.

    Returns
    -------
    blobs : 2D array
        Array of integers with shape (sz,sz).

    """

    blobs = np.zeros((sz,sz))
    blob_count = 0

    for ii in range(num_circs):

        r = int(np.around(np.random.rand()*sz))
        c = int(np.around(np.random.rand()*sz))

        new_blob = _concentric_circle(r,c,rad_list,sz)

        #check to see if there is any overlap

        occupied1 = new_blob>0

        occupied2 = blobs>0

        overlap = np.logical_and(occupied1,occupied2)

        if np.any(overlap)==False:

            blobs+=new_blob

            blob_count+=1

    print('Circles Generated:', blob_count)

    return blobs

def _circle_create(r, c, rad1, sz):
    """Creates a single circle.

    Parameters
    ----------
    r, c : int
        Center of the circle.
    rad : int
        Radius of the circle.
    sz : int
        Row and column size of the space in which the circle is placed.

    Returns
    -------
    X : 2D array
    	Array containing a single circle with the circle have a value of 1 and
        the rest of the array zeros. Size (sz,sz).
    """

    y,x = np.ogrid[-r:sz-r, -c:sz-c]
    mask1 = x*x + y*y <= rad1*rad1

    X = np.zeros((sz, sz))
    X[mask1] = 1

    return X

def small_and_large_circles(sz=256, rad1=5, rad2=8, num_circs=1000):
    """Create larger circles with smaller circles spread around randomly.

    Same number of large circles and smaller circles. Calls _circle_create.

    Parameters
    ----------
    sz : int
        Row and column size of the space in which the circle is placed.
    rad1 : int
        Radius of the smaller circle.
    rad2 : int
        Radius of the larger circle.
    num_circs : int
        Number of large and small circles to attempt to create.

    Returns
    -------
    blobs : 2D array
        Array of size (sz,sz) containing all the circles.

	"""

    blobs = np.zeros((sz,sz))
    blob_count = 0

    for ii in range(num_circs):

        r1 = np.around(np.random.rand()*sz)
        c1 = np.around(np.random.rand()*sz)

        r2 = np.around(np.random.rand()*sz)
        c2 = np.around(np.random.rand()*sz)


        new_blob1 = _circle_create(r1,c1,rad1,sz)

        new_blob2 = _circle_create(r2,c2,rad2,sz)*2

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

    print('Number of circles:',blob_count)

    return blobs

def random_sized_circles(rad_list, val_list, sz=512, num_circs=3000):
    """Create random sized circles spread around randomly and assign them
    to classes in val_list.

    For example, rad_list=[1,2,3] and val_list=[4,5,6] will create circles with
    radius 1, 2, and 3 and values 4, 5, and 6 respectively.

    Parameters
    ----------
    sz : int
        Row and column size of the space in which the circle is placed.
    rad_list : list of ints
        List of radii.
    val_list : list ints
        List of values associated with the radii.
    num_circs : int
        Total number of circles to create.

    Returns
    -------
    blobs : 2D array
        Array of integers corresponding to the values given in val_list. Areas
        without a circle will be zero.

    """

    blobs = np.zeros((sz,sz))
    blob_count = 0
    rad_store = np.zeros((num_circs,))
    r_iter = 0
    for ii in range(num_circs):

        r = np.around(np.random.rand()*sz)
        c = np.around(np.random.rand()*sz)

        rad = rad_list[r_iter]

        new_blob = _circle_create(r,c,rad,sz)*val_list[r_iter]

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


def voronoi_matrix(sz=512, percent=0.01, num_classes=27):
    """Create `voronoi polygons`_.

    Parameters
    ----------
    sz : int
        Row and column size of the space.
    percent : float
        Percent of the space to place down centers of the voronoi polygons.
        Smaller percent makes the polygons larger.
    num_classes : int
        Number of classes to assign to each of the voronoi polygons.

    Returns
    -------
    X : 2D array
        2D array of size (sz,sz) containing the voronoi polygons.


    .. _voronoi polygons: https://en.wikipedia.org/wiki/Voronoi_diagram

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


def chaos_3d(sz=128, A=3.99, eps=1., steps=100, tstart = 50):
    """`Logistic map` diffused in space and taken through time. Chaos evolves in
    3rd dimension.

    .. math:: x_{t+1} = A x_t(1-x_t) \\equiv f(x_t)

    .. math:: x_{t+1,r,c} = \\frac{1}{1+4\\epsilon}[f(x_{t,r,c})+ \\\
      \\epsilon f(x_{t,r \\pm 1, c}) + \\epsilon f(x_{t,r, c \\pm 1})]

    Where :math:`A` is the parameter that controls chaos and :math:`\\epsilon`
    is the strength of the spatial coupling.

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
        Sets the random seed for numpy's random number generator.
    tstart : int
        When to start collecting the data. This allows the chaos to be
        fully developed before collection.

    Returns
    -------
    X : 2D array
        Spatiotemporal logistic map of size (sz, sz, steps).

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
