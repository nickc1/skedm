#Scikit Empirical Dynamic Modeling
# By Nick Cortale

from . import utilities
from scipy import stats as stats
from sklearn import neighbors
import numpy as np
from sklearn import metrics as skmetrics

class Regression:
    """Regression using a k-nearest neighbors method. Predictions can be made
    for each nearest neighbor (`predict_individual`) or by averaging the k
    nearest neighbors (`predict`).

    Parameters
    ----------
    weights : str
        How to weight the near neighbors. Options are:

            - 'uniform' : uniform weighting
            - 'distance' : weighted as 1/distance

    Example
    -------
    >>> X # embed time series of shape (nsamples, embedding dimension)
    >>> y # future trajectory of a point of shape (nsamples, num predictions)
    >>> import skedm as edm
    >>> R = edm.Regression()
    >>> train_len = int(len(X)*.75) # train on 75 percent
    >>> R.fit(X[0:train_len], y[0:train_len])
    >>> preds = R.predict(X[train_len:], [0,10,20]) # test at 1, 10, and 20 nn
    >>> score = M.score(ytest) # Calculate coefficient of determination
    """

    def __init__(self,weights='uniform'):

        self.weights = weights

    def fit(self, Xtrain, ytrain):
        """Fit the training data. Can also be thought of as populating the phase
        space.

        Parameters
        ----------
        Xtrain : 2D array
            Embed training time series. Features shape (nsamples, nfeatures).
        ytrain : 2D array
            Future trajectory of the points. Targets Shape (nsamples,ntargets).

        """

        self.Xtrain = Xtrain
        self.ytrain = ytrain

        max_nn = len(Xtrain)
        # initiate the class and fit the data
        self.knn = neighbors.KNeighborsRegressor(max_nn, weights=self.weights)
        self.knn.fit(Xtrain,ytrain)

    def dist_calc(self,Xtest):
        """Calculates the distance from the testing set to the training set.

        Parameters
        ----------
        Xtest : 2D array
            Test features (nsamples, nfeatures).

        """
        d,i = self.knn.kneighbors(Xtest)

        self.dist = d
        self.ind = i

        self.Xtest = Xtest

    def predict(self,Xtest,nn_list):
        """Make a prediction for a certain value of near neighbors

        Parameters
        ----------
        Xtest : 2d array
            Testing samples of shape (nsamples,nfeatures)
        nn_list : 1d array
            Values of Near Neighbors to use to make predictions

        Returns
        -------
        ypred : 2d array
            Predictions for ytest of shape(nsamples,num predictions).

        """

        #calculate distances first
        self.dist_calc(Xtest)

        ypred = []

        for nn in nn_list:

            neigh_ind = self.ind[:,0:nn]

            if self.weights == 'uniform':

                p = np.mean(self.ytrain[neigh_ind], axis=1)

            elif self.weights =='distance':

                p = np.empty((self.dist.shape[0], self.ytrain.shape[1]), dtype=np.float)

                for i in range(self.ytrain.shape[1]):
                    p[:,i] = utilities.weighted_mean(self.ytrain[neigh_ind,i], self.dist[:,0:nn])

            ypred.append(p)

        self.ypred = ypred
        self.nn_list = nn_list
        return ypred


    def score(self, ytest, how='score'):
        """Score the predictions.

        Parameters
        ----------
        ytest : 2d array
            Target values.
        how : str
            How to score the predictions. Options include:

                -'score' : Coefficient of determination.
                -'corrcoef' : Correlation coefficient.
        Returns
        -------
        score : 2d array
            Scores for the corresponding near neighbors.

        """
        scores = []
        #iterate through each pred for each nn value
        for pred in self.ypred:
            sc = np.empty(pred.shape[1]) #need to store the scores

            for i in range(pred.shape[1]):

                p = pred[:,i]

                if how == 'score':
                    sc[i] = utilities.score(p, ytest[:,i])

                if how == 'corrcoef':

                    sc[i] = utilities.corrcoef(p, ytest[:,i])

            scores.append(sc)

        scores = np.vstack(scores)
        return scores


    def predict_individual(self,Xtest,nn_list):
        """Make a prediction for each neighbor.

        Parameters
        ----------
        Xtest : 2d array
            Contains the test features.
        nn_list : 1d array of ints
            Neighbors to be tested.

        """

        #calculate distances first
        self.dist_calc(Xtest)

        ypred = []


        for nn in nn_list:

            neigh_ind = self.ind[:,nn-1]# subtract 1 since it is zero based

            ypred.append(self.ytrain[neigh_ind])

        self.ypred = ypred

        return ypred

    def dist_stats(self,nn_list):
        """Calculates the mean and std of the distances for the given nn_list.

        Parameters
        ----------
        nn_list : 1d array of ints
            Neighbors to have their mean distance and std returned.

        Returns
        -------
        mean : 1d array
            Mean of the all the test distances corresponding to the nn_list.
        std : 1d array
            Std of all the test distances corresponding to the nn_list.

        """
        nn_list = np.array(nn_list)-1
        d = self.dist[:,nn_list]

        mean = np.mean(d,axis=0)
        std = np.std(d,axis=0)

        return mean, std

class Classification:
    """Classification using a k-nearest neighbors method. Predictions can be
    made for each nearest neighbor (`predict_individual`) or by averaging the k
    nearest neighbors (`predict`).

    Parameters
    ----------
    weights : str
        Procedure to weight the near neighbors. Options:

        - 'uniform' : uniform weighting
        - 'distance' : weighted as 1/distance

    Example
    -------
    >>> X # embed series of shape (nsamples, embedding dimension)
    >>> y # future trajectory of a point of shape (nsamples, num predictions)
    >>> import skedm as edm
    >>> R = edm.Classification()
    >>> train_len = int(len(X)*.75) # train on 75 percent
    >>> R.fit(X[0:train_len], y[0:train_len])
    >>> preds = R.predict(X[train_len:], [0,10,20]) # test at 1, 10, and 20 nn
    >>> score = M.score(ytest) # Calculate klecka's tau
    """

    def __init__(self,weights='uniform'):

        self.weights = weights

    def fit(self, Xtrain, ytrain):
        """Fit the training data. Can also be thought of as reconstructing
        the attractor.

        Parameters
        ----------
        Xtrain : 2D array
            Features of shape (nsamples,nfeatures).
        ytrain : 2D array
            Targets of shape (nsamples,ntargets).
        """
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        max_nn = len(Xtrain)
        self.knn = neighbors.KNeighborsRegressor(max_nn,weights=self.weights,metric='hamming')
        self.knn.fit(Xtrain,ytrain)

    def dist_calc(self, Xtest):
        """Calculates the distance from the testing set to the training
        set.

        Parameters
        ----------
        Xtest : 2d array
            Test features (nsamples, nfeatures).

        """
        d,i = self.knn.kneighbors(Xtest)

        self.dist = d
        self.ind = i

        self.Xtest = Xtest

    def predict(self, Xtest, nn_list):
        """Make a prediction for a certain value of near neighbors

        Parameters
        ----------
        Xtest : 2d array
            Contains the test features.
        nn_list : 1d array of ints
            Neighbors to be tested.

        Returns
        -------
        Ypred : list
            Predictions returned for each nn value in nn_list. It is the same
            length as nn_list.
        """

        self.dist_calc(Xtest)
        xsize = self.dist.shape[0]
        ysize = self.ytrain.shape[1]
        ypred = []

        for nn in nn_list:

            yp = np.empty((xsize,ysize))

            if self.weights =='uniform':

                neigh_ind = self.ind[:,0:nn]

                for j in range(self.ytrain.shape[1]):

                    mode = utilities.quick_mode_axis1_keep_nearest_neigh(
                                        self.ytrain[neigh_ind,j].astype(int))
                    yp[:,j] = mode


            elif self.weights=='distance':
                dist = self.dist[:,0:nn]
                neigh_ind = self.ind[:,0:nn]
                W = 1./(dist+.000001) #to make sure we dont divide by zero

                for j in range(self.ytrain.shape[1]):
                    mode, _ = utilities.weighted_mode(self.ytrain[neigh_ind,j].astype(int), W, axis=1)

                    mode = np.asarray(mode.ravel(), dtype=int)

                    yp[:, j] = mode

            ypred.append(yp)

        self.ypred = ypred

        return ypred

    def predict_individual(self, Xtest, nn_list):
        """Make a prediction for each neighbor.

        Parameters
        ----------
        Xtest : 2d array
            Contains the test features.
        nn_list : 1d array of ints
            Neighbors to be tested.

        Returns
        -------
        Ypred : list
            Predictions returned for each nn value in nn_list. It is the same
            length as nn_list.
        """

        #calculate distances first
        self.dist_calc(Xtest)

        ypred = []


        for nn in nn_list:

            neigh_ind = self.ind[:,nn-1] #subtract 1 since it is zero based

            ypred.append(self.ytrain[neigh_ind])

        self.ypred = ypred

        return ypred


    def score(self, ytest, how='tau'):
        """Evalulate the predictions.

        Parameters
        ----------
        ytest : 2d array
            Contains the target values.
        how : str
            How to score the predictions. Possible values:

                - 'compare' : Percent correctly predicted. For more info, see
                    utilities.class_compare.
                - 'error' : Percent correct scaled by the most common prediction
                    of the series. See utilities.classification_error for more.
                - 'tau' : Kleckas tau

        Returns
        -------
        scores : 2d array
            Scores for the predicted values. Shape (len(nn_list),num_preds)
        """

        num_preds = ytest.shape[1]

        sc = np.empty((1,num_preds))

        scores = []

        for pred in self.ypred:
            sc = np.empty(pred.shape[1])

            for i in range(pred.shape[1]):

                p = pred[:,i]

                if how == 'compare':
                    sc[i] = utilities.class_compare(p,ytest[:,i])

                elif how == 'error':
                    sc[i] = utilities.classification_error(p,ytest[:,i])

                elif how == 'tau':
                    sc[i] = utilities.kleckas_tau(p,ytest[:,i])

            scores.append(sc)

        scores = np.vstack(scores)
        return scores

    def dist_stats(self, nn_list):
        """Returns the mean and std of the distances for the given nn_list
        """

        nn_list = np.array(nn_list)
        d = self.dist[:,nn_list-1]

        mean = np.mean(d,axis=0)
        std = np.std(d,axis=0)

        return mean, std

class Embed:
    """Embed a 1d, 2d array, or 3d array in n-dimensional space. Assists in
    choosing an embedding dimension and a lag value.

    Parameters
    ----------
    X : 1d, 2d, or 3d array
        Array to be embedded in n-dimensional space.
    """

    def __init__(self, X):
        self.X = X

    def mutual_information(self, max_lag):
        """Calculates the mutual information between a time series and a shifted
        version of itself. Uses numpy's mutual information for the calculation.

        Parameters
        ----------
        max_lag : int
            Maximum amount to shift the time series.

        Returns
        -------
        mi : 1d array
            Mutual information values for every shift value. Shape (max_lag,).
        """

        digi = utilities.mi_digitize(self.X)

        mi = np.empty(max_lag)

        for i in range(max_lag):

            ind = i+1
            unshift = digi[ind:]
            shift = digi[0:-ind]

            mi[i] = skmetrics.mutual_info_score(unshift,shift)

        return mi

    def mutual_information_spatial(self, max_lag, percent_calc=.5,
                                    digitize=True):
        """Calculates the mutual information along the rows and down columns at
        a certain number of indices (percent_calc) and returns the sum of the
        mutual informaiton along the columns and along the rows.

        Parameters
        ----------
        M : 2-D array
            Input two-dimensional image.
        max_lag : integer
            Maximum amount to shift the space.
        percent_calc : float
            Percent of rows and columns to use for the mutual information
            calculation.

        Returns
        -------
        R_mut : 1-D array
            The mutual inforation averaged down the rows (vertical).
        C_mut : 1-D array
            The mutual information averaged across the columns (horizontal).
        r_mi : 2-D array
            The mutual information down each row (vertical).
        c_mi : 2-D array
            The mutual information across the columns (horizontal).
        """

        if digitize:
            M = utilities.mi_digitize(self.X)
        else:
            M = self.X

        rs, cs = np.shape(M)

        rs_iters = int(rs*percent_calc)
        cs_iters = int(cs*percent_calc)

        r_picks = np.random.choice(np.arange(rs),size=rs_iters,replace=False)
        c_picks = np.random.choice(np.arange(cs),size=cs_iters,replace=False)


        # The r_picks are used to calculate the MI in the columns
        # and the c_picks are used to calculate the MI in the rows

        c_mi = np.zeros((rs_iters,max_lag))
        r_mi = np.zeros((cs_iters,max_lag))

        for i in range(rs_iters):
            for j in range(max_lag):

                ind = j+1
                unshift = M[r_picks[i],ind:]
                shift = M[r_picks[i],:-ind]
                c_mi[i,j] = skmetrics.mutual_info_score(unshift,shift)

        for i in range(cs_iters):
            for j in range(max_lag):

                ind=j+1
                unshift = M[ind:, c_picks[i]]
                shift = M[:-ind, c_picks[i]]
                r_mi[i,j] = skmetrics.mutual_info_score(unshift,shift)

        r_mut = np.mean(r_mi,axis=0)
        c_mut = np.mean(c_mi,axis=0)

        return r_mut, c_mut, r_mi, c_mi

    def mutual_information_3d(self,max_lag,percent_calc=.5,digitize=True):
        """Calculates the mutual information along the rows and down columns at
        a certain number of indices (percent_calc) and returns the sum of the
        mutual informaiton along the columns and along the rows.

        Parameters
        ----------
        M : 3-D array
            Input three-dimensional array.
        max_lag : integer
            Maximum amount to shift the space.
        percent_calc : float
            Percent of rows and columns to use for the mutual information
            calculation.

        Returns
        -------
        R_mut : 1-D array
            The mutual inforation averaged down the rows (vertical)
        C_mut : 1-D array
            The mutual information averaged across the columns (horizontal)
        Z_mut : 1-D array
            The mutual information averaged along the depth.
        """

        if digitize:
            M = utilities.mi_digitize(self.X)
        else:
            M = self.X

        rs, cs, zs = np.shape(M)

        rs_iters = int(rs*percent_calc)
        cs_iters = int(cs*percent_calc)

        r_picks = np.random.choice(np.arange(rs),size=rs_iters,replace=False)
        c_picks = np.random.choice(np.arange(cs),size=cs_iters,replace=False)


        # The r_picks are used to calculate the MI in the columns
        # and the c_picks are used to calculate the MI in the rows

        c_mi = np.zeros((rs_iters,max_lag))
        r_mi = np.zeros((cs_iters,max_lag))

        for i in range(rs_iters):
            for j in range(max_lag):

                rand_z = np.random.randint(0,zs)
                ind = j+1
                unshift = M[r_picks[i],ind:,rand_z]
                shift = M[r_picks[i],:-ind,rand_z]
                c_mi[i,j] = skmetrics.mutual_info_score(unshift,shift)

        for i in range(cs_iters):
            for j in range(max_lag):

                rand_z = np.random.randint(0,zs)
                ind=j+1
                unshift = M[ind:, c_picks[i],rand_z]
                shift = M[:-ind, c_picks[i],rand_z]
                r_mi[i,j] = skmetrics.mutual_info_score(unshift,shift)

        #for the z dimension
        rs,cs = np.where(np.random.rand(rs,cs)<percent_calc)
        z_mi = np.zeros( (len(rs),max_lag) )

        for i, (rs,cs) in enumerate(zip(r_picks,c_picks)):
            for j in range(max_lag):

                ind=j+1

                unshift = M[rs, cs, ind:]
                shift = M[rs, cs, :-ind]
                z_mi[i,j] = skmetrics.mutual_info_score(unshift,shift)

        r_mut = np.mean(r_mi,axis=0)
        c_mut = np.mean(c_mi,axis=0)
        z_mut = np.mean(z_mi,axis=0)

        return r_mut, c_mut, z_mut


    def embed_vectors_1d(self,lag,embed,predict):
        """Embeds vectors from a one dimensional array in m-dimensional space.

        Parameters
        ----------
        X : array
            A 1-D array representing the training or testing set.
        lag : int
            Lag values as calculated from the first minimum of the mutual info.
        embed : int
            Embedding dimension. How many lag values to take.
        predict : int
            Distance to forecast (see example).

        Returns
        -------
        features : array of shape [num_vectors,embed]
            A 2-D array containing all of the embedded vectors.
        targets : array of shape [num_vectors,predict]
            A 2-D array containing the evolution of the embedded vectors.

        Example
        -------
        >>> X = [0,1,2,3,4,5,6,7,8,9,10]
        >>> em = 3
        >>> lag = 2
        >>> predict=3
        >>> features, targets = embed_vectors_1d(lag, embed, predict)
        >>> features # [[0,2,4], [1,3,5], [2,4,6], [3,5,7]]
        >>> targets # [[5,6,7], [6,7,8], [7,8,9], [8,9,10]]
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

    def embed_vectors_2d(self, lag, embed, predict, percent=0.1):
        """Embeds vectors from a two dimensional image in m-dimensional space.

        Parameters
        ----------
        X : array
            A 2-D array representing the training set or testing set.
        lag : tuple of ints (r,c)
            Row and column lag values (r,c) can think of as (height,width).
        embed : tuple of ints (r,c)
            Row and column embedding shape (r,c) can think of as (height,width).
            c must be odd.
        predict : int
            Distance in the space to forecast (see example).
        percent : float (default = None)
            Percent of the space to embed. Used for computation efficiency.

        Returns
        -------
        features : array of shape [num_vectors,r*c]
            A 2-D array containing all of the embedded vectors.
        targets : array of shape [num_vectors,predict]
            A 2-D array containing the evolution of the embedded vectors.


        Example
        -------
        >>> lag = (3,4)
        >>> embed = (2,5)
        >>> predict = 2
        >>> features, targets = embed_vectors_2d(lag, embed, predict)

        Notes
        -----
        The embed space above looks like the following:

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
        """Embeds vectors from a 3-dimensional matrix in n-dimensional space.

        Parameters
        ----------
        X : array
            A 3-D array representing the training set or testing set.
        lag : tuple of ints (r,c)
            Row and column lag values (r,c) can think of as (height,width).
        embed : tuple of ints (r,c,t)
            Row and column, and time embedding shape (r,c,t) can think of as
            (height,width,time). c must be odd.
        predict : int
            Distance in the space to forecast (see example).
        percent : float (default = None)
            Percent of the space to embed. Used for computation efficiency.

        Returns
        -------
        features : array of shape [num_vectors,r*c]
            A 2-D array containing all of the embedded vectors
        targets : array of shape [num_vectors,predict]
            A 2-D array containing the evolution of the embedded vectors


        Example
        -------
        >>> lag = (3,4,2) #height,width,time
        >>> embed = (3,3)
        >>> predict = 2
        >>> features, targets = embed_vectors_3d(lag, embed, predict)

        Notes
        -----
        The above example would look like the following:

        [f] _ _ _ [f] _ _ _ [f]
         |         |         |
         |         |         |
        [f] _ _ _ [f] _ _ _ [f]
         |         |         |
         |         |         |
        [f] _ _ _ [f] _ _ _ [f]

        The targets would be directly below the center [f].

        """

        rsize = self.X.shape[0]
        csize = self.X.shape[1]
        tsize = self.X.shape[2]

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

            part = self.X[rs : r_end_val, cs : c_end_val, ts : t_end_val ]

            features[ii,:] = part[::r_lag,::c_lag,::t_lag].ravel()

            rs_target = rs + r_lag
            cs_target = cs + c_lag
            targets[ii,:] = self.X[rs + int(r_lag*(rem-1)/2),
                cs+ int(c_lag*(cem-1)/2), t_end_val:t_end_val+predict].ravel()

        return features,targets
