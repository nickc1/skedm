import os.path as op
import numpy as np
import numpy.testing as npt
import skedm as edm

def test_dist_calc():

	X = np.array([
			[ 0.3,  0.6],
			[ 0.2,  1.4],
			[ 1.2,  0.2]])
	y = X.sum(axis=1,keepdims=True)

	R = edm.Regression()
	R.fit(X,y)
	R.dist_calc(X)

	d = np.array([[ 0., 0.80622577, 0.98488578],
			[ 0., 0.80622577, 1.56204994],
			[ 0., 0.98488578, 1.56204994]])

	i = np.array([[0, 1, 2],
			[1, 0, 2],
			[2, 0, 1]])

	npt.assert_almost_equal(R.dist, d)
	npt.assert_equal(R.ind, i)
