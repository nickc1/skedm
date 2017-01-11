import os.path as op
import numpy as np
import numpy.testing as npt
import skedm.utilities as ut



def test_weighted_mean():
    


def test_weighted_mode():

	x = [4, 1, 4, 2, 4, 2]
	weights = [1, 3, 0.5, 1.5, 1, 2]

	M,C = ut.weighted_mode(x, weights)

	npt.assert_equal(M,np.array([2]))
	npt.assert_equal(C,np.array([3.5]))


def test_quick_mode_axis1():

	X = np.array([[2, 1, 3, 1, 3, 3],
				[2, 0, 3, 1, 2, 2],
				[3, 1, 1, 0, 2, 2],
				[3, 1, 2, 3, 1, 1]],dtype=int)

	M = ut.quick_mode_axis1(X)
	npt.assert_equal(M, np.array([ 3.,  2.,  1.,  1.]))


def test_keep_diversity():

	X = np.array([[2, 1, 3, 1, 3, 3],
				[2, 2, 2, 2, 2, 2],
				[3, 1, 1, 0, 2, 2],
				[1, 1, 1, 1, 1, 1]],dtype=int)

	M = ut.keep_diversity(X)
	npt.assert_equal(M, np.array([ True, False,  True, False], dtype=bool) )
