Performance
===========

Here we provide some basic guidelines on performance. All of these were run on an early 2015 macbook air. The code can be found in the accompanying `jupyter notebook`_.

As a baseline, the example given in the Quick Example section completes in 4.2 seconds.


Near Neighbor
^^^^^^^^^^^^^

Here we increase the amount of near neighbors with everything else held constant. This is the time it takes to complete every near neighbor value between the one given on the x axis and 1. For example, the first point is the amount of time it takes to complete the near neighbor calculation for [1,2,3,4,5,6,7,8,9,10] near neighbors.


.. image:: /_static/edm/benchmark_nn_size.png
   :align: center





Train Size
^^^^^^^^^^

Here the testing size, and the number of near neighbors is held constant. We simply iterate through the length of the training set.

.. image:: /_static/edm/benchmark_test_size.png
   :align: center

Test Size
^^^^^^^^^

Here the training size and the number of near neighbors is held constant. We simply iterate through the length of the training set.

.. image:: /_static/edm/benchmark_test_size.png
   :align: center

.. _jupyter notebook: https://github.com/nickc1/skedm/blob/master/scripts/skedm_examples.ipynb
