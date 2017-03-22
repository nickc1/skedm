Score
=====

The next step is to examine the forecast skill. This is done by comparing the actual trajectories to the forecasted trajectories. We will see different patterns in the forecast skill depending on whether the system is dominated by deterministic or noisy dynamics. Consider the two systems below.

.. image:: /_static/edm/scoring_example_ts.png
   :align: center


The top plot is the `logistic map`_, a classic chaotic system in a certain parameter range. There is also a small amount of noise added to the system. The bottom plot is a white noise series. After calculating near neighbors, calculating the forecast and forecast skill, the following plot is produced.


.. image:: /_static/edm/scoring_example.png
   :align: center


The logistic map's :math:`R^2` values fall off as the state space distance of neighbor trajectories used for forcasting is increased. The plot is a little different from what was shown in the :ref:`example` section. Here we are looking at the forecast skill (y-axis) plotted against the average distance to a particular near neighbor (x-axis). To clarify, the first point on the plots above is the average distance to the first near neighbor for all the points in the testing set. For example, if there were 3 samples in our testing set and the first near neighbor to those points had distances [1.3, 4.5, 2.7] respectively. We would say that the average distance for the first near neighbor is:

.. math::

  \frac{.18 + .45 + .27}{3} = .30


This would be plotted against the :math:`R^2` calculated for those three points.

For these series, different trends are apparent. The first is the initial value of the forecast skill. The logistic map has a high forecast skill at low distances in the phase space. The white noise, however, has a forecast skill of near-zero for the first near neighbor. This is to be expected as forecasting a truly noisy system is impossible. Additionally, the logistic map's forecast skill falls off as average distance is increased while the noisy system stays steady around 0.

.. _logistic map: https://www.wikiwand.com/en/Logistic_map
