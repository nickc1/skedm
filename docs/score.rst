Score
=====

The next step is to examine the forecast skill. This is done by comparing the actual trajectories to the forecasted trajectories. We will see different patterns in the forecast skill depending if the system is deterministic or noisy. Consider the three systems below.

.. image:: /_static/edm/chaos_rand_noise.png
   :align: center


The top is the [logistic map][logistic-map-wiki]. It is a classic chaotic system. The second is a sine wave with a little bit of noise added. The bottom is white noise. After calculating near neighbors, calculating the forecast and forecast skill, the following plot is produced.


.. image:: /_static/edm/forecast_skill_chaos_periodic_noise.png
   :align: center


Both the logistic map and periodic map :math:`R^2` values fall off as the distance away in the phase space is increased. The sine wave, however, has almost a perfect forecast skill. The plot above is a little different from what was shown in the quick example above. Here we are looking at the forecast skill (y-axis) plotted against the average distance to a particular near neighbor (x-axis). To clarify, the first point on the plots above is the average distance to the first near neighbor for all the points in the testing set. For example, if there were 3 samples in our testing set and the first near neighbor to those points had distances [1.3, 4.5, 2.7] respectively. We would say that the average distance for the first near neighbor is:

.. math::

  \frac{.18 + .45 + .27}{3} = .30


This would be plotted against the :math:`R^2` calculated for those three points.

For these three different series, three different trends are apparent. The first is the initial value of the forecast skill. The logistic map and sine wave both have high forecast skills at low distances in the phase space. The white noise, however, has a forecast skill of zero for the first near neighbor. This is to be expected as forecasting a truly noisy system is impossible.

The difference between the sine wave and the Logistic map is that the forecast skill does not dramatically fall off as a function of distance, nor as a function of prediction distance. The :math:`R^2` value stays high out to a distance of 0.2.
