
sknla
========

[DOCUMENTATION][sk-nla]

This package is an implementation of nonlinear analysis (nla) using scikit-learn's style. It reconstructs [phase spaces][phase-space] to analyze behavior and make forecasts. The technique is described in depth in [Nonlinear Time Series Analysis by Kantz and Schreiber][nlf-book].

Quick Explanation
-----------------

sknla looks for past configurations of the system that are similar to the present configuration of the system. It then looks at how the system evolved when it was in those similar configurations and uses that knowledge to make forecasts about future evolution. The forecasts are then compared to the actual evolution of the system.


Functionality
-------------
skNLF can forecast both coninuous 1D time series, 2D spatio-temporal patterns, and 2D discrete spatial images. See the included notebooks for full examples.


[sk-nla]: http://nickc1.github.io/sknla/
[phase-space]: https://en.wikipedia.org/wiki/Phase_space)
[nlf-book]: http://www.amazon.com/Nonlinear-Time-Analysis-Holger-Kantz/dp/0521529026/ref=sr_1_8?ie=UTF8&qid=1452278686&sr=8-8&keywords=nonlinear+analysis
[nn-algo]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
[sknn-algo]: http://scikit-learn.org/stable/modules/neighbors.html
[scikit]: http://scikit-learn.org/stable/
[false-nn]: http://www.mpipks-dresden.mpg.de/~tisean/TISEAN_2.1/docs/chaospaper/node9.html
