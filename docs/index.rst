.. skedm documentation master file, created by
   sphinx-quickstart on Thu Jan  5 14:38:25 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

skedm
=================================


|License Type| |Travis CI|

**Scikit Emperical Dynamic Modeling**

Scikit Emperical Dynamic Modeling (skedm) can be used as a way to forecast time series, spatio-temporal 2D or 3D arrays, and even discrete spatial arrangements. More importantly, skedm can provide insight into the underlying dynamics of a system, specifically whether a system is nonlinear and deterministic or whether it is dominated by noise.

For a quick explanation of this package, I suggest checking out the :ref:`example` section as well as the wikipedia article on `nonlinear analysis`_ . Additionally, `Dr. Sugihara's lab`_ has produced some good summary videos of the topic:

1. `Time Series and Dynamic Manifolds`_
2. `Reconstructed Shadow Manifold`_


For a more complete background, I suggest checking out `Nonlinear Analysis by Kantz`_ as well as `Practical implementation of nonlinear time series methods: The TISEAN package`_.

This software is useful both for forecasting and exploring underlying dynamical processes in a broad range of systems.  The target audience is also wide-ranging as the software can be used to explore any dynamical system.  Previous work using similar analyses has explored `ecological systems`_, `physical systems`_, and `physiological applications`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quick-example
   generate-data
   embed
   predict
   score
   module-reference
   performance
   acknowledgements


.. _Scikit Emperical Dynamic Modeling: https://github.com/NickC1/skedm
.. _Nonlinear Analysis by Kantz: https://www.amazon.com/Nonlinear-Time-Analysis-Holger-Kantz/dp/0521529026/ref=sr_1_1?s=books&ie=UTF8&qid=1475599671&sr=1-1&keywords=nonlinear+time+series+analysis

.. _Practical implementation of nonlinear time series methods\: The TISEAN package : http://scitation.aip.org/content/aip/journal/chaos/9/2/10.1063/1.166424

.. _example: http://skedm.readthedocs.io/en/latest/example.html
.. _nonlinear analysis: https://www.wikiwand.com/en/Nonlinear_functional_analysis

.. _dr. sugihara's lab: http://deepeco.ucsd.edu/

.. _Time Series and Dynamic Manifolds: https://www.youtube.com/watch?v=fevurdpiRYg

.. _Reconstructed Shadow Manifold: https://www.youtube.com/watch?v=rs3gYeZeJcw

.. _phase spaces: https://github.com/ericholscher/reStructuredText-Philosophy

.. _ecological systems: http://deepeco.ucsd.edu/~george/publications/90_nonlinear_forecasting.pdf

.. _physical systems: http://aip.scitation.org/doi/abs/10.1063/1.4931801

.. _physiological applications: http://www.pnas.org/content/93/6/2608.short

.. |License Type| image:: https://img.shields.io/github/license/mashape/apistatus.svg
    :target: https://github.com/NickC1/skedm/blob/master/LICENSE
.. |Travis CI| image:: https://travis-ci.org/nickc1/skedm.svg?branch=master
    :target: https://travis-ci.org/NickC1/skedm
