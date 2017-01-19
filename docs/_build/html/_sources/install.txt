Install
=======


pip
^^^
::

  pip install skedm


Conda (Recommended)
^^^^^^^^^^^^^^^^^^^

To create a conda environment, you can use the following conda environment.yml file::

  name: skedm_env
  dependencies:
    - python=3
    - numpy
    - numba
    - scikit-learn
    - scipy
    - pip:
      - skedm

Then you can simply create the environment with::

  conda env create -f environment.yml

Contribute, Report Issues, Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To contribute, we suggest making a `pull request`_.

To report issues, we suggest `opening an issue`_.



.. _github: https://github.com/NickC1/skedm
.. _pull request: https://github.com/NickC1/skedm/pulls
.. _opening an issue: https://github.com/NickC1/skedm/issues
