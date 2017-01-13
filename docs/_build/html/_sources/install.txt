Install
=================================


pip
^^^
::

  pip install skedm


Requirements
^^^^^^^^^^^^

To simply create the approriate environment, you can use a conda environment.yml file. The file takes the form of:

::

  name: skedm_env
  dependencies:
    - python=3
    - numpy
    - numba
    - scikit-learn
    - pip:
      - skedm

Then you can simply create the environment with:

::

  conda env create -f environment.yml

Contribute, Report Issues, Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To contribute, we suggest making a pull request.

To report issues, we suggest making opening an issue



.. _github: https://github.com/NickC1/skedm
