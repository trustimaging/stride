========
Download
========

The recommended way to install Stride is through Anaconda's package manager (version >=4.9), which can be downloaded
in:

.. _Anaconda: https://www.continuum.io/downloads
.. _Miniconda: https://conda.io/miniconda.html

A Python version above 3.7 is recommended to run Stride.

To install Stride, follow these steps:

.. code-block:: shell

    git clone git@github.com:trustimaging/stride.git
    cd stride
    conda env create -f environment.yml
    conda activate stride
    pip install -e .
