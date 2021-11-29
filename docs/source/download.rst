========
Download
========

Jump right in using a Jupyter notebook directly in your browser, using `binder <https://mybinder.org/v2/gh/trustimaging/stride/HEAD>`_.

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


Additional packages
-------------------

To access the 3D visualisation capabilities, we also recommend installing MayaVi:

.. code-block:: shell

    conda install -c conda-forge mayavi


and installing Jupyter notebook is recommended to access all the examples:

.. code-block:: shell

    conda install -c conda-forge notebook


GPU support
-----------

The Devito library uses OpenACC to generate GPU code. The recommended way to access the necessary
compilers is to install the `NVIDIA HPC SDK <https://developer.nvidia.com/nvidia-hpc-sdk-downloads>`_.

.. code-block:: shell

    wget https://developer.download.nvidia.com/hpc-sdk/21.2/nvhpc_2021_212_Linux_x86_64_cuda_multi.tar.gz
    tar xpzf nvhpc_2021_212_Linux_x86_64_cuda_multi.tar.gz
    cd nvhpc_2021_212_Linux_x86_64_cuda_multi/
    sudo ./install

During the installation, select the `single system install` option.

Once the installation is done, you can add the following lines to your `~/.bashrc`:

.. code-block:: shell

    export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/compilers/bin/:$PATH
    export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/compilers/lib/:$LD_LIBRARY_PATH
    export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/comm_libs/mpi/bin/:$PATH
    export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/comm_libs/mpi/lib/:$LD_LIBRARY_PATH

