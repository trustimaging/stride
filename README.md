

# Stride - A (somewhat) general optimisation framework for medical ultrasound tomography

[![Build Status](https://github.com/trustimaging/stride/workflows/CI/badge.svg)](https://github.com/trustimaging/stride/actions?query=workflow%3ACI)

Stride is an open-source library for medical ultrasound modelling and tomography. 
It lets users easily prototype medical imaging algorithms with only a few lines of Python code that 
can run seamlessly on a Jupyter notebook, a multi-node CPU cluster or a DGX station with production-grade performance. 
Stride provides end-to-end definition of the imaging process using state-of-the-art reconstruction algorithms, 
and the flexibility to (re)define every step of the optimisation.

- [Documentation](https://strideimaging.readthedocs.io/)


## Quickstart

The recommended way to install Stride is through Anaconda's package manager (version >=4.9), which can be downloaded
in [Anaconda](https://www.continuum.io/downloads) or [Miniconda](https://conda.io/miniconda.html).
A Python version above 3.7 is recommended to run Stride.

To install Stride, follow these steps:

```sh
git clone git@github.com:trustimaging/stride.git
cd stride
conda env create -f environment.yml
conda activate stride
pip install -e .
```


## Running the examples

The easiest way to start working with Stride is to open the Jupyter notebooks under ``examples/stride/breast2D`` 
or ``examples/stride/breast2D``. 

You can also execute the corresponding Python scrips from any terminal. To perform a forward run on the breast2D example:

```sh
cd examples/stride/breast2D
mrun python 03_script_foward.py
```

You can control the number of workers and threads per worker by running:

```sh
mrun -nw 2 -nth 5 python 03_script_foward.py
```

You can configure the devito solvers using environment variables. For example, to run the same code on a GPU with OpenACC you can:

```sh
export DEVITO_COMPILER=pgcc
export DEVITO_LANGUAGE=openacc
export DEVITO_PLATFORM=nvidiaX
mrun -nw 1 -nth 5 python 03_script_foward.py
```

Once you've run anastasio2D forward, you can run the corresponding inverse problem by doing:

```sh
mrun python 04_script_inverse.py
```


## Documentation

The documentation for Stride is available online [here](https://strideimaging.readthedocs.io/).

You can also build and access the documentation by running:

```sh
cd docs
make html
```

and opening the generated ``build/index.html`` in your browser.



## Additional packages

To access the 3D visualisation capabilities, we also recommend installing MayaVi:

```sh
conda install -c conda-forge mayavi
```

and installing Jupyter notebook is recommended to access all the examples:

```sh
conda install -c conda-forge notebook
```


## GPU support

The Devito library uses OpenACC to generate GPU code. The recommended way to access the necessary 
compilers is to install the [NVIDIA HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-downloads).

```sh
wget https://developer.download.nvidia.com/hpc-sdk/21.2/nvhpc_2021_212_Linux_x86_64_cuda_multi.tar.gz
tar xpzf nvhpc_2021_212_Linux_x86_64_cuda_multi.tar.gz
cd nvhpc_2021_212_Linux_x86_64_cuda_multi/
sudo ./install
```

During the installation, select the ``single system install`` option.

Once the installation is done, you can add the following lines to your ``~/.bashrc``:

```sh
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/compilers/bin/:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/compilers/lib/:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/comm_libs/mpi/bin/:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/comm_libs/mpi/lib/:$LD_LIBRARY_PATH
```
