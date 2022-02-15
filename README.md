
<div align="center">
<img src="docs/source/_static/stride_logo.png" width="400" style="max-width:100%; margin:0 auto; display:block;" alt="logo"></img>
</div>

# Stride - A modelling and optimisation framework for medical ultrasound

[![Build Status](https://github.com/trustimaging/stride/workflows/CI/badge.svg)](https://github.com/trustimaging/stride/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/stridecodes/badge/?version=latest)](https://stridecodes.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/trustimaging/stride/HEAD)



[Stride](https://www.stride.codes) is an open-source library for ultrasound modelling and tomography that provides flexibility and scalability 
together with production-grade performance.

[**Quickstart**](#quickstart)
| [**Tutorials**](https://github.com/trustimaging/stride/tree/master/examples/tutorials)
| [**Other examples**](#running-the-examples)
| [**Additional packages**](#additional-packages)
| [**GPU support**](#gpu-support)
| [**Documentation**](https://stridecodes.readthedocs.io/en/latest/)


## Key features

#### High-performance modelling

We provide high-performance, finite-difference, time-domain solvers for modelling ultrasound propagation in the human body, 
including:

- Variable speed of sound, density, and attenuation.
- Off-grid sources and receivers.
- A variety of absorbing boundary conditions.
- Targeting both CPUs and GPUs with the same code.

#### Intuitive inversion algorithms

Stride also lets users easily prototype medical tomography algorithms with only a few lines of Python code by providing:
 
- Composable, automatic gradient calculations. 
- State-of-the-art reconstruction algorithms. 
- The flexibility to (re)define every step of the optimisation.

#### Flexibility

Solvers in Stride are written in [Devito](https://www.devitoproject.org/), using math-like symbolic expressions. This means
that anyone can easily add new physics to Stride, which will also run on both CPUs and GPUs.

#### Scalability

Stride can scale seamlessly from a Jupyter notebook in a local workstation, to a multi-node CPU cluster or a GPU cluster 
with production-grade performance.


## Quickstart

Jump right in using a Jupyter notebook directly in your browser, using [binder](https://mybinder.org/v2/gh/trustimaging/stride/HEAD).

Otherwise, the recommended way to install Stride is through Anaconda's package manager (version >=4.9), which can be downloaded
in [Anaconda](https://www.continuum.io/downloads) or [Miniconda](https://conda.io/miniconda.html).
A Python version above 3.7 is recommended to run Stride.

To install Stride, follow these steps:

```sh
git clone https://github.com/trustimaging/stride.git
cd stride
conda env create -f environment.yml
conda activate stride
pip install -e .
```


## Running the examples

The easiest way to start working with Stride is to open the Jupyter notebooks under 
[examples/tutorials](https://github.com/trustimaging/stride/tree/master/examples/tutorials). 

You can also check fully worked examples of breast imaging in 2D and 3D under 
[examples/breast2D](https://github.com/trustimaging/stride/tree/master/examples/examples/breast2D) and 
[examples/breast2D](https://github.com/trustimaging/stride/tree/master/examples/examples/breast3D).
To perform a forward run on the breast2D example, you can execute from any terminal:

```sh
cd examples/examples/breast2D
mrun python 01_script_foward.py
```

You can control the number of workers and threads per worker by running:

```sh
mrun -nw 2 -nth 5 python 01_script_forward.py
```

You can configure the devito solvers using environment variables. For example, to run the same code on a GPU with OpenACC you can:

```sh
export DEVITO_COMPILER=pgcc
export DEVITO_LANGUAGE=openacc
export DEVITO_PLATFORM=nvidiaX
mrun -nw 1 -nth 5 python 01_script_forward.py
```

Once you've run it forward, you can run the corresponding inverse problem by doing:

```sh
mrun python 02_script_inverse.py
```

You can also open our interactive Jupyter notebooks in the public [binder](https://mybinder.org/v2/gh/trustimaging/stride/HEAD).

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

To run a solver using the GPU, simply add the option ``platform="nvidia-acc"``:

```python
pde = IsoAcousticDevito(...)
await pde(..., platform="nvidia-acc")
```

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

## Citing Stride

If you use Stride in your research, please cite our [paper](https://arxiv.org/abs/2110.03345):

```
@misc{cueto2021-stride,
	title          =    { Stride: a flexible platform for high-performance ultrasound computed tomography  },
	author         =    { Carlos Cueto and Oscar Bates and George Strong and Javier Cudeiro and Fabio Luporini
				and Oscar Calderon Agudo and Gerard Gorman and Lluis Guasch and Meng-Xing Tang },
	year           =    { 2021 },
	eprint         =    { 2110.03345 },
	archivePrefix  =    { arXiv },
	primaryClass   =    { physics.med-ph },
	url            =    { https://arxiv.org/abs/2110.03345 }
}
```


## Contact us

Join the [conversation](https://join.slack.com/t/stridecodes/shared_invite/zt-xr1dlqv7-Lesu9nFYOqF~AjA6VPUdhw) 
to share your projects, contribute, and get your questions answered.


## Documentation

For details about the Stride API, check our [latest documentation](https://stridecodes.readthedocs.io/en/latest/).
