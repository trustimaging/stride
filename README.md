

# Stride - A (somewhat) general optimisation framework for medical ultrasound imaging

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

To perform a forward run on the alpha2D example:

```sh
cd examples/stride/alpha2D
mrun python foward.py
```

You can control the number of workers and threads per worker by running:

```sh
mrun -nw 2 -nth 5 python foward.py
```

You can configure the devito solvers using environment variables. For example, to run the same code on a GPU with OpenACC you can:

```sh
export DEVITO_COMPILER=pgcc
export DEVITO_LANGUAGE=openacc
export DEVITO_PLATFORM=nvidiaX
mrun -nw 1 -nth 5 python foward.py
```

Once you've run alpha2D forward, you can run the corresponding inverse problem by doing:

```sh
mrun python inverse.py
```


## Documentation

You can build and access the documentation by running:

```sh
cd docs
make html
```

and opening the generated ``_build/index.html`` in your browser.
