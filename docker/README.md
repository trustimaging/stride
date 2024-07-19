# Stride docker image library

In order to facilitate the dissemination, usage, and development of [Stride], we provide a series of docker images, based 
on the base images generated and maintained by the [Devito] team. This documentation is based on that provided by [Devito].
For more information on different available architectures and how to configure them, check [their documentation](https://github.com/devitocodes/devito/tree/master/docker)

These images support numerous architectures and compilers and are tagged accordingly. You can find all the available images at 
[DockerHub](https://hub.docker.com/r/stridecodes/). 

## Stride images

We provide two main images:

### CPU images

We provide two CPU images:

- `stridecodes/stride:gcc-latest` with the standard GNU gcc compiler.
- `stridecodes/stride:icc-latest` with the intel C compiler for intel architectures.

To run this image locally, you will need `docker` to be installed. Once available, the following commands will get you started:

```bash
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --add-host=host.docker.internal:host-gateway stridecodes/stride:gcc-latest
```

or to run in user context on a cluster with shared filesystem, you can add the correct user config as docker options e.g.:

```bash
docker run --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) --add-host=host.docker.internal:host-gateway stridecodes/stride:gcc-latest python stride_examples/tutorials/07_script_running.py
```


### GPU images

We also provide three types of images on GPUs:

- `stridecodes/stride:nvidia-nvc-latest` is intended to be used on NVIDIA GPUs. It comes with the configuration to use the `nvc` compiler for `openacc` offloading. 
- `stridecodes/stride:nvidia-clang-latest` is intended to be used on NVIDIA GPUs. It comes with the configuration to use the `clang` compiler for `openmp` offloading.
- `stridecodes/stride:amd-latest` is intended to be used on AMD GPUs. It comes with the configuration to use the `aoompcc` compiler for `openmp` offloading. Additionally, this image can be used on AMD CPUs as well since the Rocm compiler are preinstalled.

#### NVIDIA

To run the NVIDIA GPU version, you will need `nvidia-docker` installed and specify the gpu to be used at runtime. 
See for examples a few runtime commands for the NVIDIA `nvc` images:


```bash
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --add-host=host.docker.internal:host-gateway stridecodes/stride:nvidia-nvc-latest
docker run --gpus all --rm -it --add-host=host.docker.internal:host-gateway stridecodes/stride:nvidia-nvc-latest python stride_examples/tutorials/07_script_running.py
```

or to run in user context on a cluster with shared filesystem, you can add the correct user config as docker options e.g.:

```bash
docker run --gpus all --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) stridecodes/stride:nvidia-nvc-latest python stride_examples/tutorials/07_script_running.py
```


#### AMD

Unlike NVIDIA, AMD does not require an additional docker setup and runs with the standard docker. 
You will however need to pass some flags so that the image is linked to the GPU devices. 
You can find a short walkthrough in these 
[AMD notes](https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter5/Chapter5.3_%20KerasMultiGPU_ROCm.pdf) 
for their tensorflow GPU docker image.


## Build a Stride image

To build the images yourself, all you need is to run the standard build command using the provided Dockerfile. 
The different target architectures will be determined by the base image that you choose.


To build the (default) CPU image, simply run:

```bash
docker build --network=host --file docker/Dockerfile.stride --tag stride .
```

And to build the GPU image with `openacc` offloading and the `nvc` compiler, simply run:

```bash
docker build --build-arg base=devitocodes/bases:nvidia-nvc --network=host --file docker/Dockerfile.stride --tag stride .
```

or if you wish to use the `llvm-15` (clang) compiler with `openmp` offlaoding:

```bash
docker build --build-arg base=devitocodes/bases:nvidia-clang --network=host --file docker/Dockerfile.stride --tag stride .
```

and finally for AMD architectures:

```bash
docker build --build-arg base=devitocodes/bases:amd --network=host --file docker/Dockerfile.stride --tag stride .
```


[Stride]:https://github.com/trustimaging/stride
[Devito]:https://github.com/devitocodes/devito
