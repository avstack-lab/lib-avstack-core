# AVstack CORE

This is the core library of `AVstack`. It is independent of any dataset or simulator. `AVstack` was published and presented at ICCPS 2023 - [find the paper here][avstack-preprint] that accompanies the repository.

## Philosophy

Pioneers of autonomous vehicles (AVs) promised to revolutionize the driving experience and driving safety. However, milestones in AVs have materialized slower than forecast. Two culprits are (1) the lack of verifiability of proposed state-of-the-art AV components, and (2) stagnation of pursuing next-level evaluations, e.g.,~vehicle-to-infrastructure (V2I) and multi-agent collaboration. In part, progress has been hampered by: the large volume of software in AVs, the multiple disparate conventions, the difficulty of testing across datasets and simulators, and the inflexibility of state-of-the-art AV components. To address these challenges, we present `AVstack`, an open-source, reconfigurable software platform for AV design, implementation, test, and analysis. `AVstack` solves the validation problem by enabling first-of-a-kind trade studies on datasets and physics-based simulators. `AVstack` solves the stagnation problem as a reconfigurable AV platform built on dozens of open-source AV components in a high-level programming language.

## Troubleshooting

* If you install poetry but your systems says it is not found, you may need to add the poetry path to your path. On linux, this would be: `export PATH="$HOME/.local/bin:$PATH"`. I recommend adding this to your `.bashrc` or `.zshrc` file.
* Through an ssh connection, poetry may have keyring issues. If this is true, you can run the following: `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`


## Installation
**NOTE:** This currently only works on a Linux distribution (tested on Ubuntu 22.04). It also only works with Python 3.10 (to be expanded in the future).

This package includes extras for perception: `perception` and model serving: `serve` so that you can install `avstack` without all the large packages necessary for perception/serving if you do not need them. 

### Using pip

To be available soon.

### Local Installation

First, clone the repositry and submodules. If you are not running perception, then you may not have to recurse the submodules.
```
git clone --recurse-submodules https://github.com/avstack-lab/lib-avstack-core.git 
```

Dependencies are managed with [`poetry`][poetry]. This uses the `pyproject.toml` file to create a `poetry.lock` file. To install poetry, see [this page](https://python-poetry.org/docs/#installation). 

To install the base package, run:
```
poetry install
```
To install with perception extras, run 
```
poetry install --extras "percep"
```

We provide a `Makefile` that you can inspect and that has a `make install` command pre-defined.


## Initializing Perception Models

We integrate [mmlab](https://github.com/open-mmlab/)'s `mmdet` and `mmdet3d` as third party submodules for perception. Running perception models requires a GPU! 

At a minimum, you may want to run the provided unit tests. These require `mmdet` and `mmdet3d` perception models from the [`mmdet` model zoo][mmdet-modelzoo] and [`mmdet3d` model zoo][mmdet3d-modelzoo]. To do an automated installation of the necessary models, run:
```
cd models
./download_mmdet_models.sh
./download_mmdet3d_models.sh
```
This will download the models to the `models` folder and will *attempt* to establish a symbolic link for `mmdet` and `mmdet3d`. We provide some error checking, but it is up to you to verify that the symbolic link worked.


## Running Tests

Since we are using `poetry`, run:
```
poetry run pytest tests
```
These should pass if either: (a) you did not install perception, or (b) if you installed perception *and* downloaded the models. 

We also provide a `make test` line in the `Makefile` for your convenience.


## (UNDER CONSTRUCTION) Setting up model serving

We use `mmdeploy` to handle model serving. We have included some of the setup in the poetry file, but there is still some degree of manual process that needs to happen on the user's end. We outline an example of how to serve a model here.


### Setting up ONNX Runtime

1. Ensure that you have the poetry plugin to read `.env` files installed. If you do not, run `poetry self add poetry-dotenv-plugin` to install it.
1. In the `deployment/libraries` folder, run `wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz`
1. Untar the file with `tar -zxvf onnxruntime-linux-x64-gpu-1.15.1.tgz`


### Setting up TensorRT Runtime
Optional if you're on an x86 architecture. Not optional if you're on an ARM platform.

1. Ensure that you have the poetry plugin to read `.env` files installed. If you do not, run `poetry self add poetry-dotenv-plugin` to install it.
1. Download the [TensorRT 8.5 GA Update 2 tar file][tensorrt]  and put in the `deployment/libraries` folder. Untar it with e.g., `tar -xvf TensorRT-8.5.3.1*`.
1. Download the [appropriate cudnn file][cudnn] (appropriate meaning it matches the TensorRT compatibility) and put it in the `deployment/libraries` folder. Untar it with e.g., `tar -xvf cudnn-*`. 
1. Download the [appropriate cuda version][cuda] (check the [compatibility matrix][tensorrt_compat]). Not sure yet, but you most likely want to match this to the version of cuda used by `avstack` and `mmdetection`. See the [`pyproject.toml`][toml] file for details.


### Converting a model

Let's assume you have downloaded the perception models using the instructions above. In that case, we've done most of the work for you. 

1. Activate a poetry shell
1. An example conversion procedure is provided in `deployment/mmdeploy/`. Go there and run either `run_test_convert_tensorrt.sh` or `run_test_convert_onnx.sh` depending if you did TensorRT above.
1. If all goes well, you'll be able to serve the model. Try out (in the poetry shell): `python test_model_deployment.py mmdeploy_models/cascade_rcnn_coco` (change the path to the model you converted).


# Contributing

See [CONTRIBUTING.md][contributing] for further details.


# LICENSE

Copyright 2023 Spencer Hallyburton

AVstack specific code is distributed under the MIT License.



[avstack-preprint]: https://arxiv.org/pdf/2212.13857.pdf
[poetry]: https://github.com/python-poetry/poetry
[mmdet-modelzoo]: https://mmdetection.readthedocs.io/en/stable/model_zoo.html
[mmdet3d-modelzoo]: https://mmdetection3d.readthedocs.io/en/stable/model_zoo.html
[tensorrt]: https://developer.nvidia.com/tensorrt-getting-started
[cudnn]: https://developer.nvidia.com/rdp/cudnn-archive
[cuda]: https://developer.nvidia.com/cuda-downloads
[tensorrt_compat]: https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html
[contributing]: https://github.com/avstack-lab/lib-avstack-core/blob/main/CONTRIBUTING.md
[license]: https://github.com/avstack-lab/lib-avstack-core/blob/main/LICENSE.md 
