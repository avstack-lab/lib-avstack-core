# AVstack CORE

This is the core library of `AVstack`. It is independent of any dataset or simulator. [This preprint][avstack-preprint] accompanies the repository.

## Philosophy

Pioneers of autonomous vehicles (AVs) promised to revolutionize the driving experience and driving safety. However, milestones in AVs have materialized slower than forecast. Two culprits are (1) the lack of verifiability of proposed state-of-the-art AV components, and (2) stagnation of pursuing next-level evaluations, e.g.,~vehicle-to-infrastructure (V2I) and multi-agent collaboration. In part, progress has been hampered by: the large volume of software in AVs, the multiple disparate conventions, the difficulty of testing across datasets and simulators, and the inflexibility of state-of-the-art AV components. To address these challenges, we present `AVstack`, an open-source, reconfigurable software platform for AV design, implementation, test, and analysis. `AVstack` solves the validation problem by enabling first-of-a-kind trade studies on datasets and physics-based simulators. `AVstack` solves the stagnation problem as a reconfigurable AV platform built on dozens of open-source AV components in a high-level programming language.


## Installation

**NOTE:** This currently only works on a Linux distribution (tested on Ubuntu 20.04). It also only works with Python 3.8 (to be expanded in the future).

First, clone the repositry and submodules. If you are not running perception, then you may not have to recurse the submodules.
```
git clone --recurse-submodules https://github.com/avstack-lab/lib-avstack-core.git 
```
Dependencies are managed with [`poetry`][poetry]. This uses the `pyproject.toml` file to create a `poetry.lock` file. It includes an optional `perception` group so that you can install `avstack` without all the large packages necessary for perception. To install poetry, see [this page](https://python-poetry.org/docs/#installation). 


### Without Perception

**NOTE:** for now it is not possible to install without perception. We tried using poetry `groups` with the optional flag but there were unexpected consequences of being unable to install perception when used as a sub-project. This will hopefully be fixed soon.

### With Perception

For installation with perception, run
```
poetry install
```

#### Perception Models

We integrate [mmlab](https://github.com/open-mmlab/)'s `mmdet` and `mmdet3d` as third party submodules for perception. Running perception models requires a GPU! 

At a minimum, you may want to run the provided unit tests. These require `mmdet` and `mmdet3d` perception models from the [`mmdet` model zoo][mmdet-modelzoo] and [`mmdet3d` model zoo][mmdet3d-modelzoo]. To do an automated installation of the necessary models, run:
```
cd models
./download_mmdet_models.sh
./download_mmdet3d_models.sh
```
This will download the models to the `models` folder and will *attempt* to establish a symbolic link for `mmdet` and `mmdet3d`. We provide some error checking, but it is up to you to verify that the symbolic link worked.


### RSS w/ Python Bindings (optional)

To use the safety library, build `ad-lib-rss` with python bindings. Try running `rebuild.sh` in `ad-lib-rss` submodule. Otherwise, follow the instructions [HERE](https://intel.github.io/ad-rss-lib/BUILDING/) and [HERE](https://intel.github.io/ad-rss-lib/ad_rss/ad_rss_python/index.html). Make sure to build with a python version that `poetry` is using for this project (check `pyproject.toml`.). To verify your installation is compatible with `avstack`, once installed, try running
```
poetry run pytest tests/modules/safety
```
and observe the output.


## Running Tests

Since we are using `poetry`, run:
```
poetry run pytest tests
```
These should pass if either: (a) you did not install perception, or (b) if you installed perception *and* downloaded the models. 

# Contributing

See [CONTRIBUTING.md][contributing] for further details.


# LICENSE

Copyright 2022 Spencer Hallyburton

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Logo by [Sydney Jeffs](https://twitter.com/sydney_jeffs).


[avstack-preprint]: https://arxiv.org/pdf/2212.13857.pdf
[poetry]: https://github.com/python-poetry/poetry
[mmdet-modelzoo]: https://mmdetection.readthedocs.io/en/stable/model_zoo.html
[mmdet3d-modelzoo]: https://mmdetection3d.readthedocs.io/en/stable/model_zoo.html
[contributing]: https://github.com/avstack-lab/lib-avstack-core/blob/main/CONTRIBUTING.md
[license]: https://github.com/avstack-lab/lib-avstack-core/blob/main/LICENSE.md
