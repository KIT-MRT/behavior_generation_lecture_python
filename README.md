# Python Code for the Lecture Decision-Making and Motion Planning for Automated Driving at KIT

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub CI](https://github.com/KIT-MRT/behavior_generation_lecture_python/actions/workflows/ci.yml/badge.svg)](https://github.com/KIT-MRT/behavior_generation_lecture_python/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/kit-mrt/behavior_generation_lecture_python)](LICENSE)

This repository contains the python code for the lecture [Decision-Making and Motion Planning for Automated Driving](https://www.mrt.kit.edu/english/lehre_WS_Decision-Making_and_Motion_Planning_for_Automated_Driving.php) at KIT.
It is targeted towards both, exemplifying the content of the lecture, and giving a brief introduction to software development. (Please bare with us, the code is largely ported from matlab.)

An API documentation for new parts of the code and exemplary jupyter notebooks can be found in the [documentation](https://kit-mrt.github.io/behavior_generation_lecture_python/).

## Preparing the environment

We use [`uv`](https://docs.astral.sh/uv/) as package and project manager. Having `uv` installed, run

```sh
# clone this repo
git clone https://github.com/KIT-MRT/behavior_generation_lecture_python.git

# change into the repo folder
cd behavior_generation_lecture_python

# set up a virtual env and install the requirements
uv sync
```

<details>
<summary>Making uv kernels available to jupyter?</summary>
<br>
<ul>
<li>create a kernel <code>uv run ipython kernel install --user --name=behavior_generation_lecture</code></li>
<li>run jupyter <code>uv run --with jupyter jupyter lab</code> and chose kernel <code>behavior_generation_lecture</code> in the browser
</ul>
</details>

## Structure

The structure of this repo is inspired by [the PyPA sample project](https://github.com/pypa/sampleproject).

- `src` contains the modules, which is the core implementation, at best browsed in your favorite IDE
- `tests` contains unittests, at best browsed in your favorite IDE
- `scripts` contains scripts that depict exemplary usage of the implemented modules, they can be run from the command line
- `notebooks` contains [jupyter](https://jupyter.org) notebooks, that can be browsed online, and interactively be run using jupyter

## Contribution

Feel free to open an issue if you found a bug or have a request. 
You can also contribute to the lecture code yourself: Just fork this repository and open a pull request.

## License

Unless otherwise stated, this repo is distributed under the 3-Clause BSD License, see [LICENSE](LICENSE).
