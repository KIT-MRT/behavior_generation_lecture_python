# Python Code for the Lecture "Verhaltensgenerierung für Fahrzeuge" (Behavior Generation for Vehicles) at KIT

![GitHub CI](https://github.com/KIT-MRT/behavior_generation_lecture_python/actions/workflows/ci.yml/badge.svg) 
![License](https://img.shields.io/github/license/kit-mrt/behavior_generation_lecture_python)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

This repository contains the python code for the lecture ["Verhaltensgenerierung für Fahrzeuge" (Behavior Generation for Vehicles)](https://www.mrt.kit.edu/lehre_WS_Verhaltensgenerierung_Fahrzeuge.php) at KIT.
It is targeted towards both, exemplifying the content of the lecture, and giving a brief introduction to software development. (Please bare with us, the code is largely ported from matlab.)

An API documentation for new parts of the code and exemplary jupyter notebooks can be found in the [documentation](https://kit-mrt.github.io/behavior_generation_lecture_python/).

## Preparing the environment

We encourage the usage of [conda](https://conda.io/) or [virtualenv](https://virtualenv.pypa.io) instead of installing packages to your system directly.

Having activated your environment

- install this package in editable mode: `pip install --editable .`

<details>
<summary>Making venv kernels available to jupyter?</summary>
<br>
<ul>
<li>from without the venv, install ipykernel <code>pip install --user ipykernel</code></li>
<li>list the current venvs available in jupyter: <code>jupyter kernelspec list</code> (your venv is not yet in there)</li>
<li>activate the venv <code>source activate venv-name</code></li>
<li>add the venv to the kernel list: <code>python -m ipykernel install --user --name=venv-name-for-jupyter</code>, where <code>venv-name-for-jupyter</code> can but must not match the name of the activated venv</li>
<li>check that the venv is in the list: <code>jupyter kernelspec list</code></li>
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
