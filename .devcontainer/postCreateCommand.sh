#! /usr/bin/env bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Dependencies
uv sync --all-extras

# Create jupyter kernel
uv run ipython kernel install --user --name=behavior_generation_lecture