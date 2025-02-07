# Pytorch Template

Template for Pytorch projects using Poetry.


# Getting Started

To make full use of this template, you'll likely want to use `poetry`. If you don't currently have it [head here to install](https://python-poetry.org/). Once you have poetry installed, run:

```bash
poetry install
```

This will install all the necessary dependencies to run this basic template.


# Scripts

In the `scripts.py` file, you'll find a `train()` function. This function runs the main training function found in `./pytorch_template/main.py`. This script is called using poetry by running:

```bash
poetry run train
```

The final piece of this functionality is found in `pyproject.toml` in these lines:

```toml
[tool.poetry.scripts]
train = "scripts:train"
```

To create more scripts to run like so, add a new function to the `scripts.py` file and add a line in the [tool.poetry.scripts] 

# Components

In the `pytorch_template` directory, you'll find several python files.

- `config.py`: functionality to load configuration variables from `config.toml`
- `data.py`: Custom Pytorch Dataset Class. Edit this to suit your data needs.
- `main.py`: Main training script
- `models.py` Custom Model Class. Edit this to suit your modeling needs
- `train.py`: Training and Evaluation loops.
- `utils.py`: Utility functions.

As you develop your project, you'll probably need to edit and extend the classes and functions found in those files. 

# Config

Currently, config vars are read into the main script from the `config.toml` file. To extend configuration variables, edit this file.

# Ruff

This repo uses ruff for formatting and linting. For lint checking:

```bash
poetry run ruff check
```

For formatting:

```bash
poetry run ruff format
```

To configure how ruff operates, see the `[tool.ruff]` sections of `pyproject.toml`.

