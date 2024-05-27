# Installation

## Stable version

The latest stable release of `SBMLtoODEjax` can be installed via `pip`:

```bash
pip install sbmltoodejax
```

## Development version

:::{warning}
This version is possibly unstable and may contain bugs.
You can install the latest development version of `SBMLtoODEjax` from source via running the following:
```bash
git clone https://github.com/flowersteam/sbmltoodejax.git
cd sbmltoodejax
pip install -e .
```
:::

:::{tip}
It is recommended to create a conda virtual environment before installing:

```bash
conda create -n sbmltoodejax python=3.9
conda activate sbmltoodejax
```

We also recommend to run the unit tests to you check your installation:
```bash
pytest tests/*
```
:::