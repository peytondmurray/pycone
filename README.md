# pycone

Pycone is a conifer data analysis toolkit written in Python. Pycone is
maintained by the [Greene research group](https://ffrm.humboldt.edu/people/david-greene)
at Cal Poly Humboldt.

## Installation

`pycone` is a Python package, so you'll need python installed on your system.
Once that's done, clone this repository from the terminal and install `pycone`
as a package:

```bash
git clone https://github.com/peytondmurray/pycone.git
cd pycone
pip install .
```

This will install the necessary dependencies for the project, and allow you to
run the code; the dependencies are specified in `pyproject.toml`.

## Running the code

```python
import pycone

cones, weather = pycone.preprocess.load_data()
mean_t = pycone.analysis.calculate_mean_t(weather)
...
```

The full analysis can be run by simply running

```bash
python -m pycone
```

## Development

To pull in the development dependencies:

```bash
pip install -e .[dev,build] --no-build-isolation
```

Here, `[dev]` adds the optional dependencies that are specified in
`pyproject.toml`, and `-e` installs the package in ["editable"
mode](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e).

We also make use of `--no-build-isolation` because we depend on the `pybind11`
C++ header file for the compiled part of this project. By default, pip will
download build-time dependencies and build the project in an isolated
environment (usually somewhere in `/tmp/`) which is not always guaranteed to
exist in the future. When the code is run and the need for a rebuild is
detected, meson-python will attempt to rebuild the compiled part of the project
using the path originally used: somewhere in `/tmp/`. If this doesn't exist,
you'll run into a confusing build error. So to get around this, we run with
`--no-build-isolation`, which just uses the `pybind11` that gets installed to
your python environment when you do `pip install .[build]`.

## Testing

Install the development dependencies, then run

```bash
pytest
```

to run all the tests. Tests are also run as part of continuous integration (CI),
which runs every time someone makes a pull request.

To measure code coverage, do

```bash
pytest --cov=.
```

to generate a coverage report.

## Package Structure

```mermaid
flowchart LR
    A[Data Ingestion & Cleaning\n`preprocess.py`] --> B[Analysis`analysis.py`]
    B --> C[Plotting\n`output.py`]
    C --> Plots
    C --> Tables
```
