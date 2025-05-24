# cvhmax

[![PyPI - Version](https://img.shields.io/pypi/v/cvhmax.svg)](https://pypi.org/project/cvhmax)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cvhmax.svg)](https://pypi.org/project/cvhmax)

-----

**Table of Contents**

- [Installation](#installation)
- [Data structure](#data-structure)
- [License](#license)

## Installation

```console
pip install git+https://github.com/yuanz271/cvhmax
```

## Data structure

- observation
    - y: observation, Array(trial, bin, obs dim)
    - ymask: missing value mask, Array(trial, bin, obs dim)
        - 0: missing, 1: normal
        - pad unequal trials with missing values
- latent
    - m: posterior mean, Array(trial, bin, lat dim)
    - V: posterior covariance, Array(trial, bin, lat dim)

## License

`cvhmax` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
