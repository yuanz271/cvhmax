# cvhmax

[![PyPI - Version](https://img.shields.io/pypi/v/cvhmax.svg)](https://pypi.org/project/cvhmax)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cvhmax.svg)](https://pypi.org/project/cvhmax)

-----

**Table of Contents**

- [Data structure](#data-structure)
- [Installation](#installation)
- [License](#license)

## Data structure
- observation: List[trial]
    - trial: Array(T, obs)
- latent: List[trial]
    - trial: Tuple(m, V)
    - m: Array(T, lat)  # posterior mean
    - V: Array(T, lat, lat)  # posterior covariance

## Installation

```console
pip install git+https://github.com/yuanz271/cvhmax
```

## License

`cvhmax` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
