# scikit-duplo

Very simple reusable blocks for scikit-learn pipelines  (inspired by scikit-lego)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/scikit-duplo.svg)](https://pypi.org/project/scikit-duplo)
[![Documentation Status](https://readthedocs.org/projects/scikit-duplo/badge/?version=latest)](https://scikit-duplo.readthedocs.io/en/latest/?badge=latest)

# Installation

Installation from the source tree:

```
python setup.py install
```

Or via pip from PyPI:

```
pip install scikit-duplo
```

# Contents

The sci-kit duplo package contains multiple classes that you can use in a sci-kit
learn compatible pipeline. There are ensemble learning classes within the `meta` subdirectory.
These classes expect you to pass in multiple other Sci-kit learn compatible 
machine learning classes. It will use these to build an ensemble of models to
predict the target variable.

There are feature engineering classes inside the `preprocessing` subdirectory. These are
ColumnTransformer compatible classes that expect to receive a dataframe and set of column
names that it will transform for the downstream pipeline processes.



