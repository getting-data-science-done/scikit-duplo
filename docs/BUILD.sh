#!/bin/bash

rm ./source/scikit-duplo.rst
rm ./source/modules.rst

make clean
sphinx-apidoc -o ./source ../scikit-duplo
make html


