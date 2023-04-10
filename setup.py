# -*- coding: utf-8 -*-
 
"""setup.py: setuptools control."""
 
from setuptools import setup, find_packages
import re
 
version = re.search(
        '^__version__\s*=\s*"(.*)"',
        open('skduplo/__init__.py').read(),
        re.M
    ).group(1)
 
with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

setup(
      license="MIT",
      name = "scikit-duplo",
      packages=find_packages(exclude=["notebooks", "tests"]),
      install_requires=[
        'pandas','numpy','scikit-learn'
      ],
      include_package_data=True,
      version = version,
      description = "Sci-kit learn tools for machine learning pipelines",
      long_description = long_descr,
      long_description_content_type='text/markdown',
      author = "John Hawkins",
      author_email = "johnc@getting-data-science-done.com",
      url = "http://getting-data-science-done.com",
      project_urls = {
          'Documentation': "http://scikit-duplo.readthedocs.io",
          'Source': 'https://github.com/getting-data-science-done/scikit-duplo',
          'Tracker': 'https://github.com/getting-data-science-done/scikit-duplo/issues',
      },
    )


