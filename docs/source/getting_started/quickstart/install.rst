.. _install:

============
Installation
============

Python Version
==============

We recommend using the latest stable version of Python. Brisk supports **Python 3.10** and later.

Dependencies
============

Several packages will be installed automatically when you install Brisk. We recommend using 
a virtual environment, such as `venv <https://docs.python.org/3/library/venv.html>`_ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/index.html>`_, to manage dependencies.

- `scikit-learn <https://scikit-learn.org/stable/>`_: provides many of the machine learning tools used in Brisk.
- `pandas <https://pandas.pydata.org/docs/>`_: provides dataframes for working with structured data.
- `numpy <https://numpy.org/doc/stable/>`_: provides support for arrays, matrices, and a wide range of mathematical functions.
- `matplotlib <https://matplotlib.org/stable/contents.html>`_: provides plotting functionality for creating visualizations.
- `seaborn <https://seaborn.pydata.org/>`_: provides a high-level interface for visualizations.
- `jinja2 <https://jinja.palletsprojects.com/en/latest/>`_: provides templating functionality for rendering HTML templates.
- `tqdm <https://tqdm.github.io/>`_: provides progress bars for visualizing the progress of loops and long-running tasks.
- `joblib <https://joblib.readthedocs.io/en/latest/>`_: provides tools for serializing and de-serializing objects.

Install Brisk
=============

Brisk is available on PyPI, so installation is as simple as:

.. code-block:: bash

   pip install brisk-ml


Verify the Installation
========================

You can verify the installation by running the following command:

.. code-block:: bash

   pip show brisk

This will print information about the installed package, including the version number. The version number should be |version|.

Next
====
With Brisk installed your ready to create your first project. See the `Create a Project <create_project.html>`_ page.
