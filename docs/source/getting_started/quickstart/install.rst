.. _install:

============
Installation
============

Python Version
==============

We recommend using the **latest stable** version of Python. Brisk supports **Python 3.10** and later.

Dependencies
============

Several packages will be installed automatically when you install Brisk. We recommend using 
a virtual environment, such as `venv <https://docs.python.org/3/library/venv.html>`_ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/index.html>`_, to manage dependencies.

- `scikit-learn <https://scikit-learn.org/stable/>`_: provides many of the machine learning tools used in Brisk.
- `pandas <https://pandas.pydata.org/docs/>`_: provides dataframes for working with structured data.
- `numpy <https://numpy.org/doc/stable/>`_: provides support for arrays, matrices, and a wide range of mathematical functions.
- `matplotlib <https://matplotlib.org/stable/contents.html>`_: provides plotting functionality for creating visualizations.
- `seaborn <https://seaborn.pydata.org/>`_: provides a high-level interface for visualizations.
- `plotnine <https://plotnine.readthedocs.io/en/stable/>`_: provides a grammar of graphics for creating plots.
- `jinja2 <https://jinja.palletsprojects.com/en/latest/>`_: provides templating functionality for rendering HTML templates.
- `tqdm <https://tqdm.github.io/>`_: provides progress bars for visualizing the progress of loops and long-running tasks.
- `joblib <https://joblib.readthedocs.io/en/latest/>`_: provides tools for serializing and de-serializing objects.
- `openpyxl <https://openpyxl.readthedocs.io/en/stable/>`_: provides tools for working with Excel files.
- `shap <https://shap.readthedocs.io/en/latest/>`_: provides SHAP-based feature importance plots for model interpretability.

Install Brisk
=============

Activate your virtual environment and then install Brisk using pip:

.. code-block:: bash

   pip install brisk-ml


Verify the Installation
========================

You can verify the installation by running the following command:

.. code-block:: bash

   pip show brisk-ml

This will print information about the installed package, including the version number. The version number should be |version|.

With Brisk installed you are ready to create your first project!
