API Reference
=============

This is the API reference for Brisk, a Python package for training machine learning models.

Package Sections
----------------

:doc:`data`
    Handles data splitting and preprocessing for machine learning models. Provides utilities 
    for creating train-test splits with various strategies and analyzing data distributions.

:doc:`configuration`
    Provides an interface for defining what models should be trained and how they should 
    be trained.

:doc:`evaluation`
    Classes involved in model evaluation, either providing methods to evaluate a model and plot results
    or to define metrics used for evaluation.

.. toctree::
   :maxdepth: 1
   :hidden:

   data
   configuration
   evaluation

API Objects
------------

.. include:: /_api_objects_table.rst
