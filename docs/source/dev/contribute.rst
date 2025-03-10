Contributing to Brisk
=====================

Thank you for your interest in contributing to Brisk! This guide outlines the process for contributing to the project.

Getting Started
---------------

1. **Clone the Repository**: ``git clone https://github.com/BFieguth/brisk.git``
2. **Create a Branch**: Create a new branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

3. **Create a Conda Environment**:

   .. code-block:: bash

      conda create -n brisk-dev python==3.12
      conda activate brisk-dev

4. **Install Poetry**:

   .. code-block:: bash

      conda install poetry

5. **Install Development Dependencies**:

   .. code-block:: bash

      poetry install

6. **Make Changes**: Implement your changes, including tests and documentation
7. **Push Your Branch**: Push your changes to the repository:

   .. code-block:: bash

      git push origin feature/your-feature-name

6. **Create a Pull Request**: Create a PR from your branch to the main branch

Issues
------

* All contributions should address an existing issue or create a new one first.
* Use the provided issue templates when creating new issues.
* Issues should clearly describe the problem or enhancement.

Pull Requests
-------------

1. **Create a Branch**: Create a branch for your changes.
2. **Make your Changes**: Implement your changes, including tests and documentation.
3. **Link to Issues**: All PRs must reference at least one issue using the syntax "Fixes #123" or "Relates to #123".
4. **Submit your PR**: Submit your changes for review.

Code Standards
--------------

* Follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_.
* Use `NumPy-style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for all functions and classes.
* Run ``pylint`` using Google's configuration before submitting PRs. A `.pylintrc` file is provided in the root of the repository.
* Include unit tests for new functionality.

Testing
-------

* Run the existing test suite before submitting changes:

  .. code-block:: bash

     pytest

* Ensure all tests pass and coverage is maintained or improved.
* The end-to-end tests can take several minutes to run so you may want to run just the unit tests until you have have your feature working:

  .. code-block:: bash

     pytest tests/unit_tests

Documentation
-------------

* Update or add documentation for any changed or new functionality.
* Build and check documentation locally:

  .. code-block:: bash

     cd docs
     make html

Code Review
-----------

* All PRs require at least one approval before merging.
* Be respectful and constructive in review discussions.

Additional Resources
--------------------

* `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_
* `NumPy Docstring Guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_

Thank you for contributing to Brisk!