Tests
=====

During the development of the package choreo, a large battery of parametrized tests is ran at frequent intervals to ensure the codebase is in working conditions. Running the tests on your local machine is a great way to detect broken or missing features in your local install.

To run tests locally on your machine, first checkout this reposity and install dependencies using pip:

.. code-block:: sh

    git clone git@github.com:gabrielfougeron/choreo.git
    cd choreo
    pip install .[test]

Then, run tests using `pytest <https://docs.pytest.org/en/latest/>`_:

.. code-block:: sh

    pytest
   
Description of tests
--------------------

.. automodule:: tests

Tests results on Github Actions
-------------------------------

As part of the Continuous Integration workflow of choreo, tests are regularly ran on several Python versions and on platforms provided by GitHub Actions. The corresponding test results are listed here.

..

* :doc:`test-report-ubuntu-latest-3.10`
* :doc:`test-report-ubuntu-latest-3.11`
* :doc:`test-report-ubuntu-latest-3.12`
* :doc:`test-report-ubuntu-latest-3.13`

..

* :doc:`test-report-windows-latest-3.10`
* :doc:`test-report-windows-latest-3.11`
* :doc:`test-report-windows-latest-3.12`
* :doc:`test-report-windows-latest-3.13`

..


* :doc:`test-report-macos-latest-3.10`
* :doc:`test-report-macos-latest-3.11`
* :doc:`test-report-macos-latest-3.12`
* :doc:`test-report-macos-latest-3.13`

..

.. needtable::
   :types: Test-File
   :columns: id, cases, passed, skipped, errors, failed

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Tests results on Github Actions
    
    test-report-macos-latest-3.10
    test-report-macos-latest-3.11
    test-report-macos-latest-3.12
    test-report-macos-latest-3.13
    
    test-report-ubuntu-latest-3.10
    test-report-ubuntu-latest-3.11
    test-report-ubuntu-latest-3.12
    test-report-ubuntu-latest-3.13
    
    test-report-windows-latest-3.10
    test-report-windows-latest-3.11
    test-report-windows-latest-3.12
    test-report-windows-latest-3.13


