BenchOpt Benchmark for the Lasso Path
=====================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and reproducible
the comparisons of optimization algorithms. This benchmark is dedicated to
benchmarking algorithms that solve the full lasso path, that is, solving

.. math::

    \min_w \frac{1}{2} \|y - Xw\|^2_2 + \lambda \|w\|_1

for a sequence of :math:`\lambda` values.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/jolars/benchmark_lasso_path
   $ benchopt run benchmark_lasso_path

Apart from the problem, options can be passed to `benchopt run`, to restrict
the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	benchopt run benchmark_lasso_path -s celer -d simulated --max-runs
	10 --n-repetitions 10

Use `benchopt run -h` for more details about these options, or visit
https://benchopt.github.io/api.html.

.. |Build Status| image::
   https://github.com/jolars/benchmark_lasso_path/workflows/Tests/badge.svg
   :target: https://github.com/jolars/benchmark_lasso_path/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
