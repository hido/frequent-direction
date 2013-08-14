==================
frequent-direction
==================

Implementation of Frequent-Direction algorithm for efficient matrix sketching [Liberty2013]_ .


Usage
=====

Locate ``fd_sketch.py`` into your current directory.
Then run following commands on pyton console:

::

  >>> import fd_sketch
  >>> import numpy as np
  >>> a = np.random.randn(1000, 100)
  >>> b = fd_sketch.sketch(a, 150)
  >>> b
  >>> fd_sketch.calculateErro(a, b)

Run unit test
=============

``fd_sketch_test.py`` contains unit tests for fd\_sketch.

::

  $ python fd_sketch_test.py

More detail
===========

We refer the interested users to the original conference paper for detailed algorithm, theoretial analysis, and performance evaluations.

.. [Liberty2013] E. Liberty, "Simple and Deterministic Matrix Sketching", ACM SIGKDD, 2013. http://www.cs.yale.edu/homes/el327/papers/simpleMatrixSketching.pdf
