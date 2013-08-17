==================
frequent-direction
==================

Implementation of Frequent-directions algorithm for efficient matrix sketching [Liberty2013]_ .


Usage
=====

Locate ``fd_sketch.py`` on your current directory.
Run the following commands on pyton console:

::

  >>> import fd_sketch
  >>> import numpy as np
  >>> a = np.random.randn(1000, 100)
  >>> b = fd_sketch.sketch(a, 150)
  >>> b
  >>> fd_sketch.calculateError(a, b)

Run unit test
=============

``fd_sketch_test.py`` contains unit tests for fd\_sketch.

::

  $ python fd_sketch_test.py
  
Run USPS PCA sample
===================

Download the USPS hand-written image dataset from the following URL and extract the archived ``zip.train`` file into ``./data/elem/usps/``.
http://www.ibis.t.u-tokyo.ac.jp/RyotaTomioka/Teaching/enshu13?action=AttachFile&do=get&target=zip.train.gz

Then run the sample script and get the plot figures in ``result`` directory.

::

  $ mkdir result
  $ python sample_usps_pca_rep.py

More details
============

We refer the interested users to the original conference paper for detailed algorithm, theoretial analysis, and performance evaluations.

.. [Liberty2013]  Edo. Liberty, "Simple and Deterministic Matrix Sketching", ACM SIGKDD, 2013. http://www.cs.yale.edu/homes/el327/papers/simpleMatrixSketching.pdf
