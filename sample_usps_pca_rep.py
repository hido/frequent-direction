# -*- coding: utf-8 -*-
#!/usr/bin/env python

from fd_sketch import sketch

import numpy as np
import numpy.linalg as ln
import math
import matplotlib.cm as cm
import pylab
import sys
import time

""" This is a sample of Frequent Direction method for matrix sketching.
Using USPS hand-written digit recognition dataset, we apply PCA with and without matrix sketching in order to extract two principal components (axes) and plot the data samples onto the two dimensional space.

Reference:
http://www.ibis.t.u-tokyo.ac.jp/RyotaTomioka/Teaching/enshu13
"""

# path to usps file
# downloaded from http://www.ibis.t.u-tokyo.ac.jp/RyotaTomioka/Teaching/enshu13
PATH_TO_USPS = 'data/elem/usps/zip.train'
PATH_TO_SAVE = './result/'

def plot_fig(labels, projected_mat, title, prefix):
    """Compute the degree of error by sketching

    :param labels: list of usps digit labels
    :param projected_mat: n x 2 matrix of data points projected along top-two principle componets
    :param title: title string for figure top
    :param prefix: prefix string for save image file
    """
    # set color map for 10 digits
    colors = cm.rainbow(np.linspace(0, 1, 10))
    
    # calculate min and max for each dimension
    max_values = np.max(projected_mat, axis=0)
    min_values = np.min(projected_mat, axis=0)
    
    # plot figure
    pylab.clf()
    pylab.title(title)
    pylab.xlabel('1st component axis')
    pylab.ylabel('2nd component axis')
    pylab.grid(False)
    pylab.xlim((min_values[0], max_values[0]))
    pylab.ylim((min_values[1], max_values[1]))

    # put each data point as colored digit
    for i in range(0, len(labels)):
        label = labels[i]
        pylab.text(projected_mat[i,0], projected_mat[i,1], str(label), {'color':colors[label], 'fontsize':10})
    
    # save figure as png image
    pylab.savefig(''.join([PATH_TO_SAVE, prefix,  ".png"]))


# load original mat and separate into label and data matrix
original_mat = np.loadtxt(PATH_TO_USPS)
labels = map(int, original_mat[:, 0].tolist())
data_mat = original_mat[:, 1:]
num_sample = len(labels)

# mean shift to zero
mean_sample = np.mean(data_mat, axis=0)
data_mat -= np.kron(np.ones((num_sample, 1)), mean_sample)

# ordinary SVD-based PCA on the original data matrix
# start measuring elapsed time
start_time = time.clock()
# compute SVD
mat_u, vec_sigma, mat_v = ln.svd(data_mat, full_matrices=False)
# stop measuring elapsed time
elapsed = '%3f' % float(time.clock() - start_time)
# preserve original axes
projection_matrix = np.dot(np.dot(mat_v.T, mat_v), mat_v.T)
original_axes = projection_matrix[:, [0, 1]]
# compute projected matrix
projected_mat = mat_u[:, [0, 1]]
plot_fig(labels, projected_mat, 'Brute force (original, elapsed time  = ' + elapsed + 's)', 'brute_force')

# repeat sketch-based PCA with different ell
for ell in [3, 4, 5, 6, 8, 16, 32, 64, 256]:
    # start measuring elapsed time
    start_time = time.clock()
    # create sketch matrix
    sketch_mat = sketch(data_mat, ell)
    # compute SVD
    mat_u, vec_sigma, mat_v = ln.svd(sketch_mat, full_matrices=False)
    # stop measuring elapsed time
    elapsed = '%3f' % float(time.clock() - start_time)
    # compute projection matrix
    projection_matrix = np.dot(np.dot(mat_v.T, mat_v), mat_v.T)
    # make sign of axis identical with original for consistent plot
    if np.dot(original_axes[:, 0], projection_matrix[:,0]) < 0.0:
        projection_matrix[:,0] *= -1
    if np.dot(original_axes[:, 1], projection_matrix[:,1]) < 0.0:
        projection_matrix[:,1] *= -1
    # compute projected matrix
    projected_mat = np.dot(data_mat, projection_matrix)[:, [0, 1]]
    title = 'Sketch (ell = ' + str(ell) + ', elapsed time  = ' + elapsed + 's)'
    prefix = 'fd_sketch_ell_' + str(ell)
    # make and save plot figure
    plot_fig(labels, projected_mat, title, prefix)
