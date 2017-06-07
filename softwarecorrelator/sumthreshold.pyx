import numpy as np
cimport numpy as np
import numpy.ma as ma
import cython
from cython.parallel import prange, parallel

import logging

#@cython.boundscheck(False) # uncommenting this will somewhat speed up end result
def sum_threshold_cython_1d(sequence_ma, threshold_sigma, window_lengths=[1, 2, 4, 8, 16, 32, 64, 128, 256]):
    '''
    Expects zero-mean masked array
    '''
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] new_mask = sequence_ma.mask.copy()
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] old_mask = sequence_ma.mask.copy()
    cdef np.ndarray[np.float32_t, ndim=1, cast=True] data = sequence_ma.data.copy()
    cdef double threshold = threshold_sigma
    cdef double z_sum = 0.0
    cdef int index = 0
    cdef int count = 0
    cdef int sequence_length = len(sequence_ma)
    cdef int j = 0
    cdef int w = 0
    for window_length in window_lengths:
        w = window_length
        threshold = threshold_sigma/(float(w)**0.35)
        z_sum = 0.0
        index = 0
        count = 0
        # Initialization
        while index < w:
            if old_mask[index] == 0:
                z_sum += data[index] #ma.sum(sequence_ma[0:window_length])
                count += 1  #sum(1-sequence_ma.mask[0:window_length])
            index += 1
        #print index, z_sum, count
        #index = w
        # Loop until end of sequence
        while index < sequence_length:
            if(abs(z_sum) > threshold*count):
                #print index, z_sum, threshold*count, old_mask[index], old_mask[index-w]
                #print 'flagging %d--%d' % (index - w, index-1)
                j = index - w
                while j < index:
                    new_mask[j] = 1
                    j += 1
            if old_mask[index] == 0:
                z_sum += data[index]
                count += 1
            if old_mask[index-w] == 0:
                z_sum -= data[index-w]
                count -= 1
            index += 1
        for j in range(sequence_length):
            old_mask[j] = new_mask[j]
    #done!
    sequence_ma.mask = np.array(new_mask, dtype=np.bool)
    return sequence_ma



cdef extern from "math.h":
    float fabs(float x) nogil
    double pow (double base, double exponent) nogil


@cython.boundscheck(False) # uncommenting this will somewhat speed up end result
def sum_threshold_cython_2d(data_2d, mask_2d, threshold_sigma, window_lengths=[1, 2, 4, 8, 16, 32, 64, 128, 256]):
    '''
    Expects zero-mean masked array
    flag per row in data_3d_ma
    '''
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] new_mask = mask_2d.copy()
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] old_mask = mask_2d.copy()
    cdef np.ndarray[np.float32_t, ndim=2, cast=True] data = data_2d.data.copy()
    cdef np.ndarray[np.int64_t, ndim=1, cast=True] window_lengths_array = np.array(window_lengths)
    cdef int num_rows        = data_2d.shape[0]
    cdef int sequence_length = data_2d.shape[1]
    cdef int num_windows = window_lengths_array.shape[0]
    
    cdef double threshold_s = threshold_sigma
    cdef double threshold = threshold_sigma
    cdef np.ndarray[np.float64_t, ndim=1, cast=True] z_sum = np.zeros(num_rows)
    cdef np.ndarray[np.int64_t, ndim=1, cast=True] index = np.zeros(num_rows)
    cdef np.ndarray[np.int64_t, ndim=1, cast=True] count = np.zeros(num_rows)
    cdef int row=0
    cdef int window_length=0
    cdef int window_length_index=0
    cdef int j = 0
    cdef int w = 0
    with nogil, parallel():
        for row in prange(num_rows):
            for window_length_index in range(num_windows):
                w = window_lengths_array[window_length_index]
                threshold = threshold_s/(pow(w, 0.35))
                z_sum[row] = 0.0
                index[row] = 0
                count[row] = 0
                # Initialization
                while index[row] < w:
                    if old_mask[row, index[row]] == 0:
                        z_sum[row] += data[row, index[row]] #ma.sum(sequence_ma[0:window_length])
                        count[row] += 1  #sum(1-sequence_ma.mask[0:window_length])
                    index[row] += 1
                #print index, z_sum, count
                #index = w
                # Loop until end of sequence
                while index[row] < sequence_length:
                    if(fabs(z_sum[row]) > threshold*count[row]):
                        #print index, z_sum, threshold*count, old_mask[index], old_mask[index-w]
                        #print 'flagging %d--%d' % (index - w, index-1)
                        j = index[row] - w
                        while j < index[row]:
                            new_mask[row, j] = 1
                            j += 1
                    if old_mask[row, index[row]] == 0:
                        z_sum[row] += data[row, index[row]]
                        count[row] += 1
                    if old_mask[row, index[row]-w] == 0:
                        z_sum[row] -= data[row, index[row]-w]
                        count[row] -= 1
                    index[row] += 1
                for j in range(sequence_length):
                    old_mask[row, j] = new_mask[row, j]
            #done!
    return new_mask





def sum_threshold_2d(dynamic_spectrum, mask, threshold_sigma, window_lengths=[1, 2, 4, 8, 16, 32]):
    '''dynamic_spectrum must contain real values and have mean subtracted, for example the abs(vis) or abs(xx-yy)
    **Parameters**

    dynamic_spectrum : 2D numpy.ndarray of float32
        The dynamic spectrum to flag with indices [timeslot, channel]

    threshold_sigma : float
        Flag anything above this level.

    window_lengths : list of ints
        Default = [1, 2, 4, 8, 16, 32]
    '''
    cdef int num_times = dynamic_spectrum.shape[0]
    cdef int num_freqs = dynamic_spectrum.shape[1]
    window_lengths_time = [wl for wl in window_lengths if wl < num_times]
    window_lengths_freq = [wl for wl in window_lengths if wl < num_freqs]
    by_time_mask = sum_threshold_cython_2d(dynamic_spectrum.T.copy(), mask.T.copy(),
                                           threshold_sigma, window_lengths_time).T.copy()
    by_frequency_mask = sum_threshold_cython_2d(dynamic_spectrum, by_time_mask, threshold_sigma, window_lengths_freq)
    return by_frequency_mask
