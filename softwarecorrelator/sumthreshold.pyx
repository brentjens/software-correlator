import numpy as np
cimport numpy as np
import numpy.ma as ma
import cython

#@cython.boundscheck(False) # uncommenting this will somewhat speed up end result
def sum_threshold_cython(sequence_ma, threshold_sigma, window_lengths=[1, 2, 4, 8, 16, 32, 64, 128, 256]):
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



def sum_threshold_2d(dynamic_spectrum, threshold_sigma, window_lengths=[1, 2, 4, 8, 16, 32]):
    '''dynamic_spectrum must contain real values and have mean subtracted, for example the abs(vis) or abs(xx-yy)
    **Parameters**

    dynamic_spectrum :

    threshold_sigma :

    window_lengths : list of ints
        Default = [1, 2, 4, 8, 16, 32]
    '''
    num_times, num_freqs = dynamic_spectrum.shape
    window_lengths_time = [wl for wl in window_lengths if wl < num_times]
    window_lengths_freq = [wl for wl in window_lengths if wl < num_freqs]
    print('Flagging by time')
    by_time = ma.array([sum_threshold_cython(row, threshold_sigma, window_lengths_time)
                        for row in dynamic_spectrum.T]).T
    print('Flagging by frequency')
    by_frequency = ma.array([sum_threshold_cython(row, threshold_sigma, window_lengths_freq)
                             for row in  by_time])
    dynamic_spectrum.mask = by_frequency.mask
    return dynamic_spectrum
