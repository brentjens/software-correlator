import numpy
import numpy.fft as fft
import scipy.signal as signal

def fir_filter_coefficients(num_chan, num_taps):
    '''
    Compute FIR filter coefficients for channel separation.

    **Parameters**

    num_chan : int
        Required number of channels in PPF output.

    num_taps : int
        Number of PPF taps.

    **Returns**

    A num_taps x num_chan numpy.array of 32 bit floats.

    **Example**

    >>> fir_filter_coefficients(num_chan=4, num_taps=8)
    array([[-0.00337621,  0.01111862, -0.01466139,  0.00781696],
           [ 0.00988741, -0.02981976,  0.03694931, -0.01888615],
           [-0.0233096 ,  0.06982564, -0.08770566,  0.0466728 ],
           [ 0.06241577, -0.21720791,  0.36907339, -0.46305624],
           [ 0.46305624, -0.36907339,  0.21720791, -0.06241577],
           [-0.0466728 ,  0.08770566, -0.06982564,  0.0233096 ],
           [ 0.01888615, -0.03694931,  0.02981976, -0.00988741],
           [-0.00781696,  0.01466139, -0.01111862,  0.00337621]], dtype=float32)

    '''
    raw_coefficients = signal.firwin((num_taps)*num_chan, 1/(num_chan),
                                     width=0.5/(num_chan))
    auto_fftshift = raw_coefficients*(-1)**numpy.arange(num_taps*num_chan)
    coefficients = numpy.array(auto_fftshift*(num_chan**1.5),
                               dtype=numpy.float32)
    coefficients /= numpy.sqrt(1.8e1),
    return coefficients.reshape((num_taps, num_chan))




def channelize_ppf(blocks, fir_coefficients):
    '''FIR coefficients as well as blocks are num_taps x num_chan arrays'''
    return (fft.ifft((blocks*fir_coefficients).sum(axis=0)))



def channelize_ppf_multi_ts(blocks, fir_coefficients):
    '''FIR coefficients are num_tapsxnum_chan, blocks are num_timeslots x num_taps x num_chan arrays'''
#    return array([channelize_ppf(block, fir_coefficients) for block in blocks], dtype=complex64)
    return (fft.ifft((blocks*fir_coefficients[newaxis,:,:]).sum(axis=1),
                      axis=1))
