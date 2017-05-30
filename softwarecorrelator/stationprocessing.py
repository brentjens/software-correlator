import numpy
import multiprocessing as mp
import scipy.fftpack as fft
import scipy.signal as signal

import h5py
from .utilities import working_dir
from .stationbandpass import lofar_station_subband_bandpass

def fir_filter_coefficients(num_chan, num_taps, cal_factor=1./50.0):
    '''
    Compute FIR filter coefficients for channel separation.

    **Parameters**

    num_chan : int
        Required number of channels in PPF output.

    num_taps : int
        Number of PPF taps.

    **Returns**

    A num_taps x num_chan numpy.array of float32.

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
    coefficients = numpy.array(auto_fftshift*(num_chan**0.5),
                               dtype=numpy.float32)
    coefficients *= cal_factor
    return coefficients.reshape((num_taps, num_chan))




def channelize_ppf(timeseries_taps, fir_coefficients):
    '''
    Make a polyphase-filtered spectrum of a timeseries.
    
    **Parameters**

    timeseries_taps : 2D numpy.array of complex64
        A `num_taps x num_chan` array containing the timeseries data,
        where `timeseries_taps.ravel()` should yield the input (single
        channel) timeseries data.

    fir_coefficients : 2D numpy.array of float32
        A `num_taps x num_chan` array containing the FIR coefficients,
        where `fir_coefficients.ravel()` should yield the FIR filter to
        multiply with the original (single channel) timeseries data.

    **Returns**

    A 1D numpy.array of complex64 with length num_chan containing the
    PPF output.

    **Example**
    
    >>> fir = fir_filter_coefficients(num_chan=4, num_taps=2, cal_factor=1)
    >>> fir.dtype
    dtype('float32')
    >>> timeseries = numpy.array(numpy.exp(2j*numpy.pi*2.8*numpy.arange(8)),
    ...                          dtype=numpy.complex64)
    >>> timeseries
    array([ 1.000000 +0.00000000e+00j,  0.309017 -9.51056540e-01j,
           -0.809017 -5.87785244e-01j, -0.809017 +5.87785244e-01j,
            0.309017 +9.51056540e-01j,  1.000000 -3.42901108e-15j,
            0.309017 -9.51056540e-01j, -0.809017 -5.87785244e-01j], dtype=complex64)
    >>> spectrum = channelize_ppf(timeseries.reshape(fir.shape), fir)
    >>> spectrum
    array([-0.03263591-0.01060404j, -0.00383157+0.00195229j,
           -0.00848089+0.02610143j,  0.78864020+1.54779351j], dtype=complex64)
    '''
    return (fft.fft((timeseries_taps*fir_coefficients).sum(axis=0)))



def channelize_ppf_multi_ts(timeseries_taps, fir_coefficients):
    '''FIR coefficients are num_taps x num_chan, blocks are num_timeslots x num_taps x num_chan arrays'''
    return (fft.fft((timeseries_taps*fir_coefficients[numpy.newaxis,:,:]).sum(axis=1),
                    axis=1))



def channelize_ppf_contiguous_block(timeseries_taps, fir_coefficients):
    num_taps, num_chan = fir_coefficients.shape
    num_ts_blocks = timeseries_taps.shape[0]
    num_spectra = num_ts_blocks -(num_taps-1)
    output_spectra = numpy.zeros((num_spectra, num_chan),
                                 dtype=numpy.complex64)
    for sp in range(num_spectra):
        output_spectra[sp,:] += channelize_ppf(timeseries_taps[sp:sp+num_taps,:],
                                               fir_coefficients)
    return output_spectra




def samples_per_block(block_length_s, sample_duration_s, num_chan, num_taps):
    r'''
    Calculate the number of samples per correlator intergration time,
    as well as the number of samples that must be read. The latter is
    larger because a certain number of samples before and after the
    actual interval must be read to properly fill the PPF.

    **Parameters**

    block_length_s : float
        Number of seconds per correlator interval.

    sample_duration_s : float
        Number of seconds per sample in the time series data.

    num_chan : int
        Number of channels for the PPF.

    num_taps : int
        Number of taps in the PPF


    **Returns**

    Tuple (block_length samples, samples_to_read_per_block). Both
    integers.

    **Examples**

    >>> block_length_samples, samples_to_read = samples_per_block(0.1, 1024/200e6, num_chan=256, num_taps=16)
    >>> block_length_samples, block_length_samples/256, samples_to_read/256
    (19456, 76.0, 91.0)
    >>> print(block_length_samples*1024/200e6, ' seconds')
    0.09961472  seconds

    '''
    num_spectra = int(round(block_length_s/sample_duration_s/num_chan))
    block_length_samples = num_spectra*num_chan 
    samples_to_read_per_block = (num_spectra+(num_taps-1))*num_chan
    return block_length_samples, samples_to_read_per_block
