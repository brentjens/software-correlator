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






def read_and_process_antenna_worker(h5_names, sap_id,
                                    num_sb, fir_coefficients,
                                    connection):
    r'''
    Read a complex time series from a sequence of four HDF5 groups
    containing, X_re, X_im , Y_re, Y_im, respectively. Read
    num_timeslots starting at first_timeslot. If apply_fn is not None,
    apply it to the resulting time series per sub band and return its
    result.

    **Parameters**
    
    h5_names : sequence strings
        The HDF5 file names of X_re, X_im, Y_re, and Y_im.

    first_timeslot : int
        The first timeslot to read.

    num_timeslots : int
        The number of timeslots to read.

    num_sb : int
        The number of sub bands expected in the data.

    fir_coefficients : 2D numpy.array of float32
        A `num_taps x num_chan` array containing the FIR coefficients,
        where `fir_coefficients.ravel()` should yield the FIR filter to
        multiply with the original (single channel) timeseries data.

    **Returns**
    
    Tuple of x and y numpy arrays(time, sb, channel).

    **Example**
    
    >>> None
    None
    '''
    sap_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/STOKES_%d'
    num_pol = len(h5_names)
    num_taps, num_chan = fir_coefficients.shape

    bandpass = lofar_station_subband_bandpass(num_chan)
    
#    with working_dir(dir_name):
    h5_files = [h5py.File(file_name, mode='r') for file_name in h5_names]
    h5_groups = [h5_file[sap_fmt % (sap_id, pol)]
                 for pol, h5_file in enumerate(h5_files)]
    while True:
        message = connection.recv()
        if message == 'done':
            connection.close()
            [h5_file.close() for h5_file in h5_files]
            break
        first_timeslot, num_timeslots = message
        time_series_real = numpy.zeros((4, num_timeslots, num_sb), dtype=numpy.float32)
        [h5_groups[pol].read_direct(time_series_real,
                                    numpy.s_[first_timeslot:first_timeslot+num_timeslots,:],
                                    numpy.s_[pol, :, :])
         for pol in range(num_pol)]
        time_series_complex_x = time_series_real[0,:,:] + 1j*time_series_real[1,:,:]
        time_series_complex_y = time_series_real[2,:,:] + 1j*time_series_real[3,:,:]

        result_x = numpy.array([channelize_ppf_contiguous_block(
            time_series_complex_x[:, sb].reshape((-1, num_chan)),
            fir_coefficients)/bandpass[numpy.newaxis,:]
                                for sb in range(num_sb)],
                               dtype=numpy.complex64)

        result_y = numpy.array([channelize_ppf_contiguous_block(
            time_series_complex_x[:, sb].reshape((-1, num_chan)),
            fir_coefficients)/bandpass[numpy.newaxis,:]
                                for sb in range(num_sb)],
                               dtype=numpy.complex64)
        connection.send(['x', result_x.shape, result_x.dtype])
        connection.send_bytes(result_x.tobytes())
        connection.send(['y', result_y.shape, result_y.dtype])
        connection.send_bytes(result_y.tobytes())



def time_and_freq_axes(h5_filename, sap_id=0):
    r'''
    '''
    coordinate_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/COORDINATES/COORDINATE_%d'
    h5_file = h5py.File(h5_filename, mode='r')
    time_axis, freq_axis = [
        dict([item
              for item in h5_file[coordinate_fmt %
                                     (sap_id, axis_id)].attrs.items()])
        for axis_id in [0, 1]]
    h5_file.close()
    return time_axis, freq_axis



def read_and_process_antenna_block_mp(dir_name, sas_id_string, sap_ids,
                                      fir_coefficients, interval_s=None,
                                      interval_samples=None, num_samples=256*16,
                                      max_duration_s=None):
    sap_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/STOKES_%d'
    with working_dir(dir_name):
        sap_names = [[('%s_SAP%03d_B000_S%d_P000_bf.h5' % (sas_id_string, sap_id, pol))
                      for pol in [0, 1, 2, 3]]
                     for sap_id in sap_ids]

        first_file = h5py.File(sap_names[0][0], mode='r')
        timeslots_per_file = first_file[sap_fmt % (0, 0)].shape[0]
        first_file.close()
        time_axis, freq_axis = time_and_freq_axes(sap_names[0][0], sap_id=0)
        num_sb = len(freq_axis['AXIS_VALUES_WORLD'])

        sample_duration_s = time_axis['INCREMENT']
        if interval_samples is None:
            samples_per_interval = int(numpy.floor(interval_s/sample_duration_s))
        else:
            samples_per_interval = interval_samples
        first_timeslot = 0

        pipes = [mp.Pipe() for sap_id in sap_ids]
        manager_ends = [pipe[0] for pipe in pipes]
        worker_ends = [pipe[1] for pipe in pipes]
        processes = [mp.Process(target=read_and_process_antenna_worker,
                                args=(h5_names, sap_id, num_sb, fir_coefficients, connection))
                     for h5_names, sap_id, connection in zip(sap_names, sap_ids, worker_ends)]
        [process.start() for process in processes]
        while first_timeslot < timeslots_per_file - samples_per_interval - num_samples:
            time_axis['REFERENCE_VALUE'] = (first_timeslot + num_samples/2)*sample_duration_s
            if max_duration_s is not None and (first_timeslot +num_samples)*sample_duration_s > max_duration_s:
                break
            
            [pipe.send([first_timeslot, num_samples]) for pipe in manager_ends]
            x_metadata = [pipe.recv() for pipe in manager_ends]
            x_data     = [numpy.frombuffer(pipe.recv_bytes(), dtype=x_meta[2]).reshape(x_meta[1])
                          for x_meta, pipe in zip(x_metadata, manager_ends)]
            y_metadata = [pipe.recv() for pipe in manager_ends]
            y_data     = [numpy.frombuffer(pipe.recv_bytes(), dtype=y_meta[2]).reshape(y_meta[1])
                          for y_meta, pipe in zip(y_metadata, manager_ends)]
            first_timeslot += samples_per_interval
            # Return X[sap, sb, time, chan], Y[sap, sb, time, chan], time, freq
            yield (numpy.array(x_data, dtype=numpy.complex64),
                   numpy.array(y_data, dtype=numpy.complex64), time_axis, freq_axis)
        [pipe.send('done') for pipe in manager_ends]
        [pipe.close() for pipe in manager_ends]
        [process.join() for process in processes]
        return None
