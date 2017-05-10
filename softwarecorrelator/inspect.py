import numpy
import h5py
import logging
import glob
import matplotlib.pyplot as plt

from .utilities import working_dir
from .lofarhdf5 import h5_structure, h5_complex_voltage_file_names
from .lofarhdf5 import read_timeseries_subsampled
from .stationprocessing import fir_filter_coefficients, channelize_ppf




def data_loss_fraction(dir_name, hdf5_name,
                       detection_threshold=0.01, fractional_error_at_threshold=0.1):
    with working_dir(dir_name):
        h5 = h5py.File(hdf5_name)
        try:
            h5_data_path = [n for n in h5_structure(h5) if 'STOKES' in n][0]
            num_samples = int(1/(detection_threshold*fractional_error_at_threshold**2))
            sub_sampling = int(numpy.product(h5[h5_data_path].shape)//num_samples)
            subset = h5[h5_data_path][::sub_sampling,:]
            return numpy.sum(subset == 0.0)/numpy.product(subset.shape)
        except:
            return 1.0


def data_loss_report_ascii(dir_name):
    file_names = h5_complex_voltage_file_names(dir_name)
    losses = []
    for pol in ['x_re', 'x_im', 'y_re', 'y_im']:
        losses.append(numpy.array([data_loss_fraction(dir_name, file_name)
                                   for file_name in sorted(file_names[pol])]))
    losses = (numpy.array(losses).T).tolist()
    logging.debug(losses)
    print('============================================================================')
    print('                            DATA LOSS [%]')
    print('============================================================================')
    print('NAME                                 X_RE     X_IM     Y_RE     Y_IM     MAX')
    print('----------------------------------------------------------------------------')
    for row in zip(file_names['x_re'], losses):
        print(row[0], '  %5.1f    %5.1f    %5.1f    %5.1f   %5.1f' % (tuple(numpy.array(row[1])*100) + (numpy.array(row[1]).max()*100,)))
    print('============================================================================')




def data_loss_report_plot(dir_name):
    file_names = h5_complex_voltage_file_names(dir_name)
    losses = []
    for pol in ['x_re', 'x_im', 'y_re', 'y_im']:
        losses.append(numpy.array([data_loss_fraction(dir_name, file_name)
                                   for file_name in sorted(file_names[pol])]))
    losses = numpy.array(losses).T
    max_losses = losses.sum(axis=1)
    min_losses = losses.max(axis=1)
    indices = numpy.arange(len(max_losses))
    color_code = ['blue', 'green', 'red']+['red']*80
    plt.barh(indices, max_losses*100, height=0.9,
             color=[color_code[int(loss/0.05)] for loss in max_losses],
             alpha=0.3)
    plt.barh(indices, min_losses*100, height=0.9,
             color=[color_code[int(loss/0.05)] for loss in min_losses],
             alpha=1)
    plt.xlim(0, 100)
    plt.ylim(-0.5, len(min_losses)-0.5)
    plt.yticks(indices, [name.split('_')[1][3:] for name in file_names['x_re']])
    plt.xlabel('Percentage lost')
    plt.ylabel('Antenna')
    plt.title('Data loss %s: %.1f%% total' % (file_names['x_re'][0].split('_')[0], max_losses.mean()*100))
    plt.grid(axis='x')









def subsampled_dynamic_spectra_by_timeslot(dir_name, sas_id=None,
                                           sap_ids=range(48), 
                                           num_chan=256,
                                           num_taps=16,
                                           interval_s=0.1):
    '''
    Return subsampled complex valued dynamic spectra for all antennas
    and subbands in directory `dir_name`.

    **Parameters**

    dir_name : string
        Directory containing the HDF5 files.

    sas_id : None or string
        If None, it is assumed that there is only one SAS ID present
        in the directory. Otherwise, specify the full SAS ID,
        including leading 'L'.

    sap_ids : sequence of int
        The SAP ids that must be read. Default: range(48).

    num_chan : int
        Number of channels in the output. Default: 256.
    
    num_taps : int
        Number of PPF taps. Default: 16.

    interval_s : float
        Number of seconds between output spectra. Default: 0.1.


    **Returns**

    A tuple XX, YY, time_s, freq_axis. Here, XX and YY are 4D
    numpy.arrays of complex64 with indices [antenna, subband,
    time slot, channel], time_s is the time since obervation start for
    each time slot, and freq_axis is a dict containing all the
    frequency information. The subband frequencies are found in
    freq_axis['AXIS_VALUES_WORLD'].
    '''
    fir_coef = fir_filter_coefficients(num_chan, num_taps)
    if sas_id is None:
        with working_dir(dir_name):
            names = glob.glob('*.h5')
        unique_names = numpy.unique([name[:8] for name in names])
        if len(unique_names) == 1 and unique_names[0][0] == 'L':
            try:
                sas_nr = int(unique_names[0].lstrip('L').rstrip('_'))
            except:
                logging.warning('%s might not be valid SAS ID.', unique_names[0].rstrip('_'))
            sas_id = unique_names[0].rstrip('_')
    if sas_id is None:
        raise ValueError('Unable to find SAS ID.')
    time_s = []
    xx = []
    yy = []
    report_interval_s = 2.0
    next_report_s = 0.0
    for i, (x, y, time_axis, freq_axis) in enumerate(read_timeseries_subsampled(
            dir_name, sas_id,
            sap_ids,
            interval_s=interval_s,
            num_samples=num_taps*num_chan)):
        x = x.transpose(0,2,1)
        xx.append([[channelize_ppf(x[ant, sb, :].reshape((num_taps, num_chan)), fir_coef)
                   for sb in range(x.shape[1])]
                   for ant in range(x.shape[0])])
        y = y.transpose(0,2,1)
        yy.append([[channelize_ppf(y[ant, sb, :].reshape((num_taps, num_chan)), fir_coef)
                   for sb in range(y.shape[1])]
                   for ant in range(y.shape[0])])
        time_s.append(time_axis['REFERENCE_VALUE'])

        if time_s[-1] > next_report_s:
            next_report_s += report_interval_s
            print(time_s[-1])
    xx = numpy.array(xx, dtype=numpy.complex64).transpose(1,2,0,3).copy(order='C')
    yy = numpy.array(yy, dtype=numpy.complex64).transpose(1,2,0,3).copy(order='C')
    time_s = numpy.array(time_s)
    return xx, yy, time_s, freq_axis
