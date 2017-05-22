import numpy
import h5py
import logging
import glob
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .utilities import working_dir
from .lofarhdf5 import h5_structure, h5_complex_voltage_file_names, h5_obs_header
from .lofarhdf5 import read_timeseries_subsampled
from .stationprocessing import fir_filter_coefficients, channelize_ppf




def data_loss_fraction(hdf5_name, dir_name,
                       detection_threshold=0.01, fractional_error_at_threshold=0.1):
    logging.info('data_loss_fraction(%r, %r)', dir_name, hdf5_name)
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
        with mp.Pool(processes=len(file_names[pol])) as pool:
            processes = [pool.apply_async(data_loss_fraction, (file_name, dir_name))
                         for file_name in sorted(file_names[pol])]
            losses.append(numpy.array([process.get() for process in processes]))
    losses = (numpy.array(losses).T).tolist()
    logging.debug(losses)
    logging.info('============================================================================')
    logging.info('                            DATA LOSS [%]')
    logging.info('============================================================================')
    logging.info('NAME                                 X_RE     X_IM     Y_RE     Y_IM     MAX')
    logging.info('----------------------------------------------------------------------------')
    for row in zip(file_names['x_re'], losses):
        logging.info(row[0]+'  %5.1f    %5.1f    %5.1f    %5.1f   %5.1f' %
                     (tuple(numpy.array(row[1])*100) + (numpy.array(row[1]).max()*100,)))
    logging.info('============================================================================')




def data_loss_report_plot(dir_name):
    file_names = h5_complex_voltage_file_names(dir_name)
    losses = []
    for pol in ['x_re', 'x_im', 'y_re', 'y_im']:
        with mp.Pool(processes=len(file_names[pol])) as pool:
            processes = [pool.apply_async(data_loss_fraction, (file_name, dir_name))
                         for file_name in sorted(file_names[pol])]
            losses.append(numpy.array([process.get() for process in processes]))
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





def find_sas_id(dir_name, sas_id=None):
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
    return sas_id





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
    sas_id = find_sas_id(dir_name, sas_id)
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
            logging.info(repr(time_s[-1]))
    xx = numpy.array(xx, dtype=numpy.complex64).transpose(1,2,0,3).copy(order='C')
    yy = numpy.array(yy, dtype=numpy.complex64).transpose(1,2,0,3).copy(order='C')
    time_s = numpy.array(time_s)
    return xx, yy, time_s, freq_axis, sas_id


def dynamic_spectrum_plot(dynamic_spectra_sb_time_chan, time_s, sb_freq_hz, num_taps=16, caption=''):
    num_sb, num_timeslots, num_chan = dynamic_spectra_sb_time_chan.shape
    ds = dynamic_spectra_sb_time_chan
    interval_s = time_s[1] - time_s[0]
    plt.figtext(0.5, 0.93, caption, horizontalalignment='center', fontsize=16)
    median_spectrum = numpy.median(ds[:,:,2*num_chan//num_taps:-2*num_chan//num_taps].mean(axis=1))
    for sb, freq_hz in enumerate(sb_freq_hz):
        ax_spectra = plt.subplot2grid((7, num_sb), (6, sb), colspan=1, rowspan=1)
        plt.plot(ds[sb,:,:].mean(axis=0), lw=2)
        if sb == 0:
            plt.ylim(0.5*median_spectrum, 1.5*median_spectrum)
            yticks_spectra = plt.yticks()[0]
            plt.ylabel('Power')
        else:
            plt.yticks(yticks_spectra, ['']*len(yticks_spectra))
            plt.ylim(0.5*median_spectrum, 1.5*median_spectrum)
        xticks_spectra = plt.xticks()[0]
        plt.xlim(-0.5, num_chan-0.5)
        plt.xlabel('Channel')
        
        ax_dynspec = plt.subplot2grid((7, num_sb), (0, sb), colspan=1, rowspan=6)
        plt.title('%.3f MHz' % (sb_freq_hz[sb]/1e6,))
        plt.imshow(ds[sb,:,:], vmin=0, vmax=1.5*median_spectrum,
           extent=(-0.5, num_chan-0.5, time_s[-1]+interval_s/2, -interval_s/2))
        plt.axis('tight')
        if sb == 0:
            plt.ylim(time_s[-1]+interval_s/2, -interval_s/2)
            yticks_dynspec = plt.yticks()[0]
            plt.ylabel('Time [s]')
        else:
            plt.yticks(yticks_dynspec, ['']*len(yticks_dynspec))
            plt.ylim(time_s[-1]+interval_s/2, -interval_s/2)
        plt.xticks(xticks_spectra, ['']*len(xticks_spectra))
        plt.xlim(-0.5, num_chan-0.5)
    plt.subplots_adjust(wspace=0.0)


def complex_voltage_obs_header(dir_name, sas_id=None):
    sas_id = find_sas_id(dir_name, sas_id)
    with working_dir(dir_name):
        sap_ids = range(488)
        sap_names = [[('%s_SAP%03d_B000_S%d_P000_bf.h5' % (sas_id, sap_id, pol))
                      for pol in [0, 1, 2, 3]]
                     for sap_id in sap_ids]
        first_h5_file = [h5py.File(file_names[0])
                         for file_names in sap_names
                         if os.path.exists(file_names[0])][0]
        return h5_obs_header(first_h5_file)




def write_inspection_pdf(input_dir_name, output_filename_template, sas_id=None):
    r'''
    example template: '%(sas_id)s-%(antenna_set)s-%(obs_datetime)s.pdf'
    '''
    logging.info('write_inspection_pdf(%r, %r, %r)',
                 input_dir_name, output_filename_template, sas_id)

    obs_header = complex_voltage_obs_header(input_dir_name, sas_id)
    info_dict = {}
    info_dict['sas_id'] = sas_id
    info_dict['antenna_set'] = obs_header['ANTENNA_SET'].decode('utf8')
    obs_datetime_compact = ''.join([ch for ch in obs_header['OBSERVATION_START_UTC'].decode('utf8')[0:19]
                                    if ch not in '-:T'])
    info_dict['obs_datetime'] = '%s_%s' % (obs_datetime_compact[0:8], obs_datetime_compact[8:], )

    logging.info('%r', info_dict)
    logging.info('Reading data')
    xx, yy, time_s, freq_axis, sas_id = subsampled_dynamic_spectra_by_timeslot(
                                                 input_dir_name, sas_id=sas_id)
    output_filename = output_filename_template % info_dict
    logging.info('Generating plots in file %s', output_filename)
    with PdfPages(output_filename) as pdf:
        # DATA LOSS
        plt.figure(figsize=(7,11))
        data_loss_report_plot(input_dir_name)
        pdf.savefig(papertype='a4', orientation='portrait')
        plt.close()

        # MEAN SPECTRUM
        plt.figure(figsize=(11,7), dpi=300)
        mean_dynspec = (abs(xx)**2).mean(axis=0) + (abs(yy)**2).mean(axis=0)
        dynamic_spectrum_plot(mean_dynspec,time_s,freq_axis['AXIS_VALUES_WORLD'],
                              caption='Incoherent mean of antennas')
        pdf.savefig(papertype='a4', orientation='landscape')
        plt.close()

        # MEDIAN SPECTRUM
        plt.figure(figsize=(11,7), dpi=300)
        mean_dynspec = numpy.median(abs(xx)**2, axis=0) + numpy.median(abs(yy)**2, axis=0)
        dynamic_spectrum_plot(mean_dynspec,time_s,freq_axis['AXIS_VALUES_WORLD'],
                              caption='Incoherent median of antennas')
        pdf.savefig(papertype='a4', orientation='landscape')
        plt.close()

        # ANTENNA SPECTRA
        for antenna in range(xx.shape[0]):
            plt.figure(figsize=(11,7), dpi=300)
            antenna_dynspec = (abs(xx[antenna,:,:,:])**2) + (abs(yy[antenna,:,:])**2)
            dynamic_spectrum_plot(antenna_dynspec,time_s,freq_axis['AXIS_VALUES_WORLD'],
                                  caption='Antenna %d' % antenna)
            pdf.savefig(papertype='a4', orientation='landscape')
            plt.close()
        return output_filename
