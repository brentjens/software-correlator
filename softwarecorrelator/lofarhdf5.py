import h5py
import numpy.ma as ma
import numpy
import os, copy, glob
from scipy.ndimage.filters import median_filter, gaussian_filter, minimum_filter

from .utilities import working_dir


def h5_complex_voltage_file_names(dir_name):
    '''
    Find the filenames containing real and imaginary parts of X and Y
    complex voltages for all HDF5 files in `dir_name`

    **Parameters**

    dir_name : string
               The directory to probe.

    **Returns**

    A dictionary with keys `x_re`, `x_im`, `y_re`, and `y_im`
    containing lists of file names per kind.

    **Example**
    '''
    with working_dir(dir_name):
        hdf5_file_names = glob.glob('*.h5')
        return {'x_re': sorted([n for n in hdf5_file_names if '_S0_' in n]),
                'x_im': sorted([n for n in hdf5_file_names if '_S1_' in n]),
                'y_re': sorted([n for n in hdf5_file_names if '_S2_' in n]),
                'y_im': sorted([n for n in hdf5_file_names if '_S3_' in n])}


def h5_first_beam(h5file):
    def collect(x):
        if 'STOKES_0' in x:
            return x
    return h5file.visit(collect)

def h5_station_list(h5file):
    return h5file[h5_first_beam(h5file).split('STOKES')[0]].attrs['STATIONS_LIST']

def h5_dynspec(h5file):
    return h5file[h5_first_beam(h5file)][:,:]

def h5_dynspec_avg(h5file, time_avg_factor, freq_avg_factor):
    data = h5file[h5_first_beam(h5file)][:,:]
    data = data[0:data.shape[0]-data.shape[0]%time_avg_factor, 0:data.shape[1]-data.shape[1]%freq_avg_factor]
    time_averaged = data.reshape((-1, time_avg_factor, data.shape[-1])).mean(axis=1)
    freq_averaged = time_averaged.reshape((time_averaged.shape[0], -1, freq_avg_factor)).mean(axis=-1)
    return freq_averaged

    # output = 0*data[0::time_avg_factor, 0::freq_avg_factor]
    # weights = numpy.zeros(output.shape)
    # for dt in range(time_avg_factor):
    #     for df in range(freq_avg_factor):
    #         print dt, df
    #         sub_grid = data[dt::time_avg_factor, df::freq_avg_factor]
    #         sub_nt, sub_nf = sub_grid.shape
    #         output[0:sub_nt, 0:sub_nf] += sub_grid
    #         weights[0:sub_nt, 0:sub_nf] += 1
    # return output/weights
        

def h5_print_structure(h5file):
    def print_line(x):
        print(x)
    h5file.visit(print_line)

def h5_structure(h5file):
    fields = []
    def accumulate_fields(x):
        fields.append(x)
    h5file.visit(accumulate_fields)
    return fields

    
def h5_beam_header(h5file):
    return dict(h5file[h5_first_beam(h5file).split('STOKES')[0]].attrs)


def h5_format_beam_header(h5file):
    return '\n'.join(sorted(['%27s: %r' % (key, value)
                             for (key, value)
                             in h5_beam_header.items()],
                            key=lambda x: x.split(':')[0].strip()))



def read_timeseries_subsampled_by_sap(dir_name, sas_id_string, sap_ids, interval_s=0.1,
                               interval_samples=None, num_samples=256*16):
    sap_names = [[('%s_SAP%03d_B000_S%d_P000_bf.h5' % (sas_id_string, sap_id, pol))
                  for pol in [0, 1, 2, 3]]
                 for sap_id in sap_ids]
    sap_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/STOKES_%d'
    coordinate_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/COORDINATES/COORDINATE_%d'
    with working_dir(dir_name):
        h5_files_by_sap = [[h5py.File(file_name) for file_name in names]
                           for names in sap_names]
        time_axis, freq_axis = [
            dict([i for i in h5_files_by_sap[0][0][coordinate_fmt % (sap_ids[0], axis_id)].attrs.items()])
                    for axis_id in [0, 1]]
        sample_duration_s = time_axis['INCREMENT']
        if interval_samples is None:
            samples_per_interval = int(numpy.floor(interval_s/sample_duration_s))
        else:
            samples_per_interval = interval_samples

        for sap_id, sap_files in zip(sap_ids, h5_files_by_sap):
            timeslots_per_file = sap_files[0][sap_fmt % (sap_id, 0)].shape[0]
            first_timeslot = 0
            time_series_complex_x = []
            time_series_complex_y = []
            time_axis['INCREMENT'] = samples_per_interval*sample_duration_s
            h5_groups = [h5_file[sap_fmt % (sap_id, pol)]
                          for pol, h5_file in enumerate(sap_files)]

            num_sb = len(freq_axis['AXIS_VALUES_WORLD'])
            num_timeslots = (timeslots_per_file-num_samples) // samples_per_interval
            time_series_real = numpy.zeros((4, num_samples, num_sb), dtype=numpy.float32)

            time_series_complex_x = numpy.zeros((num_timeslots, num_samples, num_sb), dtype=numpy.complex64)
            time_series_complex_y = numpy.zeros((num_timeslots, num_samples, num_sb), dtype=numpy.complex64)

            timeslot = 0
            while first_timeslot < timeslots_per_file - samples_per_interval - num_samples:
                [h5_groups[pol].read_direct(time_series_real,
                                            numpy.s_[first_timeslot:first_timeslot+num_samples,:],
                                            numpy.s_[pol,:, :])
                                          for pol in range(4)] #, h5_file in enumerate(sap_files)],
                time_series_complex_x[timeslot,:,:] += (time_series_real[0,:,:] + 1j*time_series_real[1,:,:])
                time_series_complex_y[timeslot,:,:] += (time_series_real[2,:,:] + 1j*time_series_real[3,:,:])
                first_timeslot += samples_per_interval
                timeslot += 1
            yield (time_series_complex_x,
                   time_series_complex_y,
                   time_axis, freq_axis)


def read_timeseries_subsampled(dir_name, sas_id_string, sap_ids, interval_s=0.1,
                               interval_samples=None, num_samples=256*16):
    sap_names = [[('%s_SAP%03d_B000_S%d_P000_bf.h5' % (sas_id_string, sap_id, pol))
                  for pol in [0, 1, 2, 3]]
                 for sap_id in sap_ids]
    sap_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/STOKES_%d'
    coordinate_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/COORDINATES/COORDINATE_%d'
    with working_dir(dir_name):
        h5_files_by_sap = [[h5py.File(file_name) for file_name in names]
                           for names in sap_names]
        time_axis, freq_axis = [
            dict([i for i in h5_files_by_sap[0][0][coordinate_fmt % (sap_ids[0], axis_id)].attrs.items()])
                    for axis_id in [0, 1]]
        sample_duration_s = time_axis['INCREMENT']
        if interval_samples is None:
            samples_per_interval = int(numpy.floor(interval_s/sample_duration_s))
        else:
            samples_per_interval = interval_samples
        timeslots_per_file = h5_files_by_sap[0][0][sap_fmt % (0, 0)].shape[0]
        first_timeslot = 0

        # Pre-find groups to save on h5py Group getitem calls, which previously
        # cost 20--25% of total runtime
        h5_groups = [[h5_file[sap_fmt % (sap_id, pol)]
                      for pol, h5_file in enumerate(h5_files_by_sap[sap_id])]
                     for sap_ix, sap_id in enumerate(sap_ids)]
        num_sb = len(freq_axis['AXIS_VALUES_WORLD'])
        num_timeslots = (timeslots_per_file-num_samples) // samples_per_interval
        time_series_real = numpy.zeros((len(sap_ids), 4, num_samples, num_sb), dtype=numpy.float32)
        timeslot = 0
        while first_timeslot < timeslots_per_file - samples_per_interval - num_samples:
            time_axis['REFERENCE_VALUE'] = (first_timeslot+num_samples/2)*sample_duration_s
            [[h5_groups[sap_id][pol].read_direct(time_series_real,
                                                  numpy.s_[first_timeslot:first_timeslot+num_samples,:],
                                                 numpy.s_[sap_ix, pol, :, :])
                                      for pol in range(4)]
                                      for sap_ix, sap_id in enumerate(sap_ids)]
            time_series_complex_x = time_series_real[:,0,:,:] + 1j*time_series_real[:,1,:,:]
            time_series_complex_y = time_series_real[:,2,:,:] + 1j*time_series_real[:,3,:,:]
            first_timeslot += samples_per_interval
            timeslot += 1
            yield (time_series_complex_x, time_series_complex_y, time_axis, freq_axis)



def read_timeseries(dir_name, sas_id_string, sap_id):
    names = [('%s_SAP%03d_B000_S%d_P000_bf.h5' % (sas_id_string, sap_id, pol))
             for pol in [0, 1, 2, 3]]
    sap_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/STOKES_%d'
    coordinate_fmt = 'SUB_ARRAY_POINTING_%03d/BEAM_000/COORDINATES/COORDINATE_%d'
    with working_dir(dir_name):
        h5_files = [h5py.File(file_name) for file_name in names]
        time_series = numpy.array([h5_file[sap_fmt % (sap_id, pol)][:,:]
                                   for pol, h5_file in enumerate(h5_files)],
                                  dtype=numpy.float32)
        time_axis, freq_axis = [
            dict([i for i in h5_files[0][coordinate_fmt % (sap_id, axis_id)].attrs.items()])
            for axis_id in [0, 1]]
    return (time_series[0,:,:] + 1j*time_series[1,:,:],
            time_series[2,:,:] + 1j*time_series[3,:,:],
            time_axis, freq_axis)



class MedianFilter(object):
    def __init__(self, window_width):
        self.window_width = window_width
            
    def __call__(self, x):
        return median_filter(x, self.window_width)

        
def par_median_spectrum(dynamic_spectrum_tf, freq_window_channels=37):
    from IPython.parallel import Client
    cl = Client()
    dv = cl[:]
    with dv.sync_imports():
        from scipy.ndimage.filters import median_filter, gaussian_filter, minimum_filter
    dv['MedianFilter'] = MedianFilter
        
    mf = MedianFilter(freq_window_channels)
    median_tf_plane = numpy.array(dv.map_sync(mf, dynamic_spectrum_tf))#median_filter(data, size=(1, 51))
    median_spectrum = numpy.array(dv.map_sync(numpy.median, median_tf_plane.T))
    return median_spectrum


def flag_data(dynamic_spectrum, channels_per_subband=16):
    data = dynamic_spectrum
    data_mean = data.mean()
    data_std = data.std()
    flags= numpy.logical_or(abs(data - data_mean) > 8*data_std,
                       data == 0)
    flags[:, 0::channels_per_subband] = True
    flagged_data = ma.array(data, mask=flags, copy=True)

    data_mean = ma.mean(flagged_data)
    data_std = ma.std(flagged_data)
    flags = numpy.logical_or(abs(data - data_mean) > 6*data_std,
                       data == 0)
    print(type(flags))
    flags[:, 0::channels_per_subband] = True
    flagged_data = ma.array(data, mask=flags, copy=True)

    data_mean = ma.mean(flagged_data)
    data_std = ma.std(flagged_data)
    flags = numpy.logical_or(abs(data - data_mean) > 4*data_std,
                       data == 0)
    flags[:, 0::channels_per_subband] = True
    flagged_data = ma.array(data, mask=flags, copy=True)

    data_mean = ma.mean(flagged_data)
    data_std = ma.std(flagged_data)
    flags = numpy.logical_or(abs(data - data_mean) > 4*data_std,
                       data == 0)
    flags[:, 0::channels_per_subband] = True
    flagged_data = ma.array(data, mask=flags, copy=True)

    data_mean = ma.mean(flagged_data)
    data_std = ma.std(flagged_data)
    flags = numpy.logical_or(abs(data - data_mean) > 4*data_std,
                       data == 0)
    flags[:, 0::channels_per_subband] = True
    flagged_data = ma.array(data, mask=flags, copy=True)
    
    return flagged_data

def fold_spectrum(spectrum, fold_channels):
    return spectrum.reshape((-1, fold_channels)).mean(axis=0)

def get_folded_spectrum(dir_name, num_channels, file_name):
    os.chdir(dir_name)
    h5file = h5py.File(file_name, mode='r')
    raw_data = h5_dynspec(h5file)
    raw_data /= raw_data[:, num_channels/2::num_channels].mean()
    flagged_data = flag_data(raw_data, num_channels)
    full_spectrum = ma.median(flagged_data, axis=0)
    return fold_spectrum(full_spectrum, num_channels)
