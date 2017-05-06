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
