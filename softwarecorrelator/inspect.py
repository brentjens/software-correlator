import numpy

from .utilities import working_dir
from .lofarhdf5 import h5_structure, h5_complex_voltage_file_names
import h5py



def data_loss_fraction(dir_name, hdf5_name,
                       detection_threshold=0.01, fractional_error_at_threshold=0.1):
    with working_dir(dir_name):
        h5 = h5py.File(hdf5_name)
        h5_data_path = [n for n in h5_structure(h5) if 'STOKES' in n][0]
        num_samples = int(1/(detection_threshold*fractional_error_at_threshold**2))
        sub_sampling = int(numpy.product(h5[h5_data_path].shape)//num_samples)
        subset = h5[h5_data_path][::sub_sampling,:]
        return numpy.sum(subset == 0.0)/numpy.product(subset.shape)    


def data_loss_report(dir_name):
    file_names = h5_complex_voltage_file_names(dir_name)
    losses = []
    for pol in ['x_re', 'x_im', 'y_re', 'y_im']:
        losses.append(numpy.array([data_loss_fraction(dir_name, file_name)
                                   for file_name in file_names['x_re']]))
    print('====================================================================')
    print('                            DATA LOSS [%]')
    print('====================================================================')
    print('NAME                                 X_RE     X_IM     Y_RE     Y_IM')
    print('--------------------------------------------------------------------')
    for row in zip(file_names['x_re'], *losses):
        print(row[0], '  %5.1f    %5.1f    %5.1f    %5.1f' % tuple(numpy.array(row[1:])*100))
    print('====================================================================')
