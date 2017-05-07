import numpy
import h5py
import logging
import matplotlib.pyplot as plt

from .utilities import working_dir
from .lofarhdf5 import h5_structure, h5_complex_voltage_file_names




def data_loss_fraction(dir_name, hdf5_name,
                       detection_threshold=0.01, fractional_error_at_threshold=0.1):
    with working_dir(dir_name):
        h5 = h5py.File(hdf5_name)
        h5_data_path = [n for n in h5_structure(h5) if 'STOKES' in n][0]
        num_samples = int(1/(detection_threshold*fractional_error_at_threshold**2))
        sub_sampling = int(numpy.product(h5[h5_data_path].shape)//num_samples)
        subset = h5[h5_data_path][::sub_sampling,:]
        return numpy.sum(subset == 0.0)/numpy.product(subset.shape)    


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
