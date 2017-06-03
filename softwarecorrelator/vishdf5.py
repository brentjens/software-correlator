import h5py
import numpy.ma as ma
import os


class VisHDF5(h5py.File):
    r'''
    Interface to visibilities stored in HDF5 format.
    '''

    def __init__(self, file_name, mode='r'):
        pre_existed = False
        if os.path.exists(file_name):
            pre_existed = True
        super().__init__(file_name, mode=mode)
        if pre_existed:
            self.num_ant = self['ANTENNA/NAME'].shape[0]
            self.num_bl = (self.num_ant*(self.num_ant+1))//2
        else:
            self.num_ant = 0
            self.num_bl = 0


    def close(self):
        super().close()


    def create_empty(self, num_timeslots, num_ant, num_chan, num_sb=1, num_pol=4):
        num_bl = (num_ant*(num_ant+1))//2
        num_rows = num_timeslots*num_bl*num_sb
        self.num_ant = num_ant
        self.num_bl = num_bl
        h5 = self
        h5.create_dataset('MAIN/DATA', (num_rows, num_chan, num_pol), dtype='c8',
                          fillvalue=0.0+0.0j)
        h5.create_dataset('MAIN/FLAG', (num_rows, num_chan, num_pol), dtype='b',
                          fillvalue=False)
        h5.create_dataset('MAIN/ANTENNA1', (num_rows,), dtype='i4',
                          fillvalue=-1)
        h5.create_dataset('MAIN/ANTENNA2', (num_rows,), dtype='i4',
                          fillvalue=-1)
        h5.create_dataset('MAIN/DATA_DESC_ID', (num_rows,), dtype='i4',
                          fillvalue=-1)
        h5.create_dataset('MAIN/TIME', (num_rows,), dtype='f8',
                          fillvalue=0.0)
        h5.create_dataset('MAIN/TIMESLOT', (num_rows,), dtype='i8',
                          fillvalue=-1)

        h5.create_dataset('SPECTRAL_WINDOW/NUM_CHAN', (num_sb, 1), dtype='i8',
                          fillvalue=0)
        h5.create_dataset('SPECTRAL_WINDOW/CHAN_FREQ', (num_sb, num_chan), dtype='f8',
                          fillvalue=0.0)

        h5.create_dataset('ANTENNA/NAME', (num_ant, 1), dtype='S512',
                          fillvalue=b'')
        h5.create_dataset('ANTENNA/ITRF', (num_ant, 3), dtype='f8',
                          fillvalue=0.0)


    def fill_spectral_window(self, chan_freq_per_spw):
        self['SPECTRAL_WINDOW/NUM_CHAN'][:] = chan_freq_per_spw.shape[1]
        self['SPECTRAL_WINDOW/CHAN_FREQ'] = chan_freq_per_spw


    def fill_antenna(self, antenna_names, antenna_itrf):
        self['ANTENNA/NAMES'] = antenna_names
        self['ANTENNA/ITRF'] = antenna_itrf


    def fill_main_header(self, main_header_dict):
        self.attrs = main_header_dict
        self.attrs['VISHDF5_VERSION'] = '1.0'


    def baseline_offset(self, antenna1, antenna2):
        h5 = self
        offset = arange(num_bl)[
            numpy.logical_and(h5['MAIN/ANTENNA1'][:self.num_bl] == antenna1,
                              h5['MAIN/ANTENNA2'][:self.num_bl] == antenna2)]
        if len(offset) == 1:
            return offset[0]
        else:
            raise KeyError('VisHDF5.baseline_offset(): Baseline %d--%d was found %d times; expected 1' %
                           (antenna1, antenna2, len(offset)))

        
    def get_baseline_dynspec(self, antenna1, antenna2):
        offset = baseline_offset(antenna1, antenna2)
        data = self['MAIN/DATA'][offset::self.num_bl, ...]
        flag = self['MAIN/FLAG'][offset::self.num_bl, ...]
        return ma.array(data, mask=flag)


    
    def set_baseline_flags(self, antenna1, antenna2, flags):
        offset = baseline_offset(antenna1, antenna2)
        self['MAIN/FLAG'][offset::self.num_bl, ...] = flags

