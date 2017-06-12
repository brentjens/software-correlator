import h5py
import numpy
import numpy.ma as ma
import os
import logging

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
        h5.create_dataset('MAIN/FLAGROW', (num_rows,), dtype='b',
                          fillvalue=True)
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

        h5.create_dataset('ANTENNA/NAME', (num_ant,), dtype='S512',
                          fillvalue=b'')
        h5.create_dataset('ANTENNA/ITRF', (num_ant, 3), dtype='f8',
                          fillvalue=0.0)




    def add_weight_col(self):
        try:
            _ = self['MAIN/VISWEIGHT'][0]
            logging.warning('MAIN/VISWEIGHT already exists in %s',
                            self.filename)
        except:
            logging.info('Creating MAIN/VISWEIGHT in %s',
                         self.filename)
            self.create_dataset('MAIN/VISWEIGHT', self['MAIN/DATA'].shape, dtype='f8',
                                fillvalue=0.0)


    def fill_spectral_window(self, chan_freq_per_spw):
        self['SPECTRAL_WINDOW/NUM_CHAN'][:] = chan_freq_per_spw.shape[1]
        self['SPECTRAL_WINDOW/CHAN_FREQ'][:] = chan_freq_per_spw


    def fill_antenna(self, antenna_names, antenna_itrf):
        self['ANTENNA/NAME'][:] = antenna_names
        self['ANTENNA/ITRF'][:] = antenna_itrf


    def fill_main_header(self, main_header_dict):
        [self.attrs.create(key, value) for key, value in main_header_dict.items()]
        self.attrs.create('VISHDF5_VERSION', b'1.0')

    def baseline_offset(self, antenna1, antenna2):
        h5 = self
        offset = numpy.arange(self.num_bl)[
            numpy.logical_and(h5['MAIN/ANTENNA1'][:self.num_bl] == antenna1,
                              h5['MAIN/ANTENNA2'][:self.num_bl] == antenna2)]
        if len(offset) == 1:
            return offset[0]
        else:
            raise KeyError('VisHDF5.baseline_offset(): Baseline %d--%d was found %d times; expected 1' %
                           (antenna1, antenna2, len(offset)))

        
    def get_baseline_dynspec(self, antenna1, antenna2):
        offset = self.baseline_offset(antenna1, antenna2)
        data = self['MAIN/DATA'][offset::self.num_bl, ...]
        flag = self['MAIN/FLAG'][offset::self.num_bl, ...]
        return ma.array(data, mask=flag)


    
    def set_baseline_flags(self, antenna1, antenna2, flags):
        offset = self.baseline_offset(antenna1, antenna2)
        self['MAIN/FLAG'][offset::self.num_bl, ...] = flags



    def bytes_per_timeslot_data_flags(self):
        bytes_per_vis = 9
        num_vis = self.num_bl*self['MAIN/DATA'].shape[1]*self['MAIN/DATA'].shape[2]
        return num_vis*bytes_per_vis


    def bytes_per_baseline_data_flags(self):
        bytes_per_vis = 9
        num_vis = numpy.product(self['MAIN/DATA'].shape)//self.num_bl
        return num_vis*bytes_per_vis


    def get_acm_blocks(self, first_timeslot, max_num_timeslots):
        bl = self.num_bl
        first = first_timeslot
        n = max_num_timeslots
        num_ch = self['MAIN/DATA'][first*bl:(first+n)*bl,...].shape[1]
        num_pol = self['MAIN/DATA'][first*bl:(first+n)*bl,...].shape[2]
        return {'DATA': ma.array(self['MAIN/DATA'][first*bl:(first+n)*bl,...].reshape((-1, bl, num_ch, num_pol)),
                                 mask = self['MAIN/FLAG'][first*bl:(first+n)*bl,...].reshape((-1, bl, num_ch, num_pol))),
                'TIME': self['MAIN/TIME'][first*bl:(first+n)*bl:bl],
                'ANTENNA1': self['MAIN/ANTENNA1'][first*bl:(first+1)*bl],
                'ANTENNA2': self['MAIN/ANTENNA2'][first*bl:(first+1)*bl],
                'DATA_DESC_ID': self['MAIN/DATA_DESC_ID'][first*bl:(first+n)*bl:bl],
                }


    def get_baseline_blocks(self, first_baseline, max_num_baselines, buffer=None):
        bl = self.num_bl
        first = first_baseline
        num_baselines = min(max_num_baselines, bl-first_baseline)
        n_timeslots = self['MAIN/DATA'].shape[0]//self.num_bl
        num_ch = self['MAIN/DATA'].shape[-2]
        num_pol = self['MAIN/DATA'].shape[-1]
        if num_baselines <= 0:
            return buffer, 0
#            raise ValueError('No more baselines to read')
        if buffer is None:
            data = numpy.zeros((n_timeslots, num_baselines, num_ch, num_pol),
                               dtype=numpy.complex64)
            mask = numpy.zeros(data.shape, dtype=numpy.bool)
        else:
            data = buffer.data
            mask = buffer.mask
        data_group = self['MAIN/DATA']
        for ts in range(n_timeslots):
            data_group.read_direct(
                data,
                numpy.s_[ts*bl+first:ts*bl+first+num_baselines,:,:],
                numpy.s_[ts,0:num_baselines,:,:])
        flag_group = self['MAIN/FLAG']
        for ts in range(n_timeslots):
            flag_group.read_direct(
                mask,
                numpy.s_[ts*bl+first:ts*bl+first+num_baselines,:,:],
                numpy.s_[ts,0:num_baselines,:,:])
        return ma.array(data, mask=mask), num_baselines


    def set_baseline_block_col(self, x, column, first_baseline):
        r'''
        column = 'DATA' or 'FLAG'.
        '''
        bl = self.num_bl
        first = first_baseline
        num_baselines = x.shape[1]
        n_timeslots = self['MAIN/DATA'].shape[0]//self.num_bl
        num_ch = self['MAIN/DATA'].shape[-2]
        num_pol = self['MAIN/DATA'].shape[-1]
        for ts in range(n_timeslots):
            self['MAIN/'+column][ts*bl+first:ts*bl+first+num_baselines,:,:] = x[ts,:,:,:]
