from .vishdf5 import VisHDF5
from .sumthreshold import sum_threshold_2d
import gc
import numpy
import logging

# Steps:
# -> read ACM blocks until mem full
# -> flag baselines (& write flags if needed)
# -> avg sub_block in freq (& calc weights)
# -> avg sub_block in time (& re-calc weights)
# -> write sub_block + flags


# import gc
# bls = None
# gc.collect(True)
# !date
# bl_per_chunk = 6*1024**3//vish5s[0].bytes_per_baseline_data_flags()
# i=0

# while True:
#     bls, num_read = vish5s[0].get_baseline_blocks(i*bl_per_chunk, bl_per_chunk, buffer=bls)
#     i += 1
#     print(bls.shape, num_read)
#     if num_read < bl_per_chunk:
#         break
# !date



def flag_and_average(input_filename, output_filename,
                     flagging_threshold=None,
                     output_channel_mapping=None,
                     time_avg_factor=None,
                     output_mode='w-',
                     max_mem_GB=8):
    r'''
    output_channel_mapping = [(first, last+1),
                              (first, last+1),
                              ...]
    '''
    input_mode = 'r'
    if input_filename == output_filename \
       and flagging_threshold is not None \
       and output_channel_mapping is None \
       and time_avg_factor is None:
        input_mode = 'w-'
    only_flagging = False
    input_h5 = VisHDF5(input_filename, mode=input_mode)
    if output_filename != input_filename:
        output_h5 = VisHDF5(output_filename, mode=output_mode)
        # Copy metadata
        num_chan_out = len(output_channel_mapping)
        num_bl_out = input_h5.num_bl
        if time_avg_factor is None:
            num_timeslot_out = input_h5['MAIN/TIMESLOT'].shape[0]//num_bl_out
        else:
            num_timeslot_out = input_h5['MAIN/TIMESLOT'].shape[0]//num_bl_out//time_avg_factor
            
        output_h5.create_empty(num_timeslot_out,
                               num_ant=input_h5['ANTENNA/NAME'].shape[0],
                               num_chan=num_chan_out,
                               num_sb=input_h5['SPECTRAL_WINDOW/NUM_CHAN'].shape[0],
                               num_pol=input_h5['MAIN/DATA'].shape[-1])
        output_h5.fill_main_header(input_h5.attrs)
        output_h5.fill_antenna(input_h5['ANTENNA/NAME'][:],
                               input_h5['ANTENNA/ITRF'][:])
        if output_channel_mapping is None:
            output_h5.fill_spectral_window(input_h5['SPECTRAL_WINDOW/CHAN_FREQ'][:])
        else:
            output_chan_freq = [chan_freq[first:last_plus_one]
                                for (first, last_plus_one), chan_freq
                                in zip (output_channel_mapping,
                                        input_h5['SPECTRAL_WINDOW/CHAN_FREQ'][:])]
            output_h5.fill_spectral_window(numpy.array(output_chan_freq))
    else:
        output_h5 = input_h5
        only_flagging = True
    num_pol = input_h5['MAIN/DATA'].shape[-1]
    num_timeslots_in = input_h5['MAIN/TIMESLOT'].shape[0]//input_h5.num_bl
    flag_window_lengths =  2**numpy.arange(
        int(numpy.ceil(numpy.log(num_timeslots_in)/numpy.log(2))))
    logging.debug('flag_window_lengths: %r', flag_window_lengths)
    
    buffer = None
    gc.collect(True)
    bytes_per_baseline = input_h5.bytes_per_baseline_data_flags()
    if output_filename != input_filename:
        bytes_per_baseline += output_h5.bytes_per_baseline_data_flags()
    
    baselines_per_chunk = max_mem_GB*1024**3//bytes_per_baseline
    block_index = 0
    while True:
        buffer, num_read = input_h5.get_baseline_blocks(
            block_index*baselines_per_chunk,
            baselines_per_chunk, buffer=buffer)
        if flagging_threshold is not None:
            for bl in range(num_read):
                for pol in range(num_pol):
                    frame_data = numpy.abs(buffer.data[:, bl, :, pol])
                    frame_data -= frame_data.mean()
                    data_std = frame_data.std()
                    if data_std != 0.0:
                        frame_data /= frame_data.std()
                    flags = sum_threshold_2d(frame_data,
                                             buffer.mask[:, bl, :, pol],
                                             flagging_threshold,
                                             window_lengths=flag_window_lengths)
                    buffer.mask[:, bl, :, pol] = flags
                buffer.mask[:, bl, :, :] = buffer.mask[:, bl, :, :].max(axis=-1)[:,:,numpy.newaxis]
            
                
        block_index += 1
        if num_read < baselines_per_chunk:
            break




def flag_dataset(input_filename,
                 flagging_threshold=4.0,
                 propagate_flags=['pol'],
                 max_mem_GB=8.0,
                 unflag_channels=[],
                 close_gaps=True):
    input_mode = 'r+'
    input_h5 = VisHDF5(input_filename, mode=input_mode)
    num_pol = input_h5['MAIN/DATA'].shape[-1]
    num_timeslots_in = input_h5['MAIN/TIMESLOT'].shape[0]//input_h5.num_bl
    flag_window_lengths =  2**numpy.arange(
        int(numpy.ceil(numpy.log(num_timeslots_in)/numpy.log(2))))
    logging.debug('flag_window_lengths: %r', flag_window_lengths)
    
    buffer = None
    gc.collect(True)
    bytes_per_baseline = input_h5.bytes_per_baseline_data_flags()
    
    baselines_per_chunk = int(max_mem_GB*1024**3)//bytes_per_baseline
    block_index = 0
    while True:
        buffer, num_read = input_h5.get_baseline_blocks(
            block_index*baselines_per_chunk,
            baselines_per_chunk, buffer=buffer)
        for bl in range(num_read):
            for pol in range(num_pol):
                frame_data = numpy.abs(buffer.data[:, bl, :, pol])
                frame_data -= frame_data.mean()
                data_std = frame_data.std()
                if data_std != 0.0:
                    frame_data /= frame_data.std()
                flags = sum_threshold_2d(frame_data,
                                         numpy.copy(buffer.mask[:, bl, :, pol]),
                                         flagging_threshold,
                                         window_lengths=flag_window_lengths)
                buffer.mask[:, bl, :, pol] = flags
            if 'pol' in propagate_flags:
                buffer.mask[:, bl, :, :] = buffer.mask[:, bl, :, :].max(axis=-1)[:,:,numpy.newaxis]
        for channel in unflag_channels:
            buffer.mask[:,:,channel,:] = False
        if close_gaps:
            closed_gaps = buffer.mask.copy()
            closed_gaps[:,:,1:-1,:] = numpy.logical_or(
                closed_gaps[:,:,1:-1,:],
                numpy.logical_and(buffer.mask[:,:,2:,:],
                                  buffer.mask[:,:,0:-2,:]))
            buffer.mask = closed_gaps
            closed_gaps = None
            gc.collect()
        
        input_h5.set_baseline_block_col(
            buffer.mask[:,:num_read,:,:],
            column='FLAG',
            first_baseline=block_index*baselines_per_chunk)
        
        block_index += 1
        if num_read < baselines_per_chunk:
            break
    input_h5.close()




def clear_flags(input_filename):
    h5 = VisHDF5(input_filename, mode='r+')
    h5['MAIN/FLAG'][...] *= False
    h5.close()


