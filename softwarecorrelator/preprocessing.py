from .vishdf5 import VisHDF5
from .sumthreshold import sum_threshold_2d
import gc
import numpy
import numpy.ma as ma
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
                 flagging_threshold=5.0,
                 propagate_flags=['pol'],
                 max_mem_GB=8.0,
                 unflag_channels=[],
                 close_gaps=True,
                 threshold_shrink_power=0.45):
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
                                         window_lengths=flag_window_lengths,
                                         threshold_shrink_power=threshold_shrink_power)
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






def average_dataset(input_filename, output_filename,
                    apply_flags=True,
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
    if input_filename == output_filename:
        raise RuntimeError('Can only average into other data set')
    input_h5 = VisHDF5(input_filename, mode=input_mode)
    output_h5 = VisHDF5(output_filename, mode=output_mode)

    # Copy metadata
    num_chan_out = len(output_channel_mapping)
    num_bl_out = input_h5.num_bl
    if time_avg_factor is None:
        num_timeslot_out = input_h5['MAIN/TIMESLOT'].shape[0]//num_bl_out
        time_avg_factor = 1
    else:
        num_timeslot_out = input_h5['MAIN/TIMESLOT'].shape[0]//num_bl_out//time_avg_factor

    output_h5.create_empty(num_timeslot_out,
                           num_ant=input_h5['ANTENNA/NAME'].shape[0],
                           num_chan=num_chan_out,
                           num_sb=input_h5['SPECTRAL_WINDOW/NUM_CHAN'].shape[0],
                           num_pol=input_h5['MAIN/DATA'].shape[-1])
    output_h5.add_weight_col()
    output_h5.fill_main_header(input_h5.attrs)
    output_h5.fill_antenna(input_h5['ANTENNA/NAME'][:],
                           input_h5['ANTENNA/ITRF'][:])
    if output_channel_mapping is None:
        output_h5.fill_spectral_window(input_h5['SPECTRAL_WINDOW/CHAN_FREQ'][:])
    else:
        output_chan_freq = [[chan_freq[first:last_plus_one].mean()
                             for (first, last_plus_one) in output_channel_mapping]
                            for chan_freq in input_h5['SPECTRAL_WINDOW/CHAN_FREQ'][:]]
        output_h5.fill_spectral_window(numpy.array(output_chan_freq))
    num_pol = input_h5['MAIN/DATA'].shape[-1]
    num_timeslots_in = input_h5['MAIN/TIMESLOT'].shape[0]//input_h5.num_bl

    buffer = None
    gc.collect(True)
    # go through data set by timeslot
    bytes_per_ts_block = input_h5.bytes_per_timeslot_data_flags()*time_avg_factor \
                         + output_h5.bytes_per_timeslot_data_flags()

    ts_blocks_per_chunk = max_mem_GB*1024**3//bytes_per_ts_block
    block_index = 0
    output_timeslot = 0
    while True:
        c = input_h5.get_acm_blocks(
            block_index*ts_blocks_per_chunk*time_avg_factor,
            ts_blocks_per_chunk*time_avg_factor)
        mean_fn = ma.mean
        if not apply_flags:
            mean_fn = numpy.mean
        ts_in = time_avg_factor*(c['DATA'].shape[0]//time_avg_factor)
        output_data_ch_tm_bl_pl = [
            mean_fn(
                mean_fn(c['DATA'][:ts_in,:,first:last_plus_one,:], axis=-2).reshape((-1, time_avg_factor, num_bl_out, num_pol)),
                axis=1)
            for (first, last_plus_one) in output_channel_mapping]
        output_data_ch_tm_bl_pl = ma.array([channel.data for channel in output_data_ch_tm_bl_pl],
                                           mask=[channel.mask for channel in output_data_ch_tm_bl_pl])
        output_data_proper_order =  output_data_ch_tm_bl_pl.transpose((1, 2, 0, 3))
        output_time = c['TIME'][:ts_in].reshape((-1, time_avg_factor)).mean(axis=1)
        if apply_flags:
            output_visweight = numpy.array([
            numpy.mean(
                numpy.mean(1-c['DATA'].mask[:ts_in,:,first:last_plus_one,:], axis=-2).reshape((-1, time_avg_factor, num_bl_out, num_pol)),
                axis=1)
            for (first, last_plus_one) in output_channel_mapping]).transpose((1,2,0,3))
        else:
            output_visweight = numpy.ones(output_data_proper_order.shape)

        for index, acm in enumerate(output_data_proper_order):
            #Write output stuff
            first_row = output_timeslot*num_bl_out
            last_row = (output_timeslot+1)*num_bl_out
            if output_timeslot % 100 == 0:
                print(first_row, last_row, acm.shape, output_timeslot)
            output_h5['MAIN/DATA'][first_row:last_row, :, :] = acm
            output_h5['MAIN/VISWEIGHT'][first_row:last_row, :, :] = output_visweight[index,:,:,:]
            output_h5['MAIN/ANTENNA1'][first_row:last_row] = c['ANTENNA1']
            output_h5['MAIN/ANTENNA2'][first_row:last_row] = c['ANTENNA2']
            output_h5['MAIN/TIMESLOT'][first_row:last_row] = output_timeslot
            output_h5['MAIN/TIME'][first_row:last_row] = output_time[index]
            output_h5['MAIN/DATA_DESC_ID'][first_row:last_row] = c['DATA_DESC_ID'][index]
            output_timeslot += 1

        block_index += 1
        if len(c['ANTENNA1']) < ts_blocks_per_chunk*time_avg_factor:
            break
    input_h5.close()
    output_h5.close()
