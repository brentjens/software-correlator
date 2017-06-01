import numpy
import logging
import h5py

from .stationprocessing import read_and_process_antenna_block_mp, fir_filter_coefficients, samples_per_block
from .inspect import find_sas_id, complex_voltage_obs_header

from lofarantpos.db import LofarAntennaDatabase







def create_visibility_hdf5(file_name, num_timeslots, num_ant, num_chan, num_pol, num_sb=1):
    num_bl = (num_ant*(num_ant+1))//2
    num_rows = num_timeslots*num_bl*num_sb
    h5file = h5py.File(file_name, mode='w-')
    h5file.create_dataset('MAIN/DATA', (num_rows, num_chan, num_pol), dtype='c8')
    h5file.create_dataset('MAIN/FLAG', (num_rows, num_chan, num_pol), dtype='b')
    h5file.create_dataset('MAIN/ANTENNA1', (num_rows,), dtype='i4')
    h5file.create_dataset('MAIN/ANTENNA2', (num_rows,), dtype='i4')
    h5file.create_dataset('MAIN/DATA_DESC_ID', (num_rows,), dtype='i4')
    h5file.create_dataset('MAIN/TIME', (num_rows,), dtype='f8')

    h5file.create_dataset('SPECTRAL_WINDOW/NUM_CHAN', (num_sb, 1), dtype='i8')
    h5file.create_dataset('SPECTRAL_WINDOW/CHAN_FREQ', (num_sb, num_chan), dtype='f8')

    h5file.create_dataset('ANTENNA/NAME', (num_ant, 1), dtype='S512')
    h5file.create_dataset('ANTENNA/ITRF', (num_ant, 3), dtype='f8')
    return h5file


def cross_correlate(input_dir_name, 
                    output_filename_template,
                    sas_id_string=None,
                    integration_s=0.1, num_chan=128, num_taps=16,
                    max_duration_s=None,
                    sap_ids=range(48),
                    progress_interval_s=5.0):
    r'''
    Write one file per sub band.

    output_template = os.path.join(args.output_dir,'%(sas_id)s-%(antenna_set)s-%(obs_datetime)s-SB%(subband)03d.hdf5')

    '''
    sas_id = find_sas_id(input_dir_name, sas_id_string)
    db = LofarAntennaDatabase()
    
    obs_header = complex_voltage_obs_header(input_dir_name, sas_id)
    obs_header['CORR_INTEGRATION_TIME'] = integration_s
    obs_header['CORR_INTEGRATION_TIME_UNIT'] = b's'
    obs_header['OBSERVATION_STATION_PHASE_CENTRES_ETRS'] = [
        db.phase_centres[station.decode('utf8')]
        for station in obs_header['OBSERVATION_STATIONS_LIST']]
    obs_header['OBSERVATION_PQR_TO_ETRS_MATRICES'] = [
        db.pqr_to_etrs[station.decode('utf8')]
        for station in obs_header['OBSERVATION_STATIONS_LIST']]

    obs_start_mjd_days = obs_header['OBSERVATION_START_MJD']
    obs_start_utc_str =  obs_header['OBSERVATION_START_UTC'].decode('utf8')
    obs_datetime_compact = ''.join([ch for ch in obs_start_utc_str[0:19] if ch not in '-:T'])
    obs_datetime_compact = '%s_%s' % (obs_datetime_compact[0:8], obs_datetime_compact[8:], )
    logging.info(obs_header)

    fir_coef = fir_filter_coefficients(num_chan, num_taps)

    time_s = []
    h5_output_filenames = []
    h5_output_files = []
    next_report_s = 0.0
    samples_per_interval, samples_to_read = samples_per_block(
        block_length_s=integration_s,
        sample_duration_s=512/100e6,
        num_chan=num_chan, num_taps=num_taps)
    if max_duration_s is None:
        max_duration_s = obs_header['TOTAL_INTEGRATION_TIME']
    max_samples = int(numpy.floor(max_duration_s / (512/100e6)))
    num_timeslots = int(numpy.floor((max_samples - samples_to_read)/samples_per_interval)+1)
    logging.debug('cross_correlate(): max_duration_s(%f); max_samples(%d); num_timeslots(%d)',
                  max_duration_s, max_samples, num_timeslots)
    for i, (x, y, time_axis, freq_axis) in enumerate(read_and_process_antenna_block_mp(
            input_dir_name, sas_id, sap_ids,
            fir_coefficients=fir_filter_coefficients(num_chan=num_chan, num_taps=num_taps),
            interval_samples=samples_per_interval,
            num_samples=samples_to_read,
            max_duration_s=max_duration_s)):

        if len(h5_output_filenames) == 0:
            num_sb  = x.shape[1]
            num_ant = x.shape[0]
            num_bl = (num_ant*(num_ant+1))//2
            num_rows = num_timeslots*num_bl
            logging.debug('cross_correlate(): num_sb(%d); num_ant(%d); num_bl(%d); num_rows(%d);',
                          num_sb, num_ant, num_bl, num_rows)
            current_row = 0
            h5_output_filenames = [output_filename_template % {'sas_id': sas_id,
                                                               'antenna_set': obs_header['ANTENNA_SET'].decode('utf8'),
                                                               'obs_datetime': obs_datetime_compact,
                                                               'subband': sb}
                                   for sb in range(num_sb)]
            h5_output_files = [create_visibility_hdf5(name,
                                                      num_timeslots=num_timeslots,
                                                      num_ant=num_ant,
                                                      num_chan=num_chan,
                                                      num_pol=4)
                               for name in h5_output_filenames]
        # x and y original indices: [antenna, subband, timeslot, channel]
        # x and y new indices: [antenna, subband, channel, timeslot]
        # aim is to enable efficient summation of time axis.
        x = x.transpose((0,1,3,2)).copy()
        xc = numpy.conj(x)
        y = y.transpose((0,1,3,2)).copy()
        yc = numpy.conj(y)
        for subband, h5file in enumerate(h5_output_files):
            vis = numpy.zeros((num_bl, num_chan, 4), dtype='c8')
            flags = numpy.zeros((num_bl, num_chan, 4), dtype='b')
            bl = 0
            ant1 = []
            ant2 = []
            for antenna1 in range(num_ant):
                for antenna2 in range(antenna1, num_ant):
                    vis[bl,:,0] = (x[antenna1, subband, :,:]*xc[antenna2, subband, :,:]).mean(axis=-1)
                    vis[bl,:,1] = (x[antenna1, subband, :,:]*yc[antenna2, subband, :,:]).mean(axis=-1)
                    vis[bl,:,2] = (y[antenna1, subband, :,:]*xc[antenna2, subband, :,:]).mean(axis=-1)
                    vis[bl,:,3] = (y[antenna1, subband, :,:]*yc[antenna2, subband, :,:]).mean(axis=-1)
                    ant1.append(antenna1)
                    ant2.append(antenna2)
                    bl += 1
            h5file['MAIN/DATA'][current_row:current_row+num_bl,:,:] = vis
            h5file['MAIN/FLAG'][current_row:current_row+num_bl,:,:] = flags
            h5file['MAIN/ANTENNA1'][current_row:current_row+num_bl] = ant1
            h5file['MAIN/ANTENNA2'][current_row:current_row+num_bl] = ant2
            h5file['MAIN/TIME'][current_row:current_row+num_bl] = [obs_start_mjd_days*24*3600 + time_axis['REFERENCE_VALUE']]*num_bl
        current_row += num_bl
        time_s.append(time_axis['REFERENCE_VALUE'])

        if time_s[-1] > next_report_s:
            next_report_s += progress_interval_s
            print(time_s[-1])
    [h5_file.close() for h5_file in h5_output_files]
    time_s = numpy.array(time_s)
    return h5_output_filenames
