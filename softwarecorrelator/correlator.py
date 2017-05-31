import numpy
import logging
import h5py

from .stationprocessing import read_and_process_antenna_block_mp, fir_filter_coefficients, samples_per_block
from .inspect import find_sas_id, complex_voltage_obs_header

from lofarantpos.db import LofarAntennaDatabase








def cross_correlate(input_dir_name, 
                    output_filename_template,
                    sas_id_string=None,
                    integration_s=0.1, num_chan=128, num_taps=16,
                    max_duration_s=10000,
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

    for i, (x, y, time_axis, freq_axis) in enumerate(read_and_process_antenna_block_mp(
            input_dir_name, sas_id, sap_ids,
            fir_coefficients=fir_filter_coefficients(num_chan=num_chan, num_taps=num_taps),
            interval_samples=samples_per_interval,
            num_samples=samples_to_read,
            max_duration_s=max_duration_s)):
        if len(h5_output_filenames) == 0:
            num_sb  = x.shape[1]
            num_ant = x.shape[0]
            h5_output_filenames = [output_filename_template % {'sas_id': sas_id,
                                                               'antenna_set': obs_header['ANTENNA_SET'].decode('utf8'),
                                                               'obs_datetime': obs_datetime_compact,
                                                               'subband': sb}
                                   for sb in range(num_sb)]
            h5_output_files = [h5py.File(name, mode='w-') for name in h5_output_filenames]
            
        # xx.append([[(abs(x[ant,sb,:,:])**2).mean(axis=0)
        #            for sb in range(x.shape[1])]
        #            for ant in range(x.shape[0])])
        # yy.append([[(abs(y[ant, sb,:,:])**2).mean(axis=0)
        #            for sb in range(y.shape[1])]
        #            for ant in range(y.shape[0])])
        time_s.append(time_axis['REFERENCE_VALUE'])

        if time_s[-1] > next_report_s:
            next_report_s += progress_interval_s
            print(time_s[-1])
    [h5_file.close() for h5_file in h5_output_files]
    time_s = numpy.array(time_s)
    return h5_output_filenames
