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
                    sap_ids=range(48)):
    r'''
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
    
    logging.info(obs_header)
    
    
    fir_coef = fir_filter_coefficients(num_chan, num_taps)
    time_s = []
    xx = []
    yy = []
    report_interval_s = 2.0
    next_report_s = 0.0
    samples_per_interval, samples_to_read = samples_per_block(
        block_length_s=integration_s,
        sample_duration_s=512/100e6,
        num_chan=num_chan, num_taps=num_taps)
    print(samples_per_interval, samples_to_read)
    for i, (x, y, time_axis, freq_axis) in enumerate(read_and_process_antenna_block_mp(
            input_dir_name, sas_id, sap_ids,
            fir_coefficients=fir_filter_coefficients(num_chan=num_chan, num_taps=num_taps),
            interval_samples=samples_per_interval,
            num_samples=samples_to_read)):
        if len(time_s)> 0 and time_s[-1] > max_duration_s-integration_s/2:
            break
        #print(x.shape, y.shape)            
        xx.append([[(abs(x[ant,sb,:,:])**2).mean(axis=0)
                   for sb in range(x.shape[1])]
                   for ant in range(x.shape[0])])
        yy.append([[(abs(y[ant, sb,:,:])**2).mean(axis=0)
                   for sb in range(y.shape[1])]
                   for ant in range(y.shape[0])])
        time_s.append(time_axis['REFERENCE_VALUE'])

        if time_s[-1] > next_report_s:
            next_report_s += report_interval_s
            print(time_s[-1])
    xx = numpy.array(xx, dtype=numpy.complex64).transpose(1,2,0,3)
    yy = numpy.array(yy, dtype=numpy.complex64).transpose(1,2,0,3)
    time_s = numpy.array(time_s)
    return xx, yy, time_s, freq_axis
