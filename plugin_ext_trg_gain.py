import datetime
from immutabledict import immutabledict
import strax
import numba
import numpy as np

# This makes sure shorthands for only the necessary functions
# are made available under straxen.[...]
export, __all__ = strax.exporter()

channel_list=list(np.arange(2000,2120))
#Channel List of nVeto PMTs
#Eventual Masked channels have to be added in a config file.
@export
@strax.takes_config(
    strax.Option('baseline_window',
                 default=(0,110),
                 help="Window (samples) for baseline calculation."),
    strax.Option('led_window',
                 default=(120, 220),
                 help="Window (samples) where we expect the signal in LED calibration"),
    strax.Option('integration_window',
                 default=(10, 20),
                 help="Integration window [-x,+y] from the peak"),
    strax.Option('channel_list',
                 default=(tuple(channel_list)),
                 help="List of PMTs. Defalt value: all the PMTs"))
    strax.Option('acq_window_length',
                 default=320,
                 help="Length of the Acq. Win. (samples). Defalt value: 320 samples"))

class NVLEDCalibration(strax.Plugin):
    """
    Preliminary version.
    LEDCalibration returns: channel, time, dt, lenght, Area,
    amplitudeLED and amplitude_index.
    The new variables are:
        - Area: Area computed in the given window. The integration is performed defining
        a dinamic window from the peak [ADC Counts x Samples].
        - amplitudeLED: peak amplitude of the LED on run in the given
        window [ADC Counts].
        - amplitude_index: amplitude of the LED on run in a window far
         from the signal one [Samples].
    """

    __version__ = '0.0.1'
    depends_on = ('raw_records_nv',)
    provides = 'led_cal_nv'
    data_kind = 'hitlets_nv'

    dtype = [('area', np.float32, 'Area calculated in integration window'),
             ('amplitude_led', np.float32, 'Amplitude in LED window'),
             ('channel', np.int16, 'Channel'),
             ('time', np.int64, 'Start time of the interval (ns since unix epoch)'),
             ('dt', np.int16, 'Time resolution in ns'),
             ('length', np.int32, 'Length of the interval in samples'),
             ('signal_time', np.int32, 'Sample of peak wrt trigger time (sample)')]

#------------------------------------------------------------------------------#
    def merge_waveform_longer(rr, channels=None, length=330):
        """
        Simple function to merge the records into a single waveform.
        We have two different Acquisition Window length:
        - LED_calibration_NV : 160 Samples -> 2 records (record0 and record1)
        - LED_calibration_NV_2 : 320 Samples -> 3 record
        The signal is expected between the record0 and record1. Thus the needed of a
        merged waveform.
        """
        record0=rr[rr["record_i"]==0]
        record1=rr[rr["record_i"]==1]
        record2=rr[rr["record_i"]==2]

        if channels != None:
            mask0           = np.where(np.in1d(record0['channel'], channels))[0]
            mask1           = np.where(np.in1d(record1['channel'], channels))[0]
            mask2           = np.where(np.in1d(record2['channel'], channels))[0]
            _raw_records0   = record0[mask0]
            _raw_records1   = record1[mask1]
            _raw_records2   = record2[mask2]
        else:
            _raw_records0 = record0
            _raw_records1 = record1
            _raw_records1 = record2


        record_length = length
        _dtype = [(('Channel/PMT number', 'channel'),                      np.int16),
                      (('Waveform data in raw ADC counts', 'data'), 'f8', (record_length,))]
        waveform = np.zeros(len(_raw_records0), dtype=_dtype)
        waveform['channel'] = _raw_records0['channel']
        waveform['data']    = np.concatenate((_raw_records0['data'][:, :],_raw_records1['data'][:, :],_raw_records2['data'][:, :]),axis=1)
        return waveform
#------------------------------------------------------------------------------#
    def get_baseline(raw_records,  channels=None, window_bsl=(0,110)):
        '''
        Function which estimates the baseline and its rms
        within the specified number of samples.
        '''
        if window_bsl == None: window_bsl = self.baseline_window

        if channels != None:
            mask           = np.where(np.in1d(raw_records['channel'], channels))[0]
            _raw_records   = raw_records[mask]
        else: _raw_records = raw_records
        _dtype = [(('Channel/PMT number', 'channel'),                      np.int16),
                      (('Baseline in the given window', 'baseline'),           np.float32),
                      (('Baseline error in the given window', 'baseline_err'), np.float32)]
        baseline = np.zeros(len(_raw_records), dtype=_dtype)
        baseline['channel']      = _raw_records['channel']
        baseline['baseline']     = _raw_records['data'][:, window_bsl[0]:window_bsl[1]].mean(axis=1)
        baseline['baseline_err'] = _raw_records['data'][:, window_bsl[0]:window_bsl[1]].std(axis=1)/np.sqrt(window_bsl[1] - window_bsl[0])
        return baseline
#------------------------------------------------------------------------------#

    def get_signal(raw_records, baseline, channels=None):
        '''
        Function which subtract the baseline to the waveform and invert it.
        '''


        if channels != None:
            mask           = np.where(np.in1d(raw_records['channel'], channels))[0]
            _raw_records   = raw_records[mask]
        else: _raw_records = raw_records

        record_length = np.shape(_raw_records.dtype['data'])[0]
        _dtype = [(('Channel/PMT number', 'channel'), '<i2'),
                  (('Waveform data in raw ADC counts', 'data'), 'f8', (record_length,))]
        signal = np.zeros(len(_raw_records), dtype=_dtype)

        signal['channel'] = _raw_records['channel']
        bsl               =  baseline ["baseline"]
        signal['data']    = -1. * (_raw_records['data'][:, :].transpose() - bsl[:]).transpose()

        return signal
#-----------------------------------------------------------------------------------------------------#
    def get_amplitude(records, channels=None,window=(0,110)):
        '''
        Function that compute the max (amplitude) in a given windows.
        '''
        if window == None: window = self.led_window

        if channels != None:
            mask       = np.where(np.in1d(records['channel'], channels))[0]
            _records   = records[mask]
        else: _records = records

        _dtype = [(('Channel/PMT number', 'channel'),                                    np.int16),
                  (('Amplitude in the given window', 'amplitude'),                       np.float32),
                  (('Sample/index of amplitude in the given window', 'amplitude_index'), np.float32)]
        amplitude = np.zeros(len(_records), dtype = _dtype)

        amplitude['channel']         = _records['channel']
        amplitude['amplitude']       = _records['data'][:, window[0]:window[1]].max(axis=1)
        amplitude['amplitude_index'] = _records['data'][:, window[0]:window[1]].argmax(axis=1) + window[0]

        return amplitude

#-----------------------------------------------------------------------------------------------------#

    def get_area(signal,  channels=None,window=(120,160)):
        '''
        Compute area in a given window.
        '''
        if window == None: window_bsl = self.integration_window
        if channels != None:
            mask       = np.where(np.in1d(signal['channel'], channels))[0]
            _records   = signal[mask]
        else: _records = signal

        _dtype = [(('Channel/PMT number', 'channel'),                   np.int16),
                  (('Integrated charge in a the given window', 'area'), np.float32)]
        area = np.zeros(len(_records), dtype = _dtype)

        area['channel'] = _records['channel']
        area['area']    = _records['data'][:, window[0]:window[1]].sum(axis=1)

        return area



    def compute(self, raw_records_nv):
        '''
        The data for LED calibration are build for those PMT which belongs to channel list.
        This is used for the different ligh levels. As defaul value all the PMTs are considered.
        '''
        channels=list(self.channel_list)
        mask = np.where(np.in1d(raw_records_nv['channel'],channels))[0]
        wave_signal=merge_waveform_longer(raw_records_nv,channels=channels, length=330)
        baseline=get_baseline(wave_signal,  channels=channels, window_bsl=(0,110))["baseline"]
        signal=get_signal(wave_signal, channels=channels,baseline=baseline)
        #signal    = get_records(rr, baseline_window=self.config['baseline_window'])
        del raw_records_nv
        temp = np.zeros(len(signal), dtype=self.dtype)
        #strax.copy_to_buffer(r, temp, "_recs_to_temp_led")

        on = get_amplitude(signal,channels=channels, window=(120,200))

        #on, off = get_amplitude(r, self.config['led_window'], self.config['noise_window'])
        temp['amplitude_led']   = on['amplitude']
        temp['signal_time']   = on['amplitude_index']
        #temp['amplitude_noise'] = off['amplitude']

        area = get_area(signal,channels=channels)
        #area = get_area(r, self.config['led_window'])
        temp['area'] = area['area']
        return temp
