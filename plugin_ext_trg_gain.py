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

    def compute(self, raw_records_nv):
            '''
            The data for LED calibration are build for those PMT which belongs to channel list.
            This is used for the different ligh levels. As defaul value all the PMTs are considered.
            '''
            channels=list(self.channel_list)
            mask = np.where(np.in1d(raw_records_nv['channel'],channels))[0]
