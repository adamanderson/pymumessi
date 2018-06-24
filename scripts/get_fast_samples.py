import argparse as ap
from pymumessi.umux_logging import LoggingManager
from pymumessi.roach import Roach2
import logging
from ConfigParser import ConfigParser

P = ap.ArgumentParser(description='Get fast (1Msample/sec) phase samples '
                      'from board.',
                      formatter_class=ap.RawTextHelpFormatter)

P.add_argument('roachNum', default=100, action='store', type=int,
               help='Roach number to initialize')
P.add_argument('duration', default=1.0, action='store', type=float,
               help='Duration of phase samples to save.')

P.add_argument('-c', '--config', action='store', default='init.cfg',
               help='Config file to load')
channel_group = P.add_mutually_exclusive_group()
channel_group.add_argument('--channels', nargs='*', action='store', type=int,
                           default=[], help='Channel numbers for which to '
                           'take fast samples (0-indexed)')
channel_group.add_argument('--all-channels', action='store_true',
                           default=False,
                           help='Get fast samples on all channels')

args = P.parse_args()

LM = LoggingManager()
logger = LM.get_child_logger(target=args.roachNum,
                             alg_name='initialize',
                             setup=True)
rootlogger = logging.getLogger()
    
roach = Roach2(args.roachNum, args.config, False)
roach.connect() # do we really need to do this every time?

if args.all_channels:
    chanlist = roach.resonator_ids
else:
    chanlist = args.channels

output = dict()
for channel in chanlist:
    output[channel] = roach.takePhaseStreamDataOfFreqChannel(freqChan=channel,
                                                             duration=args.duration,
                                                             pktsPerFrame=100,
                                                             fabric_port=roach.port,
                                                             hostIP=roach.ip)
