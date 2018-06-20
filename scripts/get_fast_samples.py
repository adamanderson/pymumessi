import argparse as ap
from pymumessi.umux_logging import LoggingManager
import logging

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
channel_group.add_argument('--channels', nargs='*', action='store', default=[],
                           help='Channel numbers for which to take fast '
                           'samples (0-indexed)')
channel_group.add_argument('--all-channels', action='store_true',
                           default=False,
                           help='Get fast samples on all channels')

args = P.parse_args()

LM = LoggingManager()
logger = LM.get_child_logger(target=args.roachNum,
                             alg_name='initialize',
                             setup=True)
rootlogger = logging.getLogger()

# Get the configuration
logger.info('Reading configuration file: {}'.format(args.config))
config = ConfigParser.ConfigParser()
config.read(args.config)
for section in config.sections():
    logger.debug('   CONFIGURATION SECTION : {}'.format(section))
    for item in config.items(section):
        logger.debug(msg='   {:20s} : {}'.format(item[0], item[1]))

roach = Roach2(args.roachNum, args.config, False)
roach.connect() # do we really need to do this every time?
takePhaseStreamDataOfFreqChannel
if args.all_channels:
    # what function is this?
elif len(args.channels > 0):
    for channel in args.channels:
        roach.takePhaseStreamDataOfFreqChannel(freqChan=channel, duration=args.duration, pktsPerFrame=100, fabric_port=config.port, hostIP=roach.ip)
