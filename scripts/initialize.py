"""
Sample script to demonstrate what InitGui.py does when you click on all the buttons
"""
# NB: Does not support multiple boards. This needs to be added.

import ConfigParser, pickle, time
import argparse as ap
import numpy as np
from pymumessi.roach import Roach2
from pymumessi.autoZdokCal import loadDelayCal, findCal
from pymumessi.myQdr import Qdr as myQdr
from pymumessi.umux_logging import LoggingManager
import logging

P = ap.ArgumentParser(description="Initialize readout boards.",
                      formatter_class=ap.RawTextHelpFormatter)

P.add_argument("roachNum", default=100, action="store", type=int,
               help="Roach number to initialize")

P.add_argument("-c", "--config", action="store", default="init.cfg",
               help="Config file to load")
P.add_argument('--stages', nargs='*', dest='stages', action='store',
               choices=['connect', 'v6', 'v7', 'Zdok', 'QDR'],
               default=['connect', 'v6', 'v7', 'Zdok', 'QDR'],
               help='Initialization stages to run. Choices are:'
               '\nconnect : connect to readout boards'
               '\nv6 : program the V6 FPGA'
               '\nv7 : initialize the V7 FPGA'
               '\nZdok : calibrate the Zdok interface'
               '\nQDR : calibrate the QDR')

args = P.parse_args()


LM = LoggingManager()
logger = LM.get_child_logger(target=args.roachNum,
                             alg_name='initialize',
                             setup=True)
rootlogger = logging.getLogger()
logger.setLevel(logging.DEBUG)
LoggingManager.set_console_level('DEBUG')

# Get the configuration
logger.log(level=logging.INFO,
           msg="Reading configuration file: {}".format(args.config))
config = ConfigParser.ConfigParser()
config.read(args.config)
for section in config.sections():
    logger.log(level=logging.INFO,
               msg="   CONFIGURATION SECTION : {}".format(section))
    for item in config.items(section):
        logger.log(level=logging.WARNING, msg="   {:20s} : {}".format(item[0], item[1]))

# Connect
if 'connect' in args.stages:
    logger.log(level=logging.INFO, msg="Connecting to roach")
    ipaddress = config.get('Roach {}'.format(args.roachNum), 'ipaddress')
    FPGAParamFile = config.get('Roach {}'.format(args.roachNum), 'FPGAParamFile')
    roachController = Roach2(args.roachNum, args.config, True, False)
    roachController.connect()

# Program V6
if 'v6' in args.stages:
    logger.log(level=logging.INFO, msg="Programming V6 FPGA")
    fpgPath = config.get('Roach {}'.format(args.roachNum), 'fpgPath')
    roachController.fpga.upload_to_ram_and_program(fpgPath)
    fpgaClockRate = roachController.fpga.estimate_fpga_clock()
    logger.log(level=logging.INFO, msg="Fpga Clock Rate: {}".format(fpgaClockRate))

# Initialize V7
if 'v7' in args.stages:
    logger.log(level=logging.INFO, msg='Initializing V6 FPGA')
    
    waitForV7Ready=config.getboolean('Roach {}'.format(args.roachNum),'waitForV7Ready')

    logger.log(level=logging.INFO, msg='Initializing UART')
    roachController.initializeV7UART(waitForV7Ready=waitForV7Ready)
    logger.log(level=logging.INFO, msg='UART initialized')

    logger.log(level=logging.INFO, msg='Initializing V7 MB')
    roachController.initV7MB()
    logger.log(level=logging.INFO, msg='V7 MB initialized')

    logger.log(level=logging.INFO, msg='Setting LO to {} GHz'.format(2))
    roachController.setLOFreq(2.e9)
    logger.log(level=logging.INFO, msg='LO frequency set to {} GHz'.format(2))

    roachController.loadLOFreq()
    print 'Set LO to 2 GHz'

# Calibrate Z-DOK
if 'Zdok' in args.stages:
    logger.log(level=logging.INFO, msg='Calibrating Z-DOK')
    roachController.sendUARTCommand(0x4)
    time.sleep(0.1)
    roachController.fpga.write_int('adc_in_i_scale',2**7) # set relative IQ scaling to 1
    roachController.fpga.write_int('run',1)
    busDelays = [14,18,14,13]
    busStarts = [0,14,28,42]
    busBitLength = 12
    for iBus in xrange(len(busDelays)):
        logger.log(level=logging.INFO, msg='starting iBus {}'.format(iBus))
        delayLut = zip(np.arange(busStarts[iBus],busStarts[iBus]+busBitLength), 
                       busDelays[iBus] * np.ones(busBitLength))
        logger.log(level=logging.INFO, msg='delayLut = '.format(delayLut))
        loadDelayCal(roachController.fpga,delayLut)
        logger.log(level=logging.INFO, msg='done with iBus {}'.format(iBus))

    calDict = findCal(roachController.fpga)
    logger.log(level=logging.INFO, msg='calDict = '.format(calDict))

    roachController.sendUARTCommand(0x5)
    logger.log(level=logging.INFO, msg='switched off ADC ZDOK Cal ramp')

# Calibrate QDR
if 'QDR' in args.stages:
    logger.log(level=logging.INFO, msg='Calibrating QDR')
    calVerbosity = 0
    bFailHard = False
    results = {}
    for iQdr,qdr in enumerate(roachController.fpga.qdrs):
        mqdr = myQdr.from_qdr(qdr)
        results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)
    logger.log(level=logging.INFO, msg='Qdr cal results: {}'.format(results))
    
