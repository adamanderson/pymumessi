"""
Sample script to demonstrate what InitGui.py does when you click on all the buttons
"""
# NB: Does not support multiple boards. This needs to be added.

import ConfigParser, pickle, time
import argparse as ap
import numpy as np
from pymumessi import hardware_map
from pymumessi.roach import Roach2
from pymumessi.autoZdokCal import loadDelayCal, findCal
from pymumessi.myQdr import Qdr as myQdr
from pymumessi.umux_logging import LoggingManager
import logging



LM = LoggingManager()
logger = LM.get_child_logger(target='all',  # fix target
                             alg_name='initialize',
                             setup=True)
rootlogger = logging.getLogger()
logger.setLevel(logging.DEBUG)
LoggingManager.set_console_level('DEBUG')


# Connect
def connect(hwm):
    logger.info("Connecting to roach")

    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        roach.connect()

        
# Program V6
def program_v6(hwm):
    logger.info("Programming V6 FPGA")
    
    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        roach.fpga.upload_to_ram_and_program(roach.FPGAFirmwareFile)
        fpgaClockRate = roach.fpga.estimate_fpga_clock()
        logger.info("Fpga Clock Rate: {}".format(fpgaClockRate))

        
# Initialize V7
def initialize_v7(hwm):
    logger.info('Initializing V7 FPGA')

    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        logger.info('Initializing UART')
        roach.initializeV7UART(waitForV7Ready=roach.waitForV7Ready)
        logger.info('UART initialized')

        logger.info('Initializing V7 MB')
        roach.initV7MB()
        logger.info('V7 MB initialized')

        logger.info('Setting LO to {} GHz'.format(2))
        roach.setLOFreq(2.e9)
        logger.info('LO frequency set to {} GHz'.format(2))

        roach.loadLOFreq()

        
# Calibrate Z-DOK
def calibrate_zdok(hwm):
    logger.info('Calibrating Z-DOK')

    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        roach.sendUARTCommand(0x4)
        time.sleep(0.1)
        roach.fpga.write_int('adc_in_i_scale',2**7) # set relative IQ scaling to 1
        roach.fpga.write_int('run',1)
        busDelays = [14,18,14,13]
        busStarts = [0,14,28,42]
        busBitLength = 12
        for iBus in xrange(len(busDelays)):
            logger.info('starting iBus {}'.format(iBus))
            delayLut = zip(np.arange(busStarts[iBus],busStarts[iBus]+busBitLength), 
                           busDelays[iBus] * np.ones(busBitLength))
            logger.info('delayLut = '.format(delayLut))
            loadDelayCal(roach.fpga,delayLut)
            logger.info('done with iBus {}'.format(iBus))

        calDict = findCal(roach.fpga)
        logger.info('calDict = '.format(calDict))

        roach.sendUARTCommand(0x5)
        logger.info('switched off ADC ZDOK Cal ramp')

        
# Calibrate QDR
def calibrate_qdr(hwm):
    logger.info('Calibrating QDR')

    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        bFailHard = False
        results = {}
        for iQdr,qdr in enumerate(roach.fpga.qdrs):
            mqdr = myQdr.from_qdr(qdr)
            results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard)
        logger.info('Qdr cal results: {}'.format(results))
    

if __name__ == '__main__':
    P = ap.ArgumentParser(description="Initialize readout boards.",
                          formatter_class=ap.RawTextHelpFormatter)
    P.add_argument('configFile', action="store",help="Config file to load")
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

    hwm = hardware_map.build(args.configFile)
    
    if 'connect' in args.stages:
        connect(hwm)

    if 'v6' in args.stages:
        program_v6(hwm)

    if 'v7' in args.stages:
        initialize_v7(hwm)

    if 'Zdok' in args.stages:
        calibrate_zdok(hwm)
        
    if 'QDR' in args.stages:
        calibrate_qdr(hwm)
