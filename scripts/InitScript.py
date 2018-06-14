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

# Get the configuration
print "===> Read configuration"
config = ConfigParser.ConfigParser()
config.read(args.config)
print "Configuration read from {}".format(args.config)
for section in config.sections():
    print "   Configuration section:",section
    for item in config.items(section):
        print "   %20s : %s"%item

# Connect
if 'connect' in args.stages:
    print "===>Connect"
    ipaddress = config.get('Roach {}'.format(args.roachNum), 'ipaddress')
    FPGAParamFile = config.get('Roach {}'.format(args.roachNum), 'FPGAParamFile')
    roachController = Roach2(args.roachNum, args.config, True, False)
    roachController.connect()

# Program V6
if 'v6' in args.stages:
    print "===>Program V6"
    fpgPath = config.get('Roach {}'.format(args.roachNum), 'fpgPath')
    roachController.fpga.upload_to_ram_and_program(fpgPath)
    fpgaClockRate = roachController.fpga.estimate_fpga_clock()
    print "Fpga Clock Rate:",fpgaClockRate

# Initialize V7
if 'v7' in args.stages:
    print "===>Initialize V7"
    waitForV7Ready=config.getboolean('Roach {}'.format(args.roachNum),'waitForV7Ready')
    roachController.initializeV7UART(waitForV7Ready=waitForV7Ready)
    print 'initialized uart'
    roachController.initV7MB()
    print 'initialized mb'
    roachController.setLOFreq(2.e9)
    roachController.loadLOFreq()
    print 'Set LO to 2 GHz'

# Calibrate Z-DOK
if 'Zdok' in args.stages:
    print "===>Calibrate Z-DOK"
    roachController.sendUARTCommand(0x4)
    time.sleep(0.1)
    roachController.fpga.write_int('adc_in_i_scale',2**7) # set relative IQ scaling to 1
    roachController.fpga.write_int('run',1)
    busDelays = [14,18,14,13]
    busStarts = [0,14,28,42]
    busBitLength = 12
    for iBus in xrange(len(busDelays)):
        print "iBus =",iBus
        delayLut = zip(np.arange(busStarts[iBus],busStarts[iBus]+busBitLength), 
                       busDelays[iBus] * np.ones(busBitLength))
        print "delayLut =",delayLut
        loadDelayCal(roachController.fpga,delayLut)
        print "done with iBus =",iBus

    calDict = findCal(roachController.fpga)
    print "calDict=",calDict

    roachController.sendUARTCommand(0x5)
    print 'switched off ADC ZDOK Cal ramp'

# Calibrate QDR
if 'QDR' in args.stages:
    print "===>Calibrate QDR"
    calVerbosity = 0
    bFailHard = False
    results = {}
    for iQdr,qdr in enumerate(roachController.fpga.qdrs):
        mqdr = myQdr.from_qdr(qdr)
        results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)
    print 'Qdr cal results:',results
    
