"""
Sample script to demonstrate what InitGui.py does when you click on all the buttons
"""
# Import what we need
import ConfigParser, pickle, sys, time
import numpy as np
from pymumessi.Roach2Controls import Roach2Controls
from pymumessi.autoZdokCal import loadDelayCal, findCal
from pymumessi.myQdr import Qdr as myQdr

if len(sys.argv) > 1:
    roachNumber = int(sys.argv[1])
else:
    roachNumber = 100

# Get the configuration
print "===> Read configuration"
config = ConfigParser.ConfigParser()
config.read('../config/chris.cfg')
print "Configuration read from init.cfg"
for section in config.sections():
    print "   Configuration section:",section
    for item in config.items(section):
        print "   %20s : %s"%item

# Connect
print "===>Connect"
ipaddress = config.get('Roach %d'%roachNumber, 'ipaddress')
FPGAParamFile = config.get('Roach %d'%roachNumber, 'FPGAParamFile')
roachController = Roach2Controls(ipaddress, FPGAParamFile, True, False)
roachController.connect()

# Program V6
print "===>Program V6"
fpgPath = config.get('Roach %d'%roachNumber, 'fpgPath')
roachController.fpga.upload_to_ram_and_program(fpgPath)
fpgaClockRate = roachController.fpga.estimate_fpga_clock()
print "Fpga Clock Rate:",fpgaClockRate

# Initialize V7
print "===>Initialize V7"
waitForV7Ready=config.getboolean('Roach %d'%roachNumber,'waitForV7Ready')
roachController.initializeV7UART(waitForV7Ready=waitForV7Ready)
print 'initialized uart'
roachController.initV7MB()
print 'initialized mb'
roachController.setLOFreq(2.e9)
roachController.loadLOFreq()
print 'Set LO to 2 GHz'

# Calibrate Z-DOK
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
print "===>Calibrate QDR"
calVerbosity = 0
bFailHard = False
results = {}
for iQdr,qdr in enumerate(roachController.fpga.qdrs):
    mqdr = myQdr.from_qdr(qdr)
    results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)
print 'Qdr cal results:',results
    
