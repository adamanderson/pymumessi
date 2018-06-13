"""
Help interactive work with the roach board.

If you add a function here, "reload(iTools)" is your friend.

Sample use from iPython:

> import iTools
> rc = iTools.setup(100, 'chris.cfg')
> iTools.plotIQ(rc)

and if you make changes to code:

reload(iTools)

# now rc is a handle to a RoachConnection.
> avgIQData = rc.roachController.takeAvgIQData(100)
# This is a dictionary of 'I' and 'Q' of the 100 measurements at each frequency.


"""
iToolsVersion = 0.2

import ConfigParser
import RoachConnection
reload(RoachConnection)
import IQPlotWindow
reload(IQPlotWindow)

def getItoolsVersion():
    return iToolsVersion

def connect(roachNumber, configFile):
    rc = RoachConnection.RoachConnection(roachNumber, configFile)
    return rc

def loadFreq(rc):
    rc.loadFreq()

def setup(roachNumber, configFile):
    """
    for the roach number and confFile, set up the connection and load LUTs to prepare
    for making measurements
    """

    rc = connect(roachNumber, configFile)
    rc.loadFreq()
    rc.defineRoachLUTs()
    rc.defineDacLUTs()
    return rc

def plotIQ(rc):
    reload(IQPlotWindow)
    iqPlotWindow = IQPlotWindow.IQPlotWindow(rc)
    iqPlotWindow.show()
    return iqPlotWindow
