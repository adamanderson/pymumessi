"""
Connect to the roach board, and keep track of state.

The logic is copied from RoachStateMachine.  RoachConnect does not have any GUI references.
"""

import ConfigParser, os, pickle
import numpy as np
from Roach2Controls import Roach2Controls

class RoachConnection():
    def __init__(self, roachNumber, configFile):
        """
        Connect to a roach board

        Inputs:
        roach number -- 100, for example, to match stanza in configFile
        configFile -- the file with the, um, configuration information
        """
        self.config = ConfigParser.ConfigParser()
        self.config.read(configFile)
        self.roachString = 'Roach '+"%d"%roachNumber
        self.FPGAParamFile = self.config.get(self.roachString,'FPGAParamFile')
        self.ipaddress = self.config.get(self.roachString,'ipaddress')
        self.roachController = Roach2Controls(self.ipaddress,self.FPGAParamFile,True,False)
        self.roachController.connect()
        self.originalDdsShift = self.roachController.checkDdsShift()
        self.newDdsShift = self.roachController.loadDdsShift(self.originalDdsShift)

    def loadFreq(self):
        '''
        Loads the resonator freq files (and attenuations, resIDs)
        divides the resonators into streams
        '''
        try:
            print 'old Freq: ', self.roachController.freqList
        except: pass
        fn = self.config.get(self.roachString,'freqfile')
        fn2=fn.rsplit('.',1)[0]+'_NEW.'+ fn.rsplit('.',1)[1]         # Check if ps_freq#_NEW.txt exists
        print "RoachStateMachine.loadFreq:  fn,fn2=",fn,fn2
        if os.path.isfile(fn2): 
            fn=fn2
            print 'Loading freqs from '+fn
        
        freqFile = np.loadtxt(fn)
        
        #if np.shape(freqFile)[1]==3:
        if len(np.shape(freqFile))==2: # more than 1 frequency given
            resIDs = np.atleast_1d(freqFile[:,0])       # If there's only 1 resonator numpy loads it in as a float.
            freqs = np.atleast_1d(freqFile[:,1])     # We need an array of floats
            attens = np.atleast_1d(freqFile[:,2])
            phaseOffsList = np.zeros(len(freqs))
            iqRatioList = np.ones(len(freqs))        
        elif len(np.shape(freqFile)) == 1: # only 1 frequency given
            resIDs = np.atleast_1d(freqFile[0])       # If there's only 1 resonator numpy loads it in as a float.
            freqs = np.atleast_1d(freqFile[1])        # Convert this to an array of floats
            attens = np.atleast_1d(freqFile[2])
            phaseOffsList = np.zeros(len(freqs))
            iqRatioList = np.ones(len(freqs))        
        else:
            raise ValueError('I can not deal with len(np.shape(freqFile)) = %d'%len(np.shape(freqFile)))

        assert(len(freqs) == len(np.unique(freqs))), "Frequencies in "+fn+" need to be unique."
        assert(len(resIDs) == len(np.unique(resIDs))), "Resonator IDs in "+fn+" need to be unique."
        argsSorted = np.argsort(freqs)  # sort them by frequency (I don't think this is needed)
        freqs = freqs[argsSorted]
        resIDs = resIDs[argsSorted]
        attens = attens[argsSorted]
        phaseOffsList = iqRatioList[argsSorted]
        iqRatioList = iqRatioList[argsSorted]
        for i in range(len(freqs)):
            print i, resIDs[i], freqs[i], attens[i], phaseOffsList[i], iqRatioList[i]
        
        self.roachController.generateResonatorChannels(freqs)
        self.roachController.attenList = attens
        self.roachController.resIDs = resIDs
        self.roachController.phaseOffsList = phaseOffsList
        self.roachController.iqRatioList = iqRatioList
        print 'new Freq: ', self.roachController.freqList

        return True
    
    def defineRoachLUTs(self):
        '''
        Defines LO Freq but doesn't load it yet
        Defines and loads channel selection blocks
        Defines and loads DDS LUTs
        
        writing the QDR takes a long time! :-(
        '''
        loFreq = int(self.config.getfloat(self.roachString,'lo_freq'))
        self.roachController.setLOFreq(loFreq)
        self.roachController.generateFftChanSelection()
        ddsTones = self.roachController.generateDdsTones()
        with open("ddsTones.pkl", 'wb') as handle:
            pickle.dump(ddsTones, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.roachController.loadChanSelection()
        self.roachController.loadDdsLUT()
        return True
    
    def defineDacLUTs(self):
        '''
        Defines and loads DAC comb
        Loads LO Freq
        Loads DAC attens 1, 2
        Loads ADC attens 1, 2
        '''

        adcAtten = self.config.getfloat(self.roachString,'adcatten')
        dacAtten = self.config.getfloat(self.roachString,'dacatten_start')
        dacAtten1 = np.floor(dacAtten*2)/4.
        dacAtten2 = np.ceil(dacAtten*2)/4.

        dacComb = self.roachController.generateDacComb(globalDacAtten=dacAtten)
        with open("dacComb.pkl", 'wb') as handle:
            pickle.dump(dacComb, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print "Initializing ADC/DAC board communication"
        self.roachController.initializeV7UART()
        print "Setting Attenuators"
        self.roachController.changeAtten(1,dacAtten1)
        self.roachController.changeAtten(2,dacAtten2)
        self.roachController.changeAtten(3,adcAtten)
        print "Setting LO Freq"
        self.roachController.loadLOFreq()
        print "Loading DAC LUT"
        self.roachController.loadDacLUT()
        return True





