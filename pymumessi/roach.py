"""
AJA: I have consolidated the classes formerly called Roach2Controls,
RoachConnection, and RoachStateMachine into a single class called a Roach.
This seemed like a more logical organization.

Author:    Alex Walter
Date:      April 25, 2016
Firmware:  darkS2*.fpg

This class is for setting and reading LUTs, registers, and other memory components in the ROACH2 Virtex 6 FPGA using casperfpga tools.
It's also the IO for the ADC/DAC board's Virtex 7 FPGA through the ROACH2

NOTE: All freqencies are considered positive. A negative frequency can be asserted by the aliased signal of large positive frequency (by adding sample rate).
      This makes things easier for coding since I can check valid frequencies > 0 and also for calculating which fftBin a frequency resides in (see generateFftChanSelection()). 


Example usage:
    # Collect MKID info
    nFreqs=1024
    loFreq = 5.e9
    spacing = 2.e6
    freqList = np.arange(loFreq-nFreqs/2.*spacing,loFreq+nFreqs/2.*spacing,spacing)
    freqList+=np.random.uniform(-spacing,spacing,nFreqs)
    freqList = np.sort(freqList)
    attenList = np.random.randint(23,33,nFreqs)
    
    # Talk to Roach
    roach_0 = Roach2Controls(ip, params, True, True)
    roach_0.setLOFreq(loFreq)
    roach_0.generateResonatorChannels(freqList)
    roach_0.generateFftChanSelection()
    roach_0.generateDacComb(freqList=None, resAttenList=attenList, globalDacAtten=17)
    roach_0.generateDdsTones()
    
    roach_0.loadChanSelection()
    roach_0.loadDacLUT()




List of Functions:
    __init__ -                      Connects to Roach2 FPGA, sets the delay between the dds lut and the end of the fft block
    connect -                       Connect to V6 FPGA on Roach2
    initializeV7UART -              Initializes the UART connection to the Virtex 7
    loadDdsShift -                  Set the delay between the dds lut and the end of the fft block
    generateResonatorChannels -     Figures out which stream:channel to assign to each resonator frequency
    generateFftChanSelection -      Assigns fftBins to each steam:channel
    loadSingleChanSelection -       Loads a channel for each stream into the channel selector LUT
    loadChanSelection -             Loops through loadSingleChanSelection()
    setLOFreq -                     Defines LO frequency as an attribute, self.LOFreq
    loadLOFreq -                    Loads the LO frequency to the IF board
    generateTones -                 Returns a list of I,Q time series for each frequency provided
    generateDacComb -               Returns a single I,Q time series representing the DAC freq comb
    loadDacLut -                    Loads the freq comb from generateDacComb() into the LUT
    generateDdsTones -              Defines interweaved tones for dds
    loadDdsLUT -                    Loads dds tones into Roach2 memory
    

    
List of useful class attributes:
    ip -                            ip address of roach2
    params -                        Dictionary of parameters
    freqList -                      List of resonator frequencies
    attenList -                     List of resonator attenuations
    freqChannels -                  2D array of frequencies. Each column is the a stream and each row is a channel. 
                                    If uneven number of frequencies this array is padded with -1's
    fftBinIndChannels -             2D array of fftBin indices corresponding to the frequencies/streams/channels in freqChannels. freq=-1 maps to fftBin=0.
    dacPhaseList -                  List of the most recent relative phases used for generating DAC frequency comb
    dacScaleFactor -                Scale factor for frequency comb to scale the sum of tones onto the DAC's dynamic range. 
                                    Careful, this is 1/scaleFactor we defined for ARCONS templar
    dacQuantizedFreqList -          List of frequencies used to define DAC frequency comb. Quantized to DAC digital limits
    dacFreqComb -                   Complex time series signal used for DAC frequency comb. 
    LOFreq -                        LO frequency of IF board
    ddsQuantizedFreqList -          2D array of frequencies shaped like freqChannels. Quantized to dds digital limits
    ddsPhaseList -                  2D array of frequencies shaped like freqChannels. Used to rotate loops.
    

TODO:
    modify takePhaseSnapshot() to work for nStreams: need register names
    In performIQSweep(), is it skipping every other LO freq???
    Changed delays in performIQSweep(), and takeAvgIQData() from 0.1 to 0.01 seconds

BUGS:
    The frequencies in freqList are assumed to be unique. 
    If they aren't, then there are problems determining which frequency corresponds to which ch/stream. 
    This should be fixed with some indexing tricks which don't rely on np.where
"""

import sys,os,time,datetime,struct,math,calendar
import inspect
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import casperfpga
import ConfigParser
import socket
import binascii
import pickle
from lib.binTools import castBin
from lib.readDict import readDict
import logging

rootlogger = logging.getLogger()

class Roach2:

    def __init__(self, roachNumber, configFile, debug=False):
        '''
        Parameters:
        ----------
        roachNumber : int
            Number of the roach, corresponding to settings in configFile
        configFile : str
            Path to file with configuration settings for each of the boards and defaults
        debug : bool
            Saves data to disk for debugging

        Returns:
        --------
        None
        '''

        self.config = ConfigParser.ConfigParser()
        self.config.read(configFile)
        self.roachString = 'Roach {:d}'.format(roachNumber)
        self.FPGAParamFile = self.config.get(self.roachString, 'FPGAParamFile')
        self.FPGAFirmwareFile = self.config.get(self.roachString, 'fpgPath')
        self.ip = self.config.get(self.roachString, 'ipaddress')
        self.port = int(self.config.get(self.roachString, 'port'))
        
        self.debug=debug

        try:
            self.params = readDict()             
            self.params.readFromFile(self.FPGAParamFile)
        except TypeError:
            self.params = self.FPGAParamFile
        
        if debug and not os.path.exists(self.params['debugDir']):
            os.makedirs(self.params['debugDir']) 

        
        #Some more parameters
        self.freqPadValue = -1      # pad frequency lists so that we have a multiple of number of streams
        self.fftBinPadValue = 0     # pad fftBin selection with fftBin 0
        self.ddsFreqPadValue = -1   # 
        self.v7_ready = 0
        self.lut_dump_buffer_size = self.params['lut_dump_buffer_size']
        self.thresholdList = -np.pi*np.ones(1024)
        self.waitForV7Ready = self.config.getboolean(self.roachString, 'waitForV7Ready')
        
        self.connect()
        self.originalDdsShift = self.checkDdsShift()
        self.newDdsShift = self.loadDdsShift(self.originalDdsShift)

        # AJA: Probably can be deprecated by the hardware map machinery.
        # # load the frequencies
        # freq_filename = self.config.get(self.roachString, 'freqfile')
        # freq_data = np.loadtxt(freq_filename)
        # self.resonator_ids = np.atleast_1d(freq_data[:,0])
        # self.frequencies = np.atleast_1d(freq_data[:,1])
        # self.attenuations = np.atleast_1d(freq_data[:,2])

        # # construct the channel information
        # self.generate_resonator_channels(self.frequencies)
        
    def connect(self):
        try:
            self.fpga = casperfpga.casperfpga.CasperFpga(self.ip,timeout=30.)
        except:
            self.fpga = casperfpga.katcp_fpga.KatcpFpga(self.ip,timeout=30.)
        time.sleep(.1)
        self.fpga._timeout = 50.
        if not self.fpga.is_running():
            
            logging.info('Firmware is not running. Start firmware, calibrate, and load wave into qdr first!')
        else:
            self.fpga.get_system_information()
    
    def checkDdsShift(self):
        '''
        This function checks the delay between the dds channels and the fft.
        It returns the difference.
        Call loadDdsShift with the difference
        
        OUTPUTS:
            ddsShift - # clock cycles delay between dds and fft channels
        '''
        self.fpga.write_int(self.params['start_reg'],1)     # Make sure fft is running
        self.fpga.write_int(self.params['read_dds_reg'],1)  # Make sure dds is running
        ddsShift_initial = self.fpga.read_int(self.params['ddsShift_reg'])
        #self.fpga.write_int(self.params['ddsShift_reg'],0)
        self.fpga.write_int(self.params['checkLag_reg'],0)    # make sure this starts as off
        self.fpga.write_int(self.params['checkLag_reg'],1)    # Tell firmware to grab dds ch and fft ch at same exact time. Stores them in ladDds_reg and lagData_reg
        self.fpga.write_int(self.params['checkLag_reg'],0)    # turn this off for next time
        data_ch = self.fpga.read_int(self.params['lagData_reg'])
        dds_ch = self.fpga.read_int(self.params['lagDds_reg'])
        #self.fpga.write_int(self.params['ddsShift_reg'],ddsShift_initial)   # load what we had in there before
                   
        ddsShift = (ddsShift_initial + dds_ch - data_ch +1) % self.params['nChannelsPerStream']    # have to add 1 here if we use the np.roll in the writeQDR() function
        #ddsShift = (ddsShift_initial + dds_ch - data_ch ) % self.params['nChannelsPerStream'] 
        
        logging.debug('current dds lag: {}'.format(ddsShift_initial))
        logging.debug('dds ch: {}'.format(dds_ch))
        logging.debug('fft ch: {}'.format(data_ch))
        
        return ddsShift
    
    def loadDdsShift(self,ddsShift=76):
        '''
        Set the delay between the dds lut and the end of the fft block (firmware dependent)
        
        INPUTS:
            ddsShift - # clock cycles
        '''
        self.fpga.write_int(self.params['ddsShift_reg'],ddsShift)
        logging.debug('dds lag: {}'.format(ddsShift))
        return ddsShift

    def loadBoardNum(self, boardNum=None):
        '''
        Loads the board number (conventionally the last 3 digits of roach IP)

        INPUTS
            boardNum - board number
        '''
        if boardNum is None:
            boardNum = int(self.ip.split('.')[3])
        self.fpga.write_int(self.params['boardNum_reg'], boardNum)

    def loadCurTimestamp(self):
        '''
        Loads current time, in seconds since Jan 1 00:00 UTC this year
        '''
        timestamp = int(time.time())
        curYr = datetime.datetime.utcnow().year
        yrStart = datetime.date(curYr, 1, 1)
        tsOffs = calendar.timegm(yrStart.timetuple())
        timestamp -= tsOffs
        self.fpga.write_int(self.params['timestamp_reg'], timestamp)
    
    def initializeV7UART(self, waitForV7Ready = True, baud_rate = None, lut_dump_buffer_size = None):
        '''
        Initializes the UART connection to the Virtex 7.  Puts the V7 in Recieve mode, sets the 
        baud rate
        Defines global variables:
            self.baud_rate - baud rate of UART connection
            self.v7_ready - 1 when v7 is ready for a command
            self.lut_dump_data_period - number of clock cycles between writes to the UART
            self.lut_dump_buffer_size - size, in bytes, of each BRAM dump
        '''
        if(baud_rate == None):
            self.baud_rate = self.params['baud_rate']
        else:
            self.baud_rate = baud_rate
        
        if(lut_dump_buffer_size == None):
            self.lut_dump_buffer_size = self.params['lut_dump_buffer_size']
        else:
            self.lut_dump_buffer_size = lut_dump_buffer_size
        
        self.lut_dump_data_period = (10*self.params['fpgaClockRate'])//self.baud_rate + 1 #10 bits per data byte
        self.v7_ready = 0
        
        self.fpga.write_int(self.params['enBRAMDump_reg'], 0)
        self.fpga.write_int(self.params['txEnUART_reg'],0)
        self.fpga.write_int('a2g_ctrl_lut_dump_data_period', self.lut_dump_data_period)
        
        self.fpga.write_int(self.params['resetUART_reg'],1)
        time.sleep(1)
        self.fpga.write_int(self.params['resetUART_reg'],0)
        
        if waitForV7Ready:
            while(not(self.v7_ready)):
                self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
        
        self.v7_ready = 0
        self.fpga.write_int(self.params['inByteUART_reg'],1) # Acknowledge that ROACH2 knows MB is ready for commands
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],1)
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],0)
    
    def initV7MB(self):
        """
        Send commands over UART to initialize V7.
        Call initializeV7UART first
        """
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(.2)
        self.v7_ready = 0
        self.sendUARTCommand(self.params['mbEnableDACs'])
        
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(.2)
        self.v7_ready = 0
        self.sendUARTCommand(self.params['mbSendLUTToDAC'])
        
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(.2)
        self.v7_ready = 0
        self.sendUARTCommand(self.params['mbInitLO'])
        
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(.2)
        self.v7_ready = 0
        self.sendUARTCommand(self.params['mbInitAtten'])

        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(.2)
        self.v7_ready = 0
        self.sendUARTCommand(self.params['mbEnFracLO'])
        
    def generateDdsTones(self, freqChannels=None, fftBinIndChannels=None, phaseList=None):
        """
        Create and interweave dds frequencies
        
        Call setLOFreq(), generateResonatorChannels(), generateFftChanSelection() first.
        
        Sets self.ddsPhaseList
        
        INPUT:
            freqChannels - Each column contains the resonantor frequencies in a single stream. The row index is the channel number. It's padded with -1's. 
                           Made by generateResonatorChannels(). If None, use self.freqChannels
            fftBinIndChannels - Same shape as freqChannels but contains the fft bin index. Made by generateFftChanSelection(). If None, use self.fftBinIndChannels
            phaseList - Same shape as freqChannels. Contains phase offsets (0 to 2Pi) for dds sampling. 
                        If None, set to self.ddsPhaseList. if self.ddsPhaseList doesn't exist then set to all zeros
        
        OUTPUT:
            dictionary with following keywords
            'iStreamList' - 2D array. Each row is an interweaved list of i values for a single stream. 
            'qStreamList' - q values
            'quantizedFreqList' - 2d array of dds frequencies. (same shape as freqChannels) Padded with self.ddsFreqPadValue
            'phaseList' - 2d array of phases for each frequency (same shape as freqChannels) Padded with 0's
        """
        #Interpret Inputs
        if freqChannels is None:
            freqChannels = self.freqChannels
        if len(np.ravel(freqChannels))>self.params['nChannels']:
            raise ValueError("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
        self.freqChannels = freqChannels
        if fftBinIndChannels is None:
            fftBinIndChannels = self.fftBinIndChannels
        if len(np.ravel(fftBinIndChannels))>self.params['nChannels']:
            raise ValueError("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
        self.fftBinIndChannels = fftBinIndChannels
        if phaseList is None:
            if hasattr(self,'ddsPhaseList'): phaseList = self.ddsPhaseList
            else: phaseList = np.zeros(np.asarray(freqChannels).shape)
        if np.asarray(phaseList).shape != np.asarray(freqChannels).shape:
            phaseList = np.zeros(np.asarray(freqChannels).shape)
    
        if not hasattr(self,'LOFreq'):
            raise ValueError("Need to set LO freq by calling setLOFreq()")
        
        logging.debug("Generating Dds Tones...")
        # quantize resonator tones to dds resolution
        # first figure out the actual frequencies being made by the DAC
        dacFreqList = freqChannels-self.LOFreq
        dacFreqList[np.where(dacFreqList<0.)] += self.params['dacSampleRate']  #For +/- freq
        dacFreqResolution = self.params['dacSampleRate']/(self.params['nDacSamplesPerCycle']*self.params['nLutRowsToUse'])
        dacQuantizedFreqList = np.round(dacFreqList/dacFreqResolution)*dacFreqResolution
        # Figure out how the dac tones end up relative to their FFT bin centers
        fftBinSpacing = self.params['dacSampleRate']/self.params['nFftBins']
        fftBinCenterFreqList = fftBinIndChannels*fftBinSpacing
        ddsFreqList = dacQuantizedFreqList - fftBinCenterFreqList
        # Quantize to DDS sample rate and make sure all freqs are positive by adding sample rate for aliasing
        ddsSampleRate = self.params['nDdsSamplesPerCycle'] * self.params['fpgaClockRate'] / self.params['nCyclesToLoopToSameChannel']
        ddsFreqList[np.where(ddsFreqList<0)]+=ddsSampleRate     # large positive frequencies are aliased back to negative freqs
        nDdsSamples = self.params['nDdsSamplesPerCycle']*self.params['nQdrRows']/self.params['nCyclesToLoopToSameChannel']
        ddsFreqResolution = 1.*ddsSampleRate/nDdsSamples
        ddsQuantizedFreqList = np.round(ddsFreqList/ddsFreqResolution)*ddsFreqResolution
        ddsQuantizedFreqList[np.where(freqChannels<0)] = self.ddsFreqPadValue     # Pad excess frequencies with -1
        self.ddsQuantizedFreqList = ddsQuantizedFreqList
        
        # For each Stream, generate tones and interweave time streams for the dds time multiplexed multiplier
        nStreams = int(self.params['nChannels']/self.params['nChannelsPerStream'])        #number of processing streams. For Gen 2 readout this should be 4
        iStreamList = []
        qStreamList = []
        for i in range(nStreams):
            # generate individual tone time streams
            toneParams={
                'freqList': ddsQuantizedFreqList[:,i][np.where(dacQuantizedFreqList[:,i]>0)],
                'nSamples': nDdsSamples,
                'sampleRate': ddsSampleRate,
                'amplitudeList': None,  #defaults to 1
                'phaseList': phaseList[:,i][np.where(dacQuantizedFreqList[:,i]>0)]}
            toneDict = self.generateTones(**toneParams)
            
            #scale amplitude to number of bits in memory and round
            nBitsPerSampleComponent = self.params['nBitsPerDdsSamplePair']/2
            maxValue = int(np.round(2**(nBitsPerSampleComponent - 1)-1))       # 1 bit for sign
            iValList = np.array(np.round(toneDict['I']*maxValue),dtype=np.int)
            qValList = np.array(np.round(toneDict['Q']*maxValue),dtype=np.int)
            
            
            #interweave the values such that we have two samples from freq 0 (row 0), two samples from freq 1, ... to freq 256. Then have the next two samples from freq 1 ...
            freqPad = np.zeros((self.params['nChannelsPerStream'] - len(toneDict['quantizedFreqList']),nDdsSamples),dtype=np.int)
            #First pad with missing resonators
            if len(iValList) >0:
                iValList = np.append(iValList,freqPad,0)    
                qValList = np.append(qValList,freqPad,0)
            else: #if no resonators in stream then everything is 0's
                iValList = freqPad
                qValList = freqPad
            iValList = np.reshape(iValList,(self.params['nChannelsPerStream'],-1,self.params['nDdsSamplesPerCycle']))
            qValList = np.reshape(qValList,(self.params['nChannelsPerStream'],-1,self.params['nDdsSamplesPerCycle']))
            iValList = np.swapaxes(iValList,0,1)
            qValList = np.swapaxes(qValList,0,1)
            iValues = iValList.flatten('C')
            qValues = qValList.flatten('C')
            
            # put into list
            iStreamList.append(iValues)
            qStreamList.append(qValues)
            phaseList[:len(toneDict['phaseList']),i] = toneDict['phaseList']    # We need this if we let self.generateTones() choose random phases
        
        self.ddsPhaseList = phaseList
        self.ddsIStreamsList = iStreamList
        self.ddsQStreamsList = qStreamList
        
        logging.debug('\tDDS freqs: {}'.format(self.ddsQuantizedFreqList))
        for i in range(nStreams):
            logging.debug('\tStream {} I vals: {}'.format(i, self.ddsIStreamsList[i]))
            logging.debug('\tStream {} Q vals: {}'.format(i, self.ddsQStreamsList[i]))
        logging.debug('...Done!')
        
        return {'iStreamList':iStreamList,
                'qStreamList':qStreamList,
                'quantizedFreqList':ddsQuantizedFreqList,
                'phaseList':phaseList}
    
    
    def loadDdsLUT(self, ddsToneDict=None):
        '''
        Load dds tones to LUT in Roach2 memory
        
        INPUTS:
            ddsToneDict - from generateDdsTones()
                dictionary with following keywords
                'iStreamList' - 2D array. Each row is an interweaved list of i values for a single stream. Columns are different streams.
                'qStreamList' - q values
        OUTPUTS:
            allMemVals - memory values written to QDR
        '''
        if ddsToneDict is None:
            try:
                ddsToneDict={'iStreamList':self.ddsIStreamsList,'qStreamList':self.ddsQStreamsList}
            except AttributeError:
                logging.error("Need to run generateDdsTones() first!")
                raise
        
        logging.debug("Loading DDS LUT...")
        
        self.fpga.write_int(self.params['read_dds_reg'],0) #do not read from qdr while writing
        memNames = self.params['ddsMemName_regs']
        allMemVals=[]
        for iMem in range(len(memNames)):
            iVals,qVals = ddsToneDict['iStreamList'][iMem],ddsToneDict['qStreamList'][iMem]
            formatWaveparams={'iVals':iVals,
                              'qVals':qVals,
                              'nBitsPerSamplePair':self.params['nBitsPerDdsSamplePair'],
                              'nSamplesPerCycle':self.params['nDdsSamplesPerCycle'],
                              'nMems':1,
                              'nBitsPerMemRow':self.params['nBytesPerQdrSample']*8,
                              'earlierSampleIsMsb':True}
            memVals = self.formatWaveForMem(**formatWaveparams)
            allMemVals.append(memVals)
            logging.debug("\twriting QDR for Stream {}".format(iMem))
            writeQDRparams={'memName':memNames[iMem],
                            'valuesToWrite':memVals[:,0],
                            'start':0,
                            'bQdrFlip':True,
                            'nQdrRows':self.params['nQdrRows']}
            self.writeQdr(**writeQDRparams)
            time.sleep(.1)
            
        self.fpga.write_int(self.params['read_dds_reg'],1)
        
        logging.debug("...Done!")
        
        return allMemVals
    
    def writeBram(self, memName, valuesToWrite, start=0,nBytesPerSample=4):
        """
        format values and write them to bram
        
        """
        if nBytesPerSample == 4:
            formatChar = 'L'
        elif nBytesPerSample == 8:
            formatChar = 'Q'
        memValues = np.array(valuesToWrite,dtype=np.uint64) #cast signed values
        nValues = len(valuesToWrite)
        toWriteStr = struct.pack('>{}{}'.format(nValues,formatChar),*memValues)
        self.fpga.blindwrite(memName,toWriteStr,start)
        
    def writeQdr(self, memName, valuesToWrite, start=0, bQdrFlip=True, nQdrRows=2**20):
        """
        format and write 64 bit values to qdr
        
        NOTE: If you see an error that looks like: WARNING:casperfpga.katcp_fpga:Could not send message '?write qdr0_memory 0 \\0\\0\\0\\0\\0 .....
              This may be because the string you are writing is larger than the socket's write buffer size.
              You can fix this by adding a monkey patch in casperfpga/casperfpga/katcp_fpga.py 
                if hasattr(katcp.CallbackClient, 'MAX_WRITE_BUFFER_SIZE'):
                    setattr(katcp.CallbackClient, 'MAX_WRITE_BUFFER_SIZE', katcp.CallbackClient.MAX_WRITE_BUFFER_SIZE * 10)
              Then reinstalling the casperfpga code: python casperfpga/setup.py install
        
        INPUTS:
        """
        nBytesPerSample = 8
        formatChar = 'Q'
        memValues = np.array(valuesToWrite,dtype=np.uint64) #cast signed values
        nValues = len(valuesToWrite)
        if bQdrFlip: #For some reason, on Roach2 with the current qdr calibration, the 64 bit word seen in firmware
            #has the first and second 32 bit chunks swapped compared to the 64 bit word sent by katcp, so to accommodate
            #we swap those chunks here, so they will be in the right order in firmware
            mask32 = int('1'*32,2)
            memValues = (memValues >> 32)+((memValues & mask32) << 32)
            #Unfortunately, with the current qdr calibration, the addresses in katcp and firmware are shifted (rolled) relative to each other
            #so to compensate we roll the values to write here
            memValues = np.roll(memValues,-1)
        toWriteStr = struct.pack('>{}{}'.format(nValues,formatChar),*memValues)
        self.fpga.blindwrite(memName,toWriteStr,start)
    
    def formatWaveForMem(self, iVals, qVals, nBitsPerSamplePair=32, nSamplesPerCycle=4096, nMems=3, nBitsPerMemRow=64, earlierSampleIsMsb=False):
        """
        put together IQ values from tones to be loaded to a firmware memory LUT
        
        INPUTS:
            iVals - time series of I values
            qVals - 
            
        """
        nBitsPerSampleComponent = nBitsPerSamplePair / 2
        #I vals and Q vals are 12 bits, combine them into 24 bit vals
        iqVals = (iVals << nBitsPerSampleComponent) + qVals
        iqRows = np.reshape(iqVals,(-1,nSamplesPerCycle))
        #we need to set dtype to object to use python's native long type
        colBitShifts = nBitsPerSamplePair*(np.arange(nSamplesPerCycle,dtype=object))
        if earlierSampleIsMsb:
            #reverse order so earlier (more left) columns are shifted to more significant bits
            colBitShifts = colBitShifts[::-1]
        
        iqRowVals = np.sum(iqRows<<colBitShifts,axis=1) #shift each col by specified amount, and sum each row
        #Now we have 2**20 row values, each is 192 bits and contain 8 IQ pairs 
        #next we divide these 192 bit rows into three 64-bit qdr rows

        #Mem0 has the most significant bits
        memRowBitmask = int('1'*nBitsPerMemRow,2)
        memMaskShifts = nBitsPerMemRow*np.arange(nMems,dtype=object)[::-1]
        #now do bitwise_and each value with the mask, and shift back down
        memRowVals = (iqRowVals[:,np.newaxis] >> memMaskShifts) & memRowBitmask

        #now each column contains the 64-bit qdr values to be sent to a particular qdr
        return memRowVals
    
    def loadDacLUT(self, combDict=None):
        """
        Sends frequency comb to V7 over UART, where it is loaded 
        into a lookup table
        
        Call generateDacComb() first
        
        INPUTS:
            combDict - return value from generateDacComb(). If None, it trys to gather information from attributes
        """
        if combDict is None:
            try:
                combDict = {'I':np.real(self.dacFreqComb).astype(np.int), 'Q':np.imag(self.dacFreqComb).astype(np.int), 'quantizedFreqList':self.dacQuantizedFreqList}
            except AttributeError:
                logging.error("Run generateDacComb() first!")
                raise

        #Format comb for onboard memory
        #Interweave I and Q arrays
        memVals = np.empty(combDict['I'].size + combDict['Q'].size)
        memVals[0::2]=combDict['Q']
        memVals[1::2]=combDict['I']
        
        if self.debug:
            np.savetxt(self.params['debugDir']+'dacFreqs.txt', combDict['quantizedFreqList']/10**6., fmt='%3.11f', header="Array of DAC frequencies [MHz]")
        
        #Write data to LUTs
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])

        if(self.v7_ready == self.params['v7Err']):
            logging.warning('MicroBlaze did not properly execute last command.  Proceed with caution...')
            
        self.v7_ready = 0
        self.fpga.write_int(self.params['inByteUART_reg'],self.params['mbRecvDACLUT'])
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],1)
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],0)
        time.sleep(0.01)
        self.fpga.write_int(self.params['enBRAMDump_reg'],1)

                
        num_lut_dumps = int(math.ceil(len(memVals)*2/self.lut_dump_buffer_size)) #Each value in memVals is 2 bytes
        logging.debug('num lut dumps {}'.format(num_lut_dumps))
        
        sending_data = 1 #indicates that ROACH2 is still sending LUT
               
        for i in range(num_lut_dumps):
            if(len(memVals)>self.lut_dump_buffer_size/2*(i+1)):
                iqList = memVals[self.lut_dump_buffer_size/2*i:self.lut_dump_buffer_size/2*(i+1)]
            else:
                iqList = memVals[self.lut_dump_buffer_size/2*i:len(memVals)]
            
            iqList = iqList.astype(np.int16)
            toWriteStr = struct.pack('<{}{}'.format(len(iqList), 'h'), *iqList)
            logging.debug('bram dump # {}\r'.format(i))
            sys.stdout.flush()
            while(sending_data):
                sending_data = self.fpga.read_int(self.params['lutDumpBusy_reg'])
            self.fpga.blindwrite(self.params['lutBramAddr_reg'],toWriteStr,0)
            time.sleep(0.01)
            self.fpga.write_int(self.params['lutBufferSize_reg'],len(toWriteStr))
            time.sleep(0.01)
            
            while(not(self.v7_ready)):
                self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])

            if(self.v7_ready != self.params['v7LUTReady']):
                raise Exception('Microblaze not ready to recieve LUT!')

            self.fpga.write_int(self.params['txEnUART_reg'],1)
            time.sleep(0.05)
            self.fpga.write_int(self.params['txEnUART_reg'],0)
            sending_data = 1
            self.v7_ready = 0
            
        self.fpga.write_int(self.params['enBRAMDump_reg'],0)
        
        
    
    def setLOFreq(self,LOFreq):
        """ 
        Sets the attribute LOFreq (in Hz)
        """
        self.LOFreq = LOFreq
    
    def loadLOFreq(self,LOFreq=None):
        """
        Send LO frequency to V7 over UART.
        Must initialize LO first.
        
        INPUTS:
            LOFreq - LO frequency in MHz
        
        Sends LO freq one byte at a time, LSB first
           sends integer bytes first, then fractional
        """
        if LOFreq is None:
            try:
                LOFreq = self.LOFreq/1e6 #IF board uses MHz
            except AttributeError:
                logging.error("Run setLOFreq() first!")
                raise        
       
        loFreqInt = int(LOFreq)
        loFreqFrac = LOFreq - loFreqInt
        
        # Put V7 into LO recv mode
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])

        self.v7_ready = 0
        self.fpga.write_int(self.params['inByteUART_reg'],self.params['mbRecvLO'])
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],1)
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],0)        
        
        for i in range(2):
            transferByte = (loFreqInt>>(i*8))&255 #takes an 8-bit "slice" of loFreqInt
            
            while(not(self.v7_ready)):
                self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])

            if(self.v7_ready == self.params['v7Err']):
                raise Exception('MicroBlaze errored out.  Try reinitializing LO.')

            self.v7_ready = 0
            self.fpga.write_int(self.params['inByteUART_reg'],transferByte)
            time.sleep(0.01)
            self.fpga.write_int(self.params['txEnUART_reg'],1)
            time.sleep(0.001)
            self.fpga.write_int(self.params['txEnUART_reg'],0)
        
        loFreqFrac = int(loFreqFrac*(2**16))
        
        # same as transfer of int bytes
        for i in range(2):
            transferByte = (loFreqFrac>>(i*8))&255
            
            while(not(self.v7_ready)):
                self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
                time.sleep(0.01)
            
            if(self.v7_ready == self.params['v7Err']):
                raise Exception('MicroBlaze errored out.  Try reinitializing LO.')

            self.v7_ready = 0
            self.fpga.write_int(self.params['inByteUART_reg'],transferByte)
            time.sleep(0.01)
            self.fpga.write_int(self.params['txEnUART_reg'],1)
            time.sleep(0.001)
            self.fpga.write_int(self.params['txEnUART_reg'],0)
    
        while(not(self.v7_ready)):      # Wait for V7 to say it's done setting LO
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(0.01)

        if(self.v7_ready == self.params['v7Err']):
            raise Exception('MicroBlaze failed to set LO!')

    def loadLOFreqDebug(self,LOFreq=None,regList=None):
        """
        Send LO frequency to V7 over UART.
        Must initialize LO first.
        
        INPUTS:
            LOFreq - LO frequency in MHz
            regList - 7 element list of SPI programming regs, before freq info is added
        
        Sends LO freq one byte at a time, LSB first
           sends integer bytes first, then fractional
        """
        if LOFreq is None:
            try:
                LOFreq = self.LOFreq/1e6 #IF board uses MHz
            except AttributeError:
                logging.error("Run setLOFreq() first!")
                raise        
       
        if not len(regList)==7:
            raise Exception('regList must have 7 register values')
        loFreqInt = int(LOFreq)
        loFreqFrac = LOFreq - loFreqInt
        
        # Put V7 into LO recv mode
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])

        self.v7_ready = 0
        self.fpga.write_int(self.params['inByteUART_reg'],self.params['mbRecvLODebug'])
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],1)
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],0)        

        for regVal in regList:
            for j in range(4):
                transferByte = (regVal>>(j*8))&255
                while(not(self.v7_ready)):
                    self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
                    time.sleep(0.01)
                self.v7_ready = 0
                self.sendUARTCommand(transferByte)
        
        for i in range(2):
            transferByte = (loFreqInt>>(i*8))&255 #takes an 8-bit "slice" of loFreqInt
            
            while(not(self.v7_ready)):
                self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])

            if(self.v7_ready == self.params['v7Err']):
                raise Exception('MicroBlaze errored out.  Try reinitializing LO.')

            self.v7_ready = 0
            self.fpga.write_int(self.params['inByteUART_reg'],transferByte)
            time.sleep(0.01)
            self.fpga.write_int(self.params['txEnUART_reg'],1)
            time.sleep(0.001)
            self.fpga.write_int(self.params['txEnUART_reg'],0)
        
        loFreqFrac = int(loFreqFrac*(2**16))
        
        # same as transfer of int bytes
        for i in range(2):
            transferByte = (loFreqFrac>>(i*8))&255
            
            while(not(self.v7_ready)):
                self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
                time.sleep(0.01)
            
            if(self.v7_ready == self.params['v7Err']):
                raise Exception('MicroBlaze errored out.  Try reinitializing LO.')

            self.v7_ready = 0
            self.fpga.write_int(self.params['inByteUART_reg'],transferByte)
            time.sleep(0.01)
            self.fpga.write_int(self.params['txEnUART_reg'],1)
            time.sleep(0.001)
            self.fpga.write_int(self.params['txEnUART_reg'],0)
    
        while(not(self.v7_ready)):      # Wait for V7 to say it's done setting LO
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(0.01)

        if(self.v7_ready == self.params['v7Err']):
            raise Exception('MicroBlaze failed to set LO!')

    def setAdcScale(self, scale=.25):
        """
        Change the scale factor applied to adc data values before 
        sending to fft, to hopefully avoid overflowing the fft.  
        There are 4 bits in the scale with 4 bits after the binary point 
        (as of darkquad17_2016_Jul_17_2216).
        INPUTS:
            scale - scale factor applied to all ADC values.  Between 0 and 0.9375, in increments of 0.0625
        """
        scaleInt = scale*(2**self.params['adcScaleBinPt'])
        scaleInt = int(scaleInt)
        logging.error('setting adc scale to {}'.format((scaleInt / 2.**self.params['adcScaleBinPt'])))
        self.fpga.write_int(self.params['adcScale_reg'],scaleInt)

    
    def changeAtten(self, attenID, attenVal):
        """
        Change the attenuation on IF Board attenuators
        Must initialize attenuator SPI connection first
        INPUTS:
            attenID 
                1 - RF Upconverter path
                2 - RF Upconverter path
                3 - RF Downconverter path
            attenVal - attenuation between 0 and 31.75 dB. Must be multiple of 0.25 dB
        """
        if attenVal > 31.75 or attenVal<0:
            raise ValueError("Attenuation must be between 0 and 31.75")
        
        attenVal = int(np.round(attenVal*4)) #attenVal register holds value 4x(attenuation)
        
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(0.01)
            
        self.v7_ready = 0
        self.sendUARTCommand(self.params['mbChangeAtten'])
        
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(0.01)
            
        self.v7_ready = 0
        self.sendUARTCommand(attenID)
        
        while(not(self.v7_ready)):
            self.v7_ready = self.fpga.read_int(self.params['v7Ready_reg'])
            time.sleep(0.01)

        self.v7_ready = 0
        self.sendUARTCommand(attenVal)
        
    def snapZdok(self,nRolls=0):
        snapshotNames = self.fpga.snapshots.names()

        self.fpga.write_int('adc_in_trig',0)
        for name in snapshotNames:
            self.fpga.snapshots[name].arm(man_valid=False,man_trig=False)

        time.sleep(.1)
        self.fpga.write_int('adc_in_trig',1)
        time.sleep(.1) #wait for other trigger conditions to be met, and fill buffers
        self.fpga.write_int('adc_in_trig',0)
        
        adcData0 = self.fpga.snapshots['adc_in_snp_cal0_ss'].read(timeout=5,arm=False)['data']
        adcData1 = self.fpga.snapshots['adc_in_snp_cal1_ss'].read(timeout=5,arm=False)['data']
        adcData2 = self.fpga.snapshots['adc_in_snp_cal2_ss'].read(timeout=5,arm=False)['data']
        adcData3 = self.fpga.snapshots['adc_in_snp_cal3_ss'].read(timeout=5,arm=False)['data']
        bus0 = np.array([adcData0['data_i0'],adcData0['data_i1'],adcData1['data_i2'],adcData1['data_i3']]).flatten('F')
        bus1 = np.array([adcData2['data_i4'],adcData2['data_i5'],adcData3['data_i6'],adcData3['data_i7']]).flatten('F')
        bus2 = np.array([adcData0['data_q0'],adcData0['data_q1'],adcData1['data_q2'],adcData1['data_q3']]).flatten('F')
        bus3 = np.array([adcData2['data_q4'],adcData2['data_q5'],adcData3['data_q6'],adcData3['data_q7']]).flatten('F')

        adcData = dict()
        adcData.update(adcData0)
        adcData.update(adcData1)
        adcData.update(adcData2)
        adcData.update(adcData3)
        iDataKeys = ['data_i0','data_i1','data_i2','data_i3','data_i4','data_i5','data_i6','data_i7']
        iDataKeys = np.roll(iDataKeys,nRolls)
        #collate
        iValList = np.array([adcData[key] for key in iDataKeys])
        iVals = iValList.flatten('F')
        qDataKeys = ['data_q0','data_q1','data_q2','data_q3','data_q4','data_q5','data_q6','data_q7']
        qDataKeys = np.roll(qDataKeys,nRolls)
        #collate
        qValList = np.array([adcData[key] for key in qDataKeys])
        qVals = qValList.flatten('F')

        return {'bus0':bus0,'bus1':bus1,'bus2':bus2,'bus3':bus3,'adcData':adcData,'iVals':iVals,'qVals':qVals}
        
    def loadDelayLut(self, delayLut):  
        nLoadDlyRegBits = 6
        notLoadVal = int('1'*nLoadDlyRegBits,2) #when load_dly is this val, no bit delays are loaded
        self.fpga.write_int('adc_in_load_dly',notLoadVal)
        for iRow,(bit,delay) in enumerate(delayLut):
            self.fpga.write_int('adc_in_dly_val',delay)
            self.fpga.write_int('adc_in_load_dly',bit)
            time.sleep(.01)
            self.fpga.write_int('adc_in_load_dly',notLoadVal)
            
    def loadFullDelayCal(self):
        delayLut0 = zip(np.arange(0,12),np.ones(12)*14)
        delayLut1 = zip(np.arange(14,26),np.ones(12)*18)
        delayLut2 = zip(np.arange(28,40),np.ones(12)*14)
        delayLut3 = zip(np.arange(42,54),np.ones(12)*13)
        self.loadDelayLut(delayLut0)
        self.loadDelayLut(delayLut1)
        self.loadDelayLut(delayLut2)
        self.loadDelayLut(delayLut3)            
    
    def generateDacComb(self, freqList=None, resAttenList=None, globalDacAtten = 0, phaseList=None, iqRatioList=None, iqPhaseOffsList=None):
        """
        Creates DAC frequency comb by adding many complex frequencies together with specified amplitudes and phases.
        
        The resAttenList holds the absolute attenuation for each resonantor signal coming out of the DAC. Zero attenuation means that the tone amplitude is set to the full dynamic range of the DAC and the DAC attenuator(s) are set to 0. Thus, all values in resAttenList must be larger than globalDacAtten. If you decrease the globalDacAtten, the amplitude in the DAC LUT decreases so that the total attenuation of the signal is the same. 
        
        Note: Usually the attenuation values are integer dB values but technically the DAC attenuators can be set to every 1/4 dB and the amplitude in the DAC LUT can have arbitrary attenuation (quantized by number of bits).
        
        INPUTS:
            freqList - list of all resonator frequencies. If None, use self.freqList
            resAttenList - list of absolute attenuation values (dB) for each resonator. If None, use 20's
            globalDacAtten - global attenuation for entire DAC. Sum of the two DAC attenuaters on IF board
            dacPhaseList - list of phases for each complex signal. If None, generates random phases. Old phaseList is under self.dacPhaseList
            
        OUTPUTS:
            dictionary with keywords
            I - I(t) values for frequency comb [signed 32-bit integers]
            Q - Q(t)
            quantizedFreqList - list of frequencies after digitial quantiziation
        """
        # Interpret Inputs
        if freqList is None:
            freqList=self.freqList
        if len(freqList)>self.params['nChannels']:
            warnings.warn("Too many freqs provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
            freqList = freqList[:self.params['nChannels']]
        freqList = np.ravel(freqList).flatten()
        if resAttenList is None:
            try: resAttenList = self.attenList
            except AttributeError: 
                warnings.warn("Individual resonator attenuations assumed to be 20")
                resAttenList=np.zeros(len(freqList))+20
        if len(resAttenList)>self.params['nChannels']:
            warnings.warn("Too many attenuations provided. Can only accommodate "+str(self.params['nChannels'])+" resonators")
            resAttenList = resAttenList[:self.params['nChannels']]
        resAttenList = np.ravel(resAttenList).flatten()
        if len(freqList) != len(resAttenList):
            raise ValueError("Need exactly one attenuation value for each resonant frequency!")
        if (phaseList is not None) and len(freqList) != len(phaseList):
            raise ValueError("Need exactly one phase value for each resonant frequency!")
        if np.any(resAttenList < globalDacAtten):
            raise ValueError("Impossible to attain desired resonator attenuations! Decrease the global DAC attenuation.")
        self.attenList = resAttenList
        self.freqList = freqList
        
        logging.error('Generating DAC comb...')
        
        # Calculate relative amplitudes for DAC LUT
        nBitsPerSampleComponent = self.params['nBitsPerSamplePair']/2
        maxAmp = int(np.round(2**(nBitsPerSampleComponent - 1)-1))       # 1 bit for sign
        amplitudeList = maxAmp*10**(-(resAttenList - globalDacAtten)/20.)
        
        # Calculate nSamples and sampleRate
        nSamples = self.params['nDacSamplesPerCycle']*self.params['nLutRowsToUse']
        sampleRate = self.params['dacSampleRate']
        
        # Calculate resonator frequencies for DAC
        if not hasattr(self,'LOFreq'):
            raise ValueError("Need to set LO freq by calling setLOFreq()")
        dacFreqList = self.freqList-self.LOFreq
        dacFreqList[np.where(dacFreqList<0.)] += self.params['dacSampleRate']  #For +/- freq
        
        # Generate and add up individual tone time series.
        if iqRatioList is None:
            if hasattr(self, 'iqRatioList'):
                iqRatioList = self.iqRatioList
        if iqPhaseOffsList is None:
            if hasattr(self, 'iqPhaseOffsList'):
                iqPhaseOffsList = self.iqPhaseOffsList

        toneDict = self.generateTones(dacFreqList, nSamples, sampleRate, amplitudeList, phaseList, iqRatioList, iqPhaseOffsList)
        self.dacQuantizedFreqList = toneDict['quantizedFreqList']
        self.dacPhaseList = toneDict['phaseList']
        iValues = np.array(np.round(np.sum(toneDict['I'],axis=0)),dtype=np.int)
        qValues = np.array(np.round(np.sum(toneDict['Q'],axis=0)),dtype=np.int)
        self.dacFreqComb = iValues + 1j*qValues
        
        # check that we are utilizing the dynamic range of the DAC correctly
        highestVal = np.max((np.abs(iValues).max(),np.abs(qValues).max()))
        expectedHighestVal_sig = scipy.special.erfinv((len(iValues)-0.1)/len(iValues))*np.sqrt(2.)   # 10% of the time there should be a point this many sigmas higher than average

        logging.debug('\tUsing {} percent of DAC dynamic range'
                      .format(1.0*highestVal/maxAmp*100))
        logging.debug('\thighest: {} out of {}'.format(highestVal, maxAmp))
        logging.debug('\tsigma_I: {} sigma_Q: {}'+format(iValues, qValues))
        logging.debug('\tLargest val_I: {} sigma. Largest val_Q: {} sigma.'
                      .format(1.0*np.abs(iValues).max()/np.std(iValues),
                              1.0*np.abs(qValues).max()/np.std(qValues)))
        logging.debug('\tExpected val: {} sigmas'
                      .format(expectedHighestVal_sig))
            
        if highestVal > expectedHighestVal_sig*np.max((np.std(iValues),np.std(qValues))):
            warnings.warn("The freq comb's relative phases may have added up sub-optimally. You should calculate new random phases")
        if highestVal > maxAmp:
            dBexcess = int(np.ceil(20.*np.log10(1.0*highestVal/maxAmp)))
            raise ValueError("Not enough dynamic range in DAC! Try decreasing the global DAC Attenuator by "+str(dBexcess)+' dB')
        elif 1.0*maxAmp/highestVal > 10**((1)/20.):
            # all amplitudes in DAC less than 1 dB below max allowed by dynamic range
            warnings.warn("DAC Dynamic range not fully utilized. Increase global attenuation by: "+str(int(np.floor(20.*np.log10(1.0*maxAmp/highestVal))))+' dB')
        
        logging.debug('...Done!')

        return {'I':iValues,'Q':qValues,'quantizedFreqList':self.dacQuantizedFreqList}
        
    
    def generateTones(self, freqList, nSamples, sampleRate, amplitudeList, phaseList, iqRatioList=None, iqPhaseOffsList=None):
        """
        Generate a list of complex signals with amplitudes and phases specified and frequencies quantized
        
        INPUTS:
            freqList - list of resonator frequencies
            nSamples - Number of time samples
            sampleRate - Used to quantize the frequencies
            amplitudeList - list of amplitudes. If None, use 1.
            phaseList - list of phases. If None, use random phase
        
        OUTPUTS:
            dictionary with keywords
            I - each element is a list of I(t) values for specific freq
            Q - Q(t)
            quantizedFreqList - list of frequencies after digitial quantiziation
            phaseList - list of phases for each frequency
        """
        if amplitudeList is None:
            amplitudeList = np.asarray([1.]*len(freqList))
        if phaseList is None:
            phaseList = np.random.uniform(0,2.*np.pi,len(freqList))
        if iqRatioList is None:
            iqRatioList = np.ones(len(freqList))
        if iqPhaseOffsList is None:
            iqPhaseOffsList = np.zeros(len(freqList))
        if len(freqList) != len(amplitudeList) or len(freqList) != len(phaseList) or len(freqList) != len(iqRatioList) or len(freqList) != len(iqPhaseOffsList):
            raise ValueError("Need exactly one phase, amplitude, and IQ correction value for each resonant frequency!")
        
        # Quantize the frequencies to their closest digital value
        freqResolution = sampleRate/nSamples
        quantizedFreqList = np.round(freqList/freqResolution)*freqResolution
        iqPhaseOffsRadList = np.pi/180*iqPhaseOffsList
        
        # generate each signal
        iValList = []
        qValList = []
        dt = 1. / sampleRate
        t = dt*np.arange(nSamples)
        for i in range(len(quantizedFreqList)):
            phi = 2.*np.pi*quantizedFreqList[i]*t
            expValues = amplitudeList[i]*np.exp(1.j*(phi+phaseList[i]))
            iScale = np.sqrt(2)*iqRatioList[i]/np.sqrt(1+iqRatioList[i]**2)
            qScale = np.sqrt(2)/np.sqrt(1+iqRatioList[i]**2)
            iValList.append(iScale*(np.cos(iqPhaseOffsRadList[i])*np.real(expValues)+np.sin(iqPhaseOffsRadList[i])*np.imag(expValues)))
            qValList.append(qScale*np.imag(expValues))
        
        return {'I':np.asarray(iValList),'Q':np.asarray(qValList),'quantizedFreqList':quantizedFreqList,'phaseList':phaseList}
        
        
    def generate_resonator_channels(self, freqList, order='F'):
        """
        Algorithm for deciding which resonator frequencies are assigned to which stream and channel number.
        This is used to define the dds LUTs and calculate the fftBin index for each freq to set the appropriate chan_sel block
        
        Try to evenly distribute the given frequencies into each stream (unless you use order='stream')
        
        INPUTS:
            freqList - list of resonator frequencies (Assumed sorted but technically doesn't need to be)
                       If it's not unique there could be problems later on. 
            order - 'F' places sequential frequencies into a single stream but forces an even distribution among streams
                    'C' places sequential frequencies into the same channel number but forces an even distribution among streams
                    'stream' sequentially fills stream 0 first, then stream 1, etc...
        OUTPUTS:
            self.freqChannels - Each column contains the resonantor frequencies in a single stream. 
                                The row index is the channel number.
                                It's padded with -1's. 
        """
        #Interpret inputs...
        if order not in ['F','C','A','stream']:  #if invalid, grab default value
            args,__,__,defaults = inspect.getargspec(Roach2Controls.generateResonatorChannels)
            order = defaults[args.index('order')-len(args)]
            logging.debug("Invalid 'order' parameter for generateResonatorChannels(). Changed to default: {}".format(order))
        if len(np.array(freqList))>self.params['nChannels']:
            logging.warning("Too many freqs provided. Can only accommodate {} resonators".format(self.params['nChannels']))
            freqList = freqList[:self.params['nChannels']]
        self.freqList = np.ravel(freqList)
        if len(np.unique(self.freqList)) != len(self.freqList):
            raise ValueError
        self.freqChannels = self.freqList
        logging.debug('Generating Resonator Channels...')
        
        #Pad with freq = -1 so that freqChannels's length is a multiple of nStreams
        nStreams = int(self.params['nChannels']/self.params['nChannelsPerStream'])        #number of processing streams. For Gen 2 readout this should be 4
        padValue = self.freqPadValue   #pad with freq=-1
        if order == 'F':
            padNum = (nStreams - (len(self.freqChannels) % nStreams))%nStreams  # number of empty elements to pad
            for i in range(padNum):
                ind = len(self.freqChannels)-i*np.ceil(len(self.freqChannels)*1.0/nStreams)
                self.freqChannels=np.insert(self.freqChannels,int(ind),padValue)
        elif order == 'C' or order == 'A':
            padNum = (nStreams - (len(self.freqChannels) % nStreams))%nStreams  # number of empty elements to pad
            self.freqChannels = np.append(self.freqChannels, [padValue]*(padNum))
        elif order == 'stream':
            nFreqs = len(self.freqList)
            if nFreqs < self.params['nChannelsPerStream']:
                padNum = nFreqs * (nStreams-1)
            else:
                padNum = self.params['nChannels'] - nFreqs
            self.freqChannels=np.append(self.freqChannels,[padValue]*padNum)
            order = 'F'
            
        #Split up to assign channel numbers
        self.freqChannels = np.reshape(self.freqChannels,(-1,nStreams),order)
        
        # Make arrays useful for conversions between channel nomenclature.
        # Probably can and should replace this with hardware map machinery,
        # but I've made a temporary cleanup of this for intermediate testing
        # purposes. I didn't change the logic significantly here, which is
        # admittedly pretty crappy.
        self.frequency_index = np.arange(len(self.freqList))
        self.channel_num = np.zeros(len(self.freqList))
        self.stream_num = np.zeros(len(self.freqList))
        for jfreq in range(len(self.freqList)):
            self.channel_num[jfreq], self.stream_num[jfreq] = \
                np.where(self.freqChannels == self.freqList[jfreq])
            
            logging.debug('\tFreq Channels: {}'.format(self.freqChannels))
            logging.debug('...Done!')

        return self.freqChannels
        
    def get_stream_num(self, frequency=None, frequency_index=None):
        '''
        Gets a stream number from a frequency value or frequency index
        (accepts only one of these arguments, not both).

        Parameters
        ----------
        frequency : float
            Frequency in Hz
        frequency_index : int
            Index of frequency in sorted list of all frequencies

        Returns
        -------
        stream_num : int
            Stream number of channel
        '''
        if (frequency == None and frequency_index == None) or \
           (frequency != None and frequency_index != None):
            raise ValueError('Exactly one keyword argument is required: frequency or frequency_index')

        if frequency != None:
            return self.stream_num[self.freqList == frequency]
        elif frequency_index != None:
            return self.stream_num[self.frequency_index == frequency_index]

    def get_channel_num(self, frequency=None, frequency_index=None):
        '''
        Gets a channel number within a stream from a frequency value or 
        frequency index (accepts only one of these arguments, not both).

        Parameters
        ----------
        frequency : float
            Frequency in Hz
        frequency_index : int
            Index of frequency in sorted list of all frequencies

        Returns
        -------
        channel_num : int
            Channel number within stream
        '''
        if (frequency == None and frequency_index == None) or \
           (frequency != None and frequency_index != None):
            raise ValueError('Exactly one keyword argument is required: frequency or frequency_index')

        if frequency != None:
            return self.channel_num[self.freqList == frequency]
        elif frequency_index != None:
            return self.channel_num[self.frequency_index == frequency_index]

    def get_frequency(self, stream_num, channel_num):
        '''
        Get a frequency index from a stream and channel number.

        Parameters
        ----------
        stream_num : int
            Stream number
        channel_num : int
            Channel number

        Returns
        -------
        frequency_index : int
            Frequency index
        '''
        return self.frequency_index[(self.stream_num == stream_num) &
                                    (self.channel_num == channel_num)]

    def get_frequency_index(self, stream_num, channel_num):
        '''
        Get a frequency value in Hz from a stream and channel number.

        Parameters
        ----------
        stream_num : int
            Stream number
        channel_num : int
            Channel number

        Returns
        -------
        frequency : float
            Frequency
        '''
        return self.freqList[(self.stream_num == stream_num) &
                             (self.channel_num == channel_num)]
        
    def generateFftChanSelection(self,freqChannels=None):
        '''
        This calculates the fftBin index for each resonant frequency and arranges them by stream and channel.
        Used by channel selector block
        Call setLOFreq() and generateResonatorChannels() first.
        
        INPUTS (optional):
            freqChannels - 2D array of frequencies where each column is the a stream and each row is a channel. If freqChannels isn't given then try to grab it from attribute. 
        
        OUTPUTS:
            self.fftBinIndChannels - Array with each column containing the fftbin index of a single stream. The row index is the channel number
            
        '''
        if freqChannels is None:
            try:
                freqChannels = self.freqChannels
            except AttributeError:
                logging.error("Run generateResonatorChannels() first!")
                raise
        freqChannels = np.asarray(freqChannels)
        logging.debug("Finding FFT Bins...")
        
        #The frequencies seen by the fft block are actually from the DAC, up/down converted by the IF board, and then digitized by the ADC
        dacFreqChannels = (freqChannels-self.LOFreq)
        dacFreqChannels[np.where(dacFreqChannels<0)]+=self.params['dacSampleRate']
        freqResolution = self.params['dacSampleRate']/(self.params['nDacSamplesPerCycle']*self.params['nLutRowsToUse'])
        dacQuantizedFreqChannels = np.round(dacFreqChannels/freqResolution)*freqResolution
        
        #calculate fftbin index for each freq
        binSpacing = self.params['dacSampleRate']/self.params['nFftBins']
        genBinIndex = dacQuantizedFreqChannels/binSpacing
        self.fftBinIndChannels = np.round(genBinIndex)
        self.fftBinIndChannels[np.where(freqChannels<0)]=self.fftBinPadValue      # empty channels have freq=-1. Assign this to fftBin=0
        
        self.fftBinIndChannels = self.fftBinIndChannels.astype(np.int)
        
        logging.debug('\tfft bin indices: ',self.fftBinIndChannels)
        logging.debug('...Done!')
        
        return self.fftBinIndChannels

        
    def loadChanSelection(self,fftBinIndChannels=None):
        """
        Loads fftBin indices to all channels (in each stream), to configure chan_sel block in firmware on self.fpga
        Call generateFftChanSelection() first

        
        INPUTS (optional):
            fftBinIndChannels - Array with each column containing the fftbin index of a single stream. The row is the channel number
        """
        if fftBinIndChannels is None:
            try:
                fftBinIndChannels = self.fftBinIndChannels
            except AttributeError:
                logging.error("Run generateFftChanSelection() first!")
                raise
        
        nStreams = self.params['nChannels']/self.params['nChannelsPerStream']
        logging.debug('Configuring chan_sel block...\n\tCh: Stream {}'.format(range(len(fftBinIndChannels[0]))))
        for row in range(self.params['nChannelsPerStream']):
            try:
                fftBinInds = fftBinIndChannels[row]
            except IndexError:
                fftBinInds = np.asarray([self.fftBinPadValue]*nStreams)
            self.loadSingleChanSelection(selBinNums=fftBinInds,chanNum=row)
        
        logging.debug('...Done!')
        if self.debug:
            np.savetxt(self.params['debugDir']+'freqChannels.txt',
                       self.freqChannels/10**9.,fmt='%2.25f',
                       header="2D Array of MKID frequencies [GHz]. \nEach column represents a stream and each row is a channel")
            np.savetxt(self.params['debugDir']+'fftBinIndChannels.txt',
                       self.fftBinIndChannels,fmt='%8i',
                       header="2D Array of fftBin Indices. \nEach column represents a stream and each row is a channel")
        
    def loadSingleChanSelection(self,selBinNums,chanNum=0):
        """
        Assigns bin numbers to a single channel (in each stream), to configure chan_sel block
        Used by loadChanSelection()

        INPUTS:
            selBinNums: array of bin numbers (for each stream) to be assigned to chanNum (4 element int array for Gen 2 firmware)
            chanNum: the channel number to be assigned
        """
        nStreams = int(self.params['nChannels']/self.params['nChannelsPerStream'])        #number of processing streams. For Gen 2 readout this should be 4
        if selBinNums is None or len(selBinNums) != nStreams:
            raise TypeError,'selBinNums must have number of elements matching number of streams in firmware'
        
        self.fpga.write_int(self.params['chanSelLoad_reg'],0) #set to zero so nothing loads while we set other registers.

        #assign the bin number to be loaded to each stream
        for i in range(nStreams):
            self.fpga.write_int(self.params['chanSel_regs'][i],selBinNums[i])
        time.sleep(.001)
        
        #in the register chan_sel_load, the lsb initiates the loading of the above bin numbers into memory
        #the 8 bits above the lsb indicate which channel is being loaded (for all streams)
        loadVal = (chanNum << 1) + 1
        self.fpga.write_int(self.params['chanSelLoad_reg'],loadVal)
        time.sleep(.001) #give it a chance to load

        self.fpga.write_int(self.params['chanSelLoad_reg'],0) #stop loading
        
        logging.debug('\t{}: {}\r'.format(chanNum, selBinNums))
        sys.stdout.flush()

    def setMaxCountRate(self, cpsLimit = 2500):
        for reg in self.params['captureCPSlim_regs']:
            try:
                self.fpga.write_int(reg,cpsLimit)
            except:
                logging.error("Couldn't write to {}".format(reg))
    
    def setThreshByFreqChannel(self,thresholdRad = -.1, freqChannel=0):
        """
        Overloads setThresh but using channel as indexed by the freqList
        
        INPUTS:
            thresholdRad: The threshold in radians.  The phase must drop below this value to trigger a photon event
            freqChannel - channel as indexed by the freqList
        """
        ch = self.get_channel_num(frequency_index = freqChannel)
        stream = self.get_stream_num(frequency_index = freqChannel)
        self.thresholdList[ch+(stream<<8)] = thresholdRad
        self.setThresh(thresholdRad = thresholdRad,ch=int(ch), stream=int(stream))
    
    def setThresh(self,thresholdRad = -.1,ch=0,stream=0):
        """Sets the phase threshold and baseline filter for photon pulse detection triggers in each channel

        INPUTS:
            thresholdRad: The threshold in radians.  The phase must drop below this value to trigger a photon event
            ch - the channel number in the stream
            stream - the stream number
        """
        
        #format it as a fix16_13 to be placed in a register
        thresholdRad = max(thresholdRad, -3.8) #overflow if threshold is less than -4
        binThreshold = castBin(thresholdRad,quantization='Round',nBits=16,binaryPoint=13,format='uint')
        sampleRate = 1.e6

        #for the baseline, we apply a second order state variable low pass filter to the phase
        #See http://www.earlevel.com/main/2003/03/02/the-digital-state-variable-filter/
        #The filter takes two parameters based on the desired Q factor and cutoff frequency
        criticalFreq = 200 #Hz
        Q=.7
        baseKf=2*np.sin(np.pi*criticalFreq/sampleRate)
        baseKq=1./Q

        #format these paramters as fix18_16 values to be loaded to registers
        binBaseKf=castBin(baseKf,quantization='Round',nBits=18,binaryPoint=16,format='uint')
        binBaseKq=castBin(baseKq,quantization='Round',nBits=18,binaryPoint=16,format='uint')
        logging.debug('threshold {} {}'.format(thresholdRad,binThreshold))
        logging.debug('Kf: {}'.format(baseKf,binBaseKf))
        logging.debug('Kq: {}'.format(baseKq,binBaseKq))
        #load the values in
        self.fpga.write_int(self.params['captureBasekf_regs'][stream],binBaseKf)
        self.fpga.write_int(self.params['captureBasekq_regs'][stream],binBaseKq)

        self.fpga.write_int(self.params['captureThreshold_regs'][stream],binThreshold)
        self.fpga.write_int(self.params['captureLoadThreshold_regs'][stream],1+(ch<<1))
        time.sleep(.003)    # Each snapshot should take 2 msec of phase data
        self.fpga.write_int(self.params['captureLoadThreshold_regs'][stream],0)

    def loadFIRCoeffs(self,coeffFile):
        '''
        This function loads the FIR coefficients into the Firmware's phase filter for every resonator
        You can provide a filter for each resonator channel or just a single filter that's applied to each resonator
        Any channels without resonators have their filter taps set to 0
        
        If self.freqList and self.freqChannels don't exist then it loads FIR coefficients into every channel
        Be careful, depending on how you set up the channel selection block you might assign the wrong filters to the resonators
        (see self.generateResonatorChannels() for making self.freqList, self.freqChannels)
        
        INPUTS:
            coeffFile - path to plain text file that contains a 2d array
                        The i'th column corresponds to the i'th resonator in the freqList
                        If there is only one column then use it for every resonator in the freqList
                        The j'th row is the filter's coefficient for the j'th tap 
        '''
        # Decide which channels to write FIRs to
        try:
            freqChans = range(len(self.freqList))
            channels = self.get_channel_num(frequency_index = freqChannel)
            streams = self.get_stream_num(frequency_index = freqChannel)
        except AttributeError:      # If we haven't loaded in frequencies yet then load FIRs into all channels
            freqChans = range(self.params['nChannels'])
            streams = np.repeat(range(self.params['nChannels']/self.params['nChannelsPerStream']), self.params['nChannelsPerStream'])
            channels = np.tile(range(self.params['nChannelsPerStream']), self.params['nChannels']/self.params['nChannelsPerStream'])
        
        # grab FIR coeff from file
        firCoeffs = np.transpose(np.loadtxt(coeffFile))
        if firCoeffs.ndim==1: firCoeffs = np.tile(firCoeffs, (len(freqChans),1))    # if using the same filter for every pixel
        else: firCoeffs = np.transpose(firCoeffs)
        firBinPt=self.params['firBinPt']
        firInts=np.asarray(firCoeffs*(2**firBinPt),dtype=np.int32)
        zeroWriteStr = struct.pack('>{}{}'.format(len(firInts[0]),'l'), *np.zeros(len(firInts[0])))     # write zeros for channels without resonators
        
        # loop through and write FIRs to firmware
        nStreams = self.params['nChannels']/self.params['nChannelsPerStream']
        for stream in range(nStreams):
            try:
                self.fpga.write_int(self.params['firLoadChan_regs'][stream],0)  #just double check that this is at 0
                ch_inds = np.where(streams==stream)     # indices in list of resonator channels that correspond to this stream
                ch_stream = np.atleast_1d(channels)[ch_inds]           # list of the stream channels with this stream
                ch_freqs = np.atleast_1d(freqChans)[ch_inds]           # list of freq channels with this stream
                for ch in range(self.params['nChannelsPerStream']):
                    if ch in np.atleast_1d(ch_stream):
                        ch_freq = int(np.atleast_1d(ch_freqs)[np.where(np.atleast_1d(ch_stream)==ch)])     # The freq channel of the resonator corresponding to ch/stream
                        toWriteStr = struct.pack('>{}{}'.format(len(firInts[ch_freq]),'l'), *firInts[ch_freq])
                        logging.info('{} Hz: ch/stream: {}/{}'+str(stream).format(ch_freq, ch, stream))
                    else:
                        toWriteStr=zeroWriteStr
                    self.fpga.blindwrite(self.params['firTapsMem_regs'][stream], toWriteStr,0)
                    time.sleep(.001)           # 1ms is more than enough. Should only take nTaps/fpgaClockRate seconds to load in
                    loadVal = (1<<8)+ch        # first bit indicates we will write, next 8 bits is the chan number for the stream
                    self.fpga.write_int(self.params['firLoadChan_regs'][stream],loadVal)
                    time.sleep(.001)
                    self.fpga.write_int(self.params['firLoadChan_regs'][stream],0)
            except:
                logging.error('Failed to write FIRs on stream {}'.format(stream))     # Often times test firmware only implements stream 0
                if stream==0: raise

    def takePhaseSnapshotOfFreqChannel(self, freqChan):
        '''
        This function overloads takePhaseSnapshot
        
        INPUTS:
            freqChan - the resonator channel as indexed in the freqList
        '''
        ch = self.get_channel_num(frequency_index = freqChannel)
        stream = self.get_stream_num(frequency_index = freqChannel)
        selChanIndex = (int(stream)<<8) + int(ch)
        logging.info("Taking phase snap from ch/stream: {} / {} selChanIndex: {}"
                     .format(ch, stream, selChanIndex))
        return self.takePhaseSnapshot(selChanIndex)
    
    def takePhaseSnapshot(self, selChanIndex):
        """
        Takes phase data using snapshot block

        INPUTS:
            selChanIndex: channel to take data from

        OUTPUTS:
            snapDict with keywords:
            phase - list of phases in radians
            trig - list of booleans indicating the firmware triggered
            time - Number of seconds for each phase point starting at 0 (1 point every 256 clock cycles)
        """
        self.fpga.write_int(self.params['phaseSnpCh_reg'],selChanIndex)
        self.fpga.snapshots[self.params['phaseSnapshot']].arm(man_valid=False)
        time.sleep(.001)
        self.fpga.write_int(self.params['phaseSnpTrig_reg'],1)#trigger snapshots
        time.sleep(.001) #wait for other trigger conditions to be met
        self.fpga.write_int(self.params['phaseSnpTrig_reg'],0)#release trigger    
                       
        snapDict = self.fpga.snapshots[self.params['phaseSnapshot']].read(timeout=5,arm=False,man_valid=False)['data'] 
        trig = np.roll(snapDict['trig'],-2) #there is an extra 2 cycle delay in firmware between we_out and phase
        snapDict['trig']=trig
        dt=self.params['nChannelsPerStream']/self.params['fpgaClockRate']
        snapDict['time']=dt*np.arange(len(trig))
        snapDict['swTrig']=self.calcSWTriggers(selChanIndex, snapDict['phase'])
        return snapDict

    def calcSWTriggers(self, selChanIndex, phaseData, nNegDerivChecks=10, nNegDerivLeniance=1, nPosDerivChecks=2, deadtime=10):
        """
        Software derived trigger on photons in phase snapshots. 
        Trigger conditions (should match firmware):
            -nNegDeriveChecks-nNegDeriveLeniance/nNegDeriveChecks negative slopes, 
                followed by nPosDeriveChecks positive slopes
            -<threshold (b/c pulses are negative)
        
        INPUTS:
            selChanIndex: channel to take data from
            phaseData: array containing phase from snapshot, in radians
            nNegDeriveChecks, nNegDerivLeniance, nPosDeriveChecks are explained above
        
        OUTPUTS:
            trigPos: array of size len(phaseData), w/ a 1
            at photon trigger positions
        """
        phaseDeriv = np.diff(phaseData)
        isNegDeriv = phaseDeriv <= 0
        isPosDeriv = phaseDeriv > 0
        phaseData = phaseData - np.median(phaseData) #baseline subtract data
        threshCond = phaseData < self.thresholdList[selChanIndex]
        threshCond = np.delete(threshCond,np.arange(0,nNegDerivChecks)) #align this condition with derivatives
        
        negDerivChecksSum = np.zeros(len(isNegDeriv[0:-nNegDerivChecks-1]))
        for i in range(nNegDerivChecks):
            negDerivChecksSum += isNegDeriv[i:i-nNegDerivChecks-1]
        negDerivCond = (negDerivChecksSum >= (nNegDerivChecks - nNegDerivLeniance))
        
        posDerivChecksSum = np.zeros(len(isPosDeriv[0:-nPosDerivChecks-1]))
        for i in range(nPosDerivChecks):
            posDerivChecksSum += isPosDeriv[i:i-nPosDerivChecks-1]
        posDerivCond = posDerivChecksSum >= nPosDerivChecks
        posDerivCond = np.delete(posDerivCond, np.arange(0,nNegDerivChecks)) #align with other conditions
        
        trigger = np.logical_and(threshCond[0:len(negDerivCond)], negDerivCond)
        trigger = np.logical_and(trigger[0:len(posDerivCond)], posDerivCond[0:len(trigger)])
        trigger = np.pad(trigger, (nNegDerivChecks,0), 'constant') 
        trigger = np.pad(trigger, (0, len(phaseData)-len(trigger)), 'constant')

        #apply deadtime
        trigSum = np.zeros(len(trigger))
        for i in range(deadtime):
            trigSum[deadtime:] += trigger[deadtime-i:len(trigger)-i]
        violatesDt = np.where(trigSum >= 2)[0]
        trigger[violatesDt] = 0
        
        return trigger


    def startPhaseStream(self,selChanIndex=0, pktsPerFrame=100, fabric_port=50000, hostIP='10.0.0.50'):
        """initiates streaming of phase timestream (after prog_fir) to the 1Gbit ethernet

        INPUTS:
            selChanIndex: stream/channel. The first two bits indicate the stream, last 8 bits for the channel
            pktsPerFrame: number of 8 byte photon words per ethernet frame
            fabric_port
            destIPID: destination IP is 10.0.0.destIPID
            
        """
        
        dest_ip = binascii.hexlify(socket.inet_aton(hostIP))
        dest_ip = int(dest_ip,16)
        self.fpga.write_int(self.params['destIP_reg'],dest_ip)
        self.fpga.write_int(self.params['phasePort_reg'],fabric_port)
        self.fpga.write_int(self.params['wordsPerFrame_reg'],pktsPerFrame)
        
        #reset the core to make sure it's in a clean state
        self.fpga.write_int(self.params['photonCapStart_reg'],0)    #make sure we're not streaming photons
        self.fpga.write_int(self.params['phaseDumpEn_reg'],0)       #can't send packets when resetting
        self.fpga.write_int(self.params['gbe64Rst_reg'],1)
        time.sleep(.1)
        self.fpga.write_int(self.params['gbe64Rst_reg'],0)

        #choose what channel to stream
        self.fpga.write_int(self.params['phaseDumpChanSel_reg'],selChanIndex)
        #turn it on
        self.fpga.write_int(self.params['phaseDumpEn_reg'],1)
    
    def stopStream(self):
        """stops streaming of phase timestream (after prog_fir) to the 1Gbit ethernet

        """
        self.fpga.write_int(self.params['phaseDumpEn_reg'],0)
    
    def recvPhaseStream(self, channel=0, duration=60, pktsPerFrame=100, host = '10.0.0.50', port = 50000):
        """
        Recieves phase timestream data over ethernet, writes it to a file.  Must call
        startPhaseStream first to initiate phase stream.
        
        The data is saved in self.phaseTimeStreamData
        
        INPUTS:
            channel - stream/channel. The first two bits indicate the stream, last 8 bits for the channel
                      channel = 0 means ch 0 on stream 0. channel = 256 means ch 0 on stream 1, etc...
            duration - duration (in seconds) of phase stream
            host - IP address of computer receiving packets 
                (represented as a string)
            port
        
        OUTPUTS:
            self.phaseTimeStreamData - phase packet data. See parsePhaseStream()
        """

        logging.debug('host {}'.format(host))
        logging.debug('port {}'.format(port))
        logging.debug('duration {}'.format(duration))

        # create dgram udp socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            logging.error('Failed to create socket')
            raise

        # Bind socket to local host and port
        try:
            sock.bind((host, port))
        except socket.error , msg:
            logging.error('Bind failed. Error Code : {} Message {}'.format(msg[0], msg[1]))
            sock.close()
            raise
        
        bufferSize = int(8*pktsPerFrame) #Each photon word is 8 bytes
        iFrame = 0
        nFramesLost = 0
        lastPack = -1
        expectedPackDiff = -1
        frameData = ''

        startTime = time.time()
        try:
            while (time.time()-startTime) < duration:
                frame = sock.recvfrom(bufferSize)
                frameData += frame[0]
                iFrame += 1

        except KeyboardInterrupt:       
            logging.error('Exiting')
            sock.close()
            self.phaseTimeStreamData = frameData
            return

        sock.close()
        self.phaseTimeStreamData = frameData
        return self.phaseTimeStreamData
    
    def takePhaseStreamDataOfFreqChannel(self, freqChan=0, duration=2, pktsPerFrame=100, fabric_port=50000, hostIP='10.0.0.50'):
        """
        This function overloads takePhaseStreamData() but uses the channel index corresponding to the freqlist instead of a ch/stream index
        
        INPUTS:
            freqChan - which channel to collect phase on. freqChan corresponds to the resonator index in freqList
            duration - duration (in seconds) of stream
            pktsPerFrame - number of 8 byte photon words per ethernet frame
            fabric_port -
            destIPID - IP address of computer receiving stream
                
        OUTPUTS:
            phases - a list of phases in radians
        """
        ch = self.get_channel_num(frequency_index = freqChan)
        stream = self.get_stream_num(frequency_index = freqChan)
        selChanIndex = (int(stream) << 8) + int(ch)
        
        return self.takePhaseStreamData(selChanIndex, duration, pktsPerFrame, fabric_port, hostIP)
        
    def takePhaseStreamData(self, selChanIndex=0, duration=2, pktsPerFrame=100, fabric_port=50000, hostIP='10.0.0.50'):
        """
        Takes phase timestream data from the specified channel for the specified amount of time
        Gets one phase value for every nChannelsPerStream/fpgaClockRate seconds
        
        INPUTS:
            selChanIndex - stream/channel. The first two bits indicate the stream, last 8 bits for the channel
                           channel = 0 means ch 0 on stream 0. channel = 256 means ch 0 on stream 1, etc...
            duration - duration (in seconds) of stream
            pktsPerFrame - number of 8 byte photon words per ethernet frame
            fabric_port - 
            destIPID - IP address of computer receiving stream
                
        OUTPUTS:
            phases - a list of phases in radians
        """
        
        self.startPhaseStream(selChanIndex, pktsPerFrame, fabric_port, hostIP )
        logging.debug("Collecting phase time stream...")
        phaseTimeStreamData=self.recvPhaseStream(selChanIndex, duration, pktsPerFrame, hostIP, fabric_port)
        self.stopStream()
        logging.debug("...Done!")
        
        return self.parsePhaseStream(phaseTimeStreamData,pktsPerFrame)
        
    
    def parsePhaseStream(self, phaseTimeStreamData=None, pktsPerFrame=100):
        """
        This function parses the packet data from recvPhaseStream()
        
        INPUTS:
            phaseTimeStreamData - phase packet data from recvPhaseStream()
            pktsPerFrame - number of 8 byte photon words per ethernet frame
        
        OUTPUTS:
            phases - a list of phases in radians
        """
        data = phaseTimeStreamData
        if phaseTimeStreamData is None:
            data = self.phaseTimeStreamData
        
        nBytes = len(data)
        nWords = nBytes/8 #64 bit words
        #break into 64 bit words
        words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)
        
        #remove headers
        headerFirstByte = 0xff
        firstBytes = words >> (64-8)
        headerIdx = np.where(firstBytes == headerFirstByte)[0]
        words = np.delete(words,headerIdx)
        
        nBitsPerPhase = 12
        binPtPhase = 9
        nPhasesPerWord = 5
        #to parse out the 5 12-bit values, we'll shift down the bits we don't want for each value, then apply a bitmask to take out
        #bits higher than the 12 we want
        #The least significant bits in the word should be the earliest phase, so the first column should have zero bitshift
        bitmask = int('1'*nBitsPerPhase,2)
        bitshifts = nBitsPerPhase*np.arange(nPhasesPerWord)

        #add an axis so we can broadcast
        #and shift away the bits we don't keep for each row
        phases = (words[:,np.newaxis]) >> bitshifts
        phases = phases & bitmask

        #now we have a nWords x nPhasesPerWord array

        #flatten so that the phases are in order
        phases = phases.flatten(order='C')
        phases = np.array(phases,dtype=np.uint64)
        signBits = np.array(phases / (2**(nBitsPerPhase-1)),dtype=np.bool)

        #check the sign bits to see what values should be negative
        #for the ones that should be negative undo the 2's complement, and flip the sign
        phases[signBits] = ((~phases[signBits]) & bitmask)+1
        phases = np.array(phases,dtype=np.double)
        phases[signBits] = -phases[signBits]
        #now shift down to the binary point
        phases = phases / 2**binPtPhase

        return phases

    
    def performIQSweep(self,startLOFreq,stopLOFreq,stepLOFreq):
        """
        Performs a sweep over the LO frequency.  Records 
        one IQ point per channel per freqeuency; stores in
        self.iqSweepData
        
        makes iqData - 4xn array - each row has I and Q values for a single stream.  For each row:
                     256 I points + 256 Q points for LO step 0, then 256 I points + 256 Q points for LO step 1, etc..
                     Shape = [4, (nChannelsPerStream+nChannelsPerStream) * nLOsteps]
                     Get one per stream (4 streams for all thousand resonators)
                     Formatted using formatIQSweepData then stored in self.iqSweepData
        
        The logic is as follows:
            set LO
            Arm the snapshot block
            trigger write enable - This grabs the first set of 256 I, 256 Q points
            disable the writeEnable
            set LO
            trigger write enable - This grabs the a second set of 256 I, 256 Q points
            read snapshot block - read 1024 points from mem
            disable writeEnable

        INPUTS:
            startLOFreq - starting sweep frequency [MHz]
            stopLOFreq - final sweep frequency [MHz]
            stepLOFreq - frequency sweep step size [MHz]
        OUTPUTS:
            iqSweepData - Dictionary with following keywords
                          I - 2D array with shape = [nFreqs, nLOsteps]
                          Q - 2D array with shape = [nFreqs, nLOsteps]
                          freqOffsets - list of offsets from LO in Hz. shape = [nLOsteps]
            
        """
        LOFreqs = np.arange(startLOFreq, stopLOFreq, stepLOFreq)
        nStreams = self.params['nChannels']/self.params['nChannelsPerStream']
        iqData = np.empty([nStreams,0])
        
        # The magic number 4 below is the number of IQ points per read
        # We get two I points and two Q points every read
        iqPt = np.empty([nStreams,self.params['nChannelsPerStream']*4])
        self.fpga.write_int(self.params['iqSnpStart_reg'],0)        
        
        for i in range(len(LOFreqs)):
            logging.debug('Sweeping LO {} MHz'.format(LOFreqs[i]))
            self.loadLOFreq(LOFreqs[i])
            if(i%2==0):
                for stream in range(nStreams):
                    self.fpga.snapshots[self.params['iqSnp_regs'][stream]].arm(man_valid = False, man_trig = False) 
            self.fpga.write_int(self.params['iqSnpStart_reg'],1)
            time.sleep(0.001)    # takes nChannelsPerStream/fpgaClockRate seconds to load all the values
            if(i%2==1):
                for stream in range(nStreams):
                    iqPt[stream]=self.fpga.snapshots[self.params['iqSnp_regs'][stream]].read(timeout = 10, arm = False)['data']['iq']
                iqData = np.append(iqData, iqPt,1)
            self.fpga.write_int(self.params['iqSnpStart_reg'],0)
        
        # if odd number of LO steps then we still need to read out half of the last buffer
        if len(LOFreqs) % 2 == 1:
            self.fpga.write_int(self.params['iqSnpStart_reg'],1)
            time.sleep(0.001)
            for stream in range(nStreams):
                iqPt[stream]=self.fpga.snapshots[self.params['iqSnp_regs'][stream]].read(timeout = 10, arm = False)['data']['iq']
            iqData = np.append(iqData, iqPt[:,:self.params['nChannelsPerStream']*2],1)
            self.fpga.write_int(self.params['iqSnpStart_reg'],0)
        
        self.loadLOFreq()   # reloads initial lo freq
        self.iqSweepData = self.formatIQSweepData(iqData)
        self.iqSweepData['freqOffsets'] = np.copy((LOFreqs*10**6. - self.LOFreq))   # [Hz]
        return self.iqSweepData
    
    def formatIQSweepData(self, iqDataStreams):
        """
        Reshapes the iqdata into a usable format
        Need to put the data in the same order as the freqList that was loaded in
        
        If we haven't loaded in a freqList then the order is channels 0..256 in stream 0, then stream 1, etc..
        
        INPUTS:
            iqDataStreams - 2D array with following shape:
                            [nStreams, (nChannelsPerStream+nChannelsPerStream) * nSteps]
        OUTPUTS:
            iqSweepData - Dictionary with following keywords
                          I - 2D array with shape = [nFreqs, nSteps]
                          Q - 2D array with shape = [nFreqs, nSteps]
        """
        # Only return IQ data for channels/streams with resonators associated with them
        try:
            freqChans = range(len(self.freqList))
            channels = self.get_channel_num(frequency_index = freqChannel)
            streams = self.get_stream_num(frequency_index = freqChannel)
        except AttributeError:      # If we haven't loaded in frequencies yet then grab all channels
            freqChans = range(self.params['nChannels'])
            streams = np.repeat(range(self.params['nChannels']/self.params['nChannelsPerStream']), self.params['nChannelsPerStream'])
            channels = np.tile(range(self.params['nChannelsPerStream']), self.params['nChannels']/self.params['nChannelsPerStream'])
        
        I_list = []
        Q_list = []
        for i in range(len(freqChans)):
            ch, stream = np.atleast_1d(channels)[i], np.atleast_1d(streams)[i]
            I = iqDataStreams[stream, ch :: self.params['nChannelsPerStream']*2]
            Q = iqDataStreams[stream, ch+self.params['nChannelsPerStream'] :: self.params['nChannelsPerStream']*2]
            I_list.append(I.flatten())
            Q_list.append(Q.flatten())
        
        return {'I':I_list, 'Q':Q_list}
                
        I_list2 = I_list[-2:] + I_list[:-2]
        Q_list2 = Q_list[-2:] + Q_list[:-2]
        
        return {'I':I_list2, 'Q':Q_list2}
    

    def takeAvgIQData(self,numPts=100):
        """
        Take IQ data with the LO fixed (at self.LOFreq)

        INPUTS:
            numPts - Number of IQ points to take 
        
        OUTPUTS:
            iqSweepData - Dictionary with following keywords
                          I - 2D array with shape = [nFreqs, nLOsteps]
                          Q - 2D array with shape = [nFreqs, nLOsteps]
        """
        
        counter = np.arange(numPts)
        nStreams = self.params['nChannels']/self.params['nChannelsPerStream']
        iqData = np.empty([nStreams,0])
        self.fpga.write_int(self.params['iqSnpStart_reg'],0)        
        iqPt = np.empty([nStreams,self.params['nChannelsPerStream']*4])
        
        for i in counter:
            logging.debug('IQ point # {}'.format(i))
            if(i%2==0):
                for stream in range(nStreams):
                    self.fpga.snapshots[self.params['iqSnp_regs'][stream]].arm(man_valid = False, man_trig = False) 
            self.fpga.write_int(self.params['iqSnpStart_reg'],1)
            time.sleep(0.001)    # takes nChannelsPerStream/fpgaClockRate seconds to load all the values
            if(i%2==1):
                for stream in range(nStreams):
                    iqPt[stream]=self.fpga.snapshots[self.params['iqSnp_regs'][stream]].read(timeout = 10, arm = False)['data']['iq']
                iqData = np.append(iqData, iqPt,1)
            self.fpga.write_int(self.params['iqSnpStart_reg'],0)
        
        # if odd number of steps then we still need to read out half of the last buffer
        if len(counter) % 2 == 1:
            self.fpga.write_int(self.params['iqSnpStart_reg'],1)
            time.sleep(0.001)
            for stream in range(nStreams):
                iqPt[stream]=self.fpga.snapshots[self.params['iqSnp_regs'][stream]].read(timeout = 10, arm = False)['data']['iq']
            iqData = np.append(iqData, iqPt[:,:self.params['nChannelsPerStream']*2],1)
            self.fpga.write_int(self.params['iqSnpStart_reg'],0)

        self.iqToneData = self.formatIQSweepData(iqData)
        return self.iqToneData
    
    def loadIQcenters(self, centers):
        """
        Load IQ centers in firmware registers
        
        INPUTS:
            centers - 2d array of centers.
                      First column is I centers, second is Q centers. 
                      Rows correspond to resonators in the same order as the freqlist
                      shape: [nFreqs, 2]
        """
        channels = self.get_channel_num(frequency_index = freqChannel)
        streams = self.get_stream_num(frequency_index = freqChannel)
        
        for i in range(len(centers)):
            ch = channels[i]
            stream=streams[i]
            I_c = int(centers[i][0]/2**3)
            Q_c = int(centers[i][1]/2**3)
            
            center = (I_c<<16) + (Q_c<<0)   # 32 bit number - 16bit I + 16bit Q
            self.fpga.write_int(self.params['iqCenter_regs'][stream], center)
            self.fpga.write_int(self.params['iqLoadCenter_regs'][stream], (ch<<1)+(1<<0))
            self.fpga.write_int(self.params['iqLoadCenter_regs'][stream], 0)
    
    def sendUARTCommand(self, inByte):
        """
        Sends a single byte to V7 over UART
        Doesn't wait for a v7_ready signal
        Inputs:
            inByte - byte to send over UART
        """
        self.fpga.write_int(self.params['inByteUART_reg'],inByte)
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],1)
        time.sleep(0.01)
        self.fpga.write_int(self.params['txEnUART_reg'],0)        
        
    
    def loadFreq(self):
        '''                   
        Loads the resonator freq files (and attenuations, resIDs)                                                                                          divides the resonators into streams
        '''
        try:
            logging.info('old Freq: {}'.format(self.roachController.freqList))
        except: pass
        fn = self.config.get(self.roachString,'freqfile')
        fn2=fn.rsplit('.',1)[0]+'_NEW.'+ fn.rsplit('.',1)[1]         # Check if ps_freq#_NEW.txt exists
        logging.info("RoachStateMachine.loadFreq:  fn, fn2 = {}, {}".format(fn,fn2))
        if os.path.isfile(fn2):
            fn=fn2
            logging.info('Loading freqs from {}'.format(fn))

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
        argsSorted = np.argsort(freqs)  # sort them by frequency
        freqs = freqs[argsSorted]
        resIDs = resIDs[argsSorted]
        attens = attens[argsSorted]
        phaseOffsList = iqRatioList[argsSorted]
        iqRatioList = iqRatioList[argsSorted]
        for i in range(len(freqs)):
            logging.info('{} {} {} {} {} {}'.format(i, resIDs[i], freqs[i],
                                                    attens[i], phaseOffsList[i],
                                                    iqRatioList[i]))

        self.generateResonatorChannels(freqs)
        self.attenList = attens
        self.resIDs = resIDs
        self.phaseOffsList = phaseOffsList
        self.iqRatioList = iqRatioList
        logging.info('new Freq: {}'.format(self.freqList))

        return True


    def defineRoachLUTs(self):
        '''
        Defines LO Freq but doesn't load it yet
        Defines and loads channel selection blocks
        Defines and loads DDS LUTs
        writing the QDR takes a long time! :-(
        '''
        loFreq = int(self.config.getfloat(self.roachString,'lo_freq'))
        self.setLOFreq(loFreq)
        self.generateFftChanSelection()
        ddsTones = self.generateDdsTones()
        self.loadChanSelection()
        self.loadDdsLUT()
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

        dacComb = self.generateDacComb(globalDacAtten=dacAtten)
        logging.info("Initializing ADC/DAC board communication")
        self.initializeV7UART()
        logging.info("Setting Attenuators")
        self.changeAtten(1,dacAtten1)
        self.changeAtten(2,dacAtten2)
        self.changeAtten(3,adcAtten)
        logging.info("Setting LO Freq")
        self.loadLOFreq()
        logging.info("Loading DAC LUT")
        self.loadDacLUT()
        return True

