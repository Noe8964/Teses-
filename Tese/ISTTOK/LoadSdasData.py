from sdas.core.client.SDASClient import SDASClient
from sdas.core.SDAStime import Date, Time, TimeStamp
import numpy as np
import sys
import matplotlib
import struct

def LoadSdasData(client, channelID, shotnr):
    try:
        dataStruct=client.getData(channelID, '0x0000', shotnr);
        dataArray=dataStruct[0].getData();
        len_d=len(dataArray);
        tstart = dataStruct[0].getTStart();
        tend = dataStruct[0].getTEnd();
        tbs= (tend.getTimeInMicros() - tstart.getTimeInMicros())*1.0/len_d;
        events = dataStruct[0].get('events')[0];
        tevent = TimeStamp(tstamp=events.get('tstamp'));
        delay = tstart.getTimeInMicros() - tevent.getTimeInMicros();
        timeVector = np.linspace(delay,delay+tbs*(len_d-1),len_d);
    except:
        print(f'Error while acquiring singnal {channelID} for pulse number {shotnr}, returning empty array.')
        return []
    return [dataArray, timeVector, tevent]

def LoadSdasRawData(client, channelID, shotnr):
    #try:
    dataStruct = client.getData(channelID, '0x0000', shotnr);
    rawData    = dataStruct[0].get('raw_data').data
    typeCode   = dataStruct[0].MIME_TYPES[dataStruct[0].get("mime_type")]
    d          = np.fromstring(rawData, typeCode)
    # print(d.byteswap())    
    # d = []
    # for n in range(int(len(data)/4)):
        # #d.append(int.from_bytes(data[4*n:4*(n+1)], 'little', signed=True))
        # d.append( struct.unpack( '>I', bytes([data[4*n+1], data[4*n+0], data[4*n+2], data[4*n+3]]) ) )
    dataArray=dataStruct[0].getData();
    len_d=len(dataArray);
    tstart = dataStruct[0].getTStart();
    tend = dataStruct[0].getTEnd();
    tbs= (tend.getTimeInMicros() - tstart.getTimeInMicros())*1.0/len_d;
    events = dataStruct[0].get('events')[0];
    tevent = TimeStamp(tstamp=events.get('tstamp'));
    delay = tstart.getTimeInMicros() - tevent.getTimeInMicros();
    timeVector = np.linspace(delay,delay+tbs*(len_d-1),len_d);
    #except:
    #    print(f'Error while acquiring singnal {channelID} for pulse number {shotnr}, returning empty array.')
    #    return []
    return [d.byteswap(), timeVector, tevent]

def LoadSdasTransferFunction(client, channelID, shotnr):
    try:
        dataStruct=client.getData(channelID, '0x0000', shotnr);
        tf = dataStruct[0].get('transfer_function')
    except:
        print(f'Error while acquiring singnal {channelID} for pulse number {shotnr}, returning empty array.')
        return []
    return tf

def LoadSdasDataByName(client, channelName, shotnr):
    try:
        params = client.searchParametersByName(channelName)
        idx = 0
        #In case the search returns more than one option (it is not case sensitive)
        if(len(params) > 1):
            for n in range(len(params)):
                signalName = params[n]['descriptorUID']['name']
                if(signalName == channelName):
                    idx = n
                    if('MARTE_NODE_IVO3' in params[idx]['descriptorUID']['uniqueID']):
                        break
        channelID = params[idx]['descriptorUID']['uniqueID']
        dataStruct=client.getData(channelID, '0x0000', shotnr);
        dataArray=dataStruct[0].getData();
        len_d=len(dataArray);
        tstart = dataStruct[0].getTStart();
        tend = dataStruct[0].getTEnd();
        tbs= (tend.getTimeInMicros() - tstart.getTimeInMicros())*1.0/len_d;
        events = dataStruct[0].get('events')[0];
        tevent = TimeStamp(tstamp=events.get('tstamp'));
        delay = tstart.getTimeInMicros() - tevent.getTimeInMicros();
        timeVector = np.linspace(delay,delay+tbs*(len_d-1),len_d);
    except:
        print(f'Error while acquiring singnal {channelName} for pulse number {shotnr}, returning empty array.')
        return []
    return [dataArray, timeVector, tevent]