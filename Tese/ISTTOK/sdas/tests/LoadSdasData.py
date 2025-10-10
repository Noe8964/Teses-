from sdas.core.client.SDASClient import SDASClient
from sdas.core.SDAStime import Date, Time, TimeStamp
import numpy as np

def LoadSdasData(client, channelID, shotnr):
    dataStruct=client.getData(channelID,'0x0000', shotnr);
    dataArray=dataStruct[0].getData();
    len_d=len(dataArray);
    tstart = dataStruct[0].getTStart();
    tend = dataStruct[0].getTEnd();
    tbs= (tend.getTimeInMicros() - tstart.getTimeInMicros())*1.0/len_d;
    events = dataStruct[0].get('events')[0];
    tevent = TimeStamp(tstamp=events.get('tstamp'));
    delay = tstart.getTimeInMicros() - tevent.getTimeInMicros();
    timeVector = np.linspace(delay,delay+tbs*(len_d-1),len_d);
    return [dataArray, timeVector]
