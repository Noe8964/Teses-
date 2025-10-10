"""
SDAStime.py

Created on June 19, 2006, 14:00 

<p>Title: SDAS</p>

<p>Description: Shared Data Access System</p>

<p>Copyright: (C) Copyright 2005-2006, by Centro de Fusao Nuclear

Project Info:      http://baco.cfn.ist.utl.pt/sdas
                     http://www.cfn.ist.utl.pt

This library is free software; you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or
(at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation,
Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.

[Java is a trademark or registered trademark of Sun Microsystems, Inc.
in the United States and other countries.]
</p>

<p>Company: CFN - EURATOM/IST- http://www.cfn.ist.utl.pt</p>
@author Bernardo Carvalho
@version 1.0.0
author: $Author$
"""
# Needs Numeric or numarray or NumPy
try:
    import numpy as _Numeric
except:
    try:
        import numarray as _Numeric  #if numarray is used it is renamed Numeric
    except:
        try:
            import Numeric as _Numeric
        except:
            msg= """
            This module requires the Numeric/numarray or NumPy module,
            which could not be imported.  It probably is not installed
            (it's not part of the standard Python distribution). See the
            Numeric Python site (http://numpy.scipy.org) for information on
            downloading source or binaries."""
            raise ImportError("Numeric,numarray or NumPy not found. \n" + msg)


import socket
import sys

from sdas.core.common.ISDASProtocol import ISDASProtocol
from sdas.core.SDAStime import Date, Time, TimeStamp
try:
    from xmlrpclib import ServerProxy, Error # Python 2 library
except ImportError:
    from xmlrpc.client import ServerProxy, Error # Python 3 library
    
if sys.version_info[0] == 2:
    range = xrange # changes the behaviour of range in Python 2 to mimic xrange, like in Python 3

class Data(dict):
    """Class to convert an array of big endian bytes to a _Numeric array
    """
    try:
        MIME_TYPES={"data/float_array":_Numeric.float32,
                "data/double_array":_Numeric.float64,
                "data/short_array":_Numeric.int16,
                "data/int_array":_Numeric.int32,
                "data/long_array":_Numeric.int64}
    except:
        MIME_TYPES={"data/float_array":_Numeric.Float32,
                "data/double_array":_Numeric.Float64,
                "data/short_array":_Numeric.Int16,
                "data/int_array":_Numeric.Int32,
                "data/long_array":_Numeric.Int64}
    def __init__(self, dict=None):             
        #self.data = {}                         
        if dict is not None: self.update(dict) # copies all the keys and values from one dictionary to another
    def getData(self):
        typeCode=self.MIME_TYPES[self.get("mime_type")]
        d= _Numeric.fromstring(self.get("raw_data").data,typeCode)
        return d.byteswap()
    def getTStart(self):
        return TimeStamp(tstamp=self.get("tstart"))
    def getTEnd(self):
        return TimeStamp(tstamp=self.get("tend"))
        
    
class SDASClient(ISDASProtocol): #ISDASProtocol
    """ Creates a new instance of SDASClient.
    """
    def __init__(self, host, port=8888, transport=None, encoding=None, verbose=0, allow_none=True):
#        ISDASProtocol.__init__(host, port)
        uri='http://%s:%d'%(host,port)
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            soc.connect((host,port))
            soc.close()
            xmlrpcClient = ServerProxy(uri, transport, encoding, verbose, allow_none=True)
            self.server=xmlrpcClient.SDASServer
            self.sessionID = self.server.authenticate({})
        except Error(v):
            print("ERROR", v)
            raise IOError("unsupported XML-RPC protocol")
        except socket.gaierror:
            print("Err")
            raise IOError("unsupported XML-RPC protocol")
        #        except :
#            print " Error"
    def searchDeclaredEventsByUniqueID(self, uniqueID):
        """Searches declared events by the unique identifier.
        @param uniqueID the unique identifier to search.
        @return an array with all the declared events information with a description that partially or fully matches the word(s) to search.
        """
        return self.server.searchDeclaredEventsByUniqueID(self.sessionID, uniqueID)

    def searchDeclaredEventsByDescription(self, search, locale=''):
        return self.server.searchDeclaredEventsByDescription(self.sessionID, search, locale)
    def searchDeclaredEventsByName(self, search, locale=''): 
        """Searches declared events by the name and locale.
        @param search the word(s) to search.
        @param locale the locale in the format languageCode_countryCode, for example: pt_PT
        @return an array with all the declared events information with a name that partially 
        or fully matches the word(s) to search in the requested locale.
        """
        return self.server.searchDeclaredEventsByName(self.sessionID, search, locale)
    def searchEventsByEventNumber(self, number): 
        """Searches events by the event number.
        @param number the number to search.
        @return all the events found that have this event number.
        """
        return self.server.searchEventsByEventNumber(self.sessionID, number)
    def searchParametersByUniqueID(self, uniqueID):
        """Search parameters by the unique identifier.
        @param uniqueID the unique identifier to search.
        @return an array with all the parameters with an unique identifier 
        that partially or fully matches the uniqueID to search.
        """
        return self.server.searchParametersByUniqueID(self.sessionID, uniqueID)
    def searchParametersByName(self, search, locale=''): 
        """Searches parameters by the name and locale.
        @param search the word(s) to search.
        @param locale the locale in the format languageCode_countryCode, for example:
        @return an array with all the parameters with a name that partially of fully 
        matches the word(s) to search in the requested locale.
        """    
        return self.server.searchParametersByName(self.sessionID, search, locale)
    def searchMaxEventNumber(self, uniqueID='0x0000'):
        """Searches the maximum event number for an event defined by this uniqueID.
        @param uniqueID the unique identifier of the event.
        @return the maximum event number or zero if the event isn't found.
        """
        return self.server.searchMaxEventNumber(self.sessionID, uniqueID)
    def searchMinEventNumber(self, uniqueID='0x0000'): 
        """Searches the minimum event number for an event defined by this uniqueID.
        @param uniqueID the unique identifier of the event.
        @return the minimum event number or zero if the event isn't found.
        """
        return self.server.searchMinEventNumber(self.sessionID, uniqueID)
    def searchDataByEvent(self, eventUniqueID, eventNumber):
        """Searches data by an event.
        This method can be very time consuming. Avoid using it unless you really need it!
        @param event the event to search.
        @param eventUniqueID the event unique identifier.
        @param eventNumber the event number.
        @return an array with all the parameters unique identifiers whose data isn't null
        for this event.
        """
        return self.server.searchDataByEvent(self.sessionID, eventUniqueID, eventNumber)
    def searchEventsByEventTimeWindow(self, tstart, tend): 
        """Searches events whose time stamp is inside a time window.
        @param tstart from this time.
        @param tend to this time.
        @return all the events whose time stamp is superior to tstart and inferior to tend.
        """
        return self.server.searchEventsByEventTimeWindow(self.sessionID, tstart.getParams(), tend.getParams())
    
    def getData(self, parameterUniqueID,  eventUniqueID, eventNumber):
        """Fetches a data structure from the SDAS server.
        @param parameterUniqueID the parameter unique identifier.
        @param eventUniqueID the event unique identifier.
        @param eventNumber the event number.
        @return the data array structure returned by the server. 
        If no data is returned from the server (for example a bad parameter uniqueID) 
        an empty structure is returned.
        """
        dt=self.server.getData(self.sessionID, parameterUniqueID, eventUniqueID, eventNumber)
        return [Data(dt[i]) for i in range(len(dt))]
        
    def getFirstData(self, parameterUniqueID,  eventUniqueID, eventNumber):
        """Fetches a data structure from the SDAS server.
        @param parameterUniqueID the parameter unique identifier.
        @param eventUniqueID the event unique identifier.
        @param eventNumber the event number.
        @return the firstdata structure returned by the server. 
        If no data is returned from the server (for example a bad parameter uniqueID) 
        an empty structure is returned. This method behaves like the previous getData.
        """
        dt=self.server.getData(self.sessionID, parameterUniqueID, eventUniqueID, eventNumber)
        return Data(dt[0])
 
    def parameterExists(self, parameterUniqueID, eventUniqueID,  eventNumber):
        """Checks if a parameter exists for a particular event.
        @param parameterUniqueID the parameter unique identifier.
        @param eventUniqueID the event unique identifier.
        @param eventNumber the events number.
        @return true if the parameter exists, false otherwise.
        """
        return self.server.parameterExists(self.sessionID, parameterUniqueID, eventUniqueID, eventNumber)
    
if __name__ == "__main__":

    # simple test program (from the XML-RPC )

    # server = ServerProxy("http://localhost:8000") # local server "http://betty.userland.com"
    host='baco.ipfn.ist.utl.pt'
    port=8888
    client = SDASClient(host,port)

    #print client
    
#    try:
#        #print server.examples.getStateName(41) 'SHOT', 'pt'
#        print client.searchDeclaredEventsByDescription('S', 'pt')
#    except Error, v:
#        print "ERROR xmlrpclib", v
#    #print client.searchDeclaredEventsByName('SH')
#    print client.searchMaxEventNumber()
#    found=client.searchMinEventNumber('0x0000')
#    #found=client.searchParametersByUniqueID('II')
#    #found=client.searchDataByEvent('0x0000', 12050)
#    print client.searchParametersByName('II', 'pt')
#    found= client.searchDeclaredEventsByUniqueID('SHOT')
#    for item in found:
#        print 'item', item
    if  client.parameterExists('CENTRAL_OS9_ADC.IIG','0x0000', 11244):
        dS=client.getData('CENTRAL_OS9_ADC.IIGBT','0x0000', 11244)
        ic=dS.getData()
    #print 'dS' , dS.get("tstart")
        print(len(ic), ic[1])
        tstart = dS.getTStart()
        tend = dS.getTEnd()
        print(tstart.getTimeInSeconds())
    date_start = Date(2005, 11, 1)
    date_end = Date(2005, 11, 31)
    #t0=Time()
    tstart = TimeStamp(date_start)
    print(tstart.getTimeInMicros())
    tend = TimeStamp(date_end)
    found=client.searchEventsByEventTimeWindow(tstart, tend)
    for item in found:
        print(item)
    
