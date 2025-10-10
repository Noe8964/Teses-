"""
ISDASProtocol.PY

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

"""


def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')

class ISDASProtocol:
    """ 
    The interface that both the client and the servers need to implement in order to use the SDAS system.
    @author bbc
    """

    SERVER_HANDLER_ID = 'SDASServer'
    def searchDeclaredEventsByUniqueID(self, uniqueID):
        """Searches declared events by the unique identifier.
        @param uniqueID the unique identifier to search.
        @return an array with all the declared events information with a description that partially or fully matches the word(s) to search.
        """
        abstract()
    def searchDeclaredEventsByName(self, search, locale=''): 
        """   Searches declared events by the name and locale.
        @param search the word(s) to search.
        @param locale the locale in the format languageCode_countryCode, for example: pt_PT
        @return an array with all the declared events information with a name that partially 
        or fully matches the word(s) to search in the requested locale.
        """
        abstract()
    def searchDeclaredEventsByDescription(self, search, locale=''): 
        """ Searches declared events by the description and locale.
         @param search the word(s) to search.
         @param locale the locale in the format languageCode_countryCode, for example: pt_PT
         @return an array with all the declared events information with a description 
         that partially or fully matches the word(s) to search in the requested locale.
        """
        abstract()
    def searchEventsByEventNumber(self, number): 
        """Searches events by the event number.
        @param number the number to search.
        @return all the events found that have this event number.
        """
        abstract()
    def searchMaxEventNumber(self, uniqueID): 
        """Searches the maximum event number for an event defined by this uniqueID.
        @param uniqueID the unique identifier of the event.
        @return the maximum event number or zero if the event isn't found.
        """
        abstract()
    def searchEventsByEventTimeWindow(self, tstart, tend):
        """Searches events whose time stamp is inside a time window.
        @param tstart from this time.
        @param tend to this time.
        @return all the events whose time stamp is superior to tstart and inferior to tend.
        """        
        abstract()
    def searchMinEventNumber(self, uniqueID): 
        """Searches the minimum event number for an event defined by this uniqueID.
        @param uniqueID the unique identifier of the event.
        @return the minimum event number or zero if the event isn't found.
        """
        abstract()
    def searchParametersByUniqueID(self, uniqueID):
        """Search parameters by the unique identifier.
        @param uniqueID the unique identifier to search.
        @return an array with all the parameters with an unique identifier 
        that partially or fully matches the uniqueID to search.
        """    
        abstract()
    def searchParametersByName(self, search, locale=''): 
        """Searches parameters by the name and locale.
        @param search the word(s) to search.
        @param locale the locale in the format languageCode_countryCode, for example:
        @return an array with all the parameters with a name that partially of fully 
        matches the word(s) to search in the requested locale.
        """    
        abstract()
    def searchParametersByDescription(self, search, locale=''): 
        """Searches parameters by the description and locale.
        @param search the word(s) to search.
        @param locale the locale in the format languageCode_countryCode, for example: pt_PT
        @return an array with all the parameters with a description that partially or fully 
        matches the word(s) to search in the requested locale.
        """
        abstract()
    def searchDataByEvent(self, eventUniqueID, eventNumber):
        """Searches data by an event.
        This method can be very time consuming. Avoid using it unless you really need it!
        @param event the event to search.
        @param eventUniqueID the event unique identifier.
        @param eventNumber the event number.
        @return an array with all the parameters unique identifiers whose data isn't null
        for this event.
        """
        abstract()
    def getData(self, parameterUniqueID,  eventUniqueID, eventNumber): 
        """Fetches a data structure from the SDAS server.
        @param parameterUniqueID the parameter unique identifier.
        @param eventUniqueID the event unique identifier.
        @param eventNumber the event number.
        @return the data structure returned by the server. 
        If no data is returned from the server (for example a bad parameter uniqueID) 
        an empty structure is returned.
        """
        abstract()
    def getMultipleData(self, parameterUniqueID,  eventUniqueID, eventNumber):
        """Fetches several data structures from the SDAS server.
        @param parameterUniqueID an array with all the parameters unique identifiers.
        @param event an array with an event for each selected parameter.
        If you wish to use the same event for all the parameters see {@link getMultipleData(java.lang.String[], IHEvent)}. 
        The size of this array should be the same of parameterUniqueID.
        @return an array with all the data for the requested parameters. 
        If data isn't available for any of the parameters an empty structure is added to the array.
        """
        abstract()
    def parameterExists(self, parameterUniqueID, eventUniqueID,  eventNumber):
        """Checks if a parameter exists for a particular event.
        @param parameterUniqueID the parameter unique identifier.
        @param eventUniqueID the event unique identifier.
        @param eventNumber the events number.
        @return true if the parameter exists, false otherwise.
        """
        abstract()
#    def (self, search): abstract()
#    """
#    """