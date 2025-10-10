"""
Date.py

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
#from UserDict import UserDict



class Date(dict):
    """This class wraps the xml-rpc date structure.
    @param year the year.
    @param month the month (zero base index - january is 0)
    @param day the day of the month.
    @param dict a class of type Date.
    """
    def __init__(self, year =1970, month=0, day= 1, dict=None):             
        if dict is not None: self.update(dict) # copies all the keys and values from one dictionary to another
        else:
            if (year <1970) or (year >3000):
                raise ValueError("%s is not a valid year"%year)
            if (month <0) or (month >11):
                raise ValueError("%s is not a valid month"%month)
            if (day <1) or (day >31):
                raise ValueError("%s is not a valid day"%day)
            self["year"]=year
            self["month"]=month
            self["day"]=day

#    return {'month': month, 'day': day, 'year': year}

#def Date(year =2003, month=9, day= 10 ):
#    """This function wraps the xml-rpc date structure.
#    @param year the year.
#    @param month the month (zero base index - january is 0)
#    @param day the day of the month.
#    """
#    year=int(year); month=int(month); day=int(day)
#    if (year <1970) or (year >3000):
#        raise ValueError, "%s is not a valid year"%year
#    if (month <0) or (month >11):
#        raise ValueError, "%s is not a valid month"%month
#    if (day <1) or (day >31):
#        raise ValueError, "%s is not a valid day"%day
#    return {'month': month, 'day': day, 'year': year}

class Time(dict):
    """This class wraps the xml-rpc time structure.
    @param hours the hour of the day.
    @param minutes the minutes.
    @param seconds the seconds.
    @param millis the milliseconds.
    @param micros the microseconds.
    @param nanos the nanoseconds.
    @param picos the picoseconds.
    """
    def __init__(self, hours=0,  minutes=0, seconds=0, millis=0, micros=0, nanos=0, picos=0, dict=None):             
        if dict is not None: self.update(dict) # copies all the keys and values from one dictionary to another
        else:
            hours=int(hours);  minutes=int(minutes); seconds=int(minutes); 
            millis=int(minutes); micros=int(micros); nanos=int(nanos); picos=int(picos)
            if (hours <0) or (hours >24):
                raise ValueError("%s is not a valid hours"%hours)
            if (minutes <0) or (minutes >60):
                raise ValueError("%s is not a valid minutes"%minutes)
            if (seconds <0) or (seconds >60):
                raise ValueError("%s is not a valid seconds"%seconds)
            self["hours"]=hours
            self["minutes"]=minutes
            self["seconds"]=seconds
            self["millis"]=millis
            self["micros"]=micros
            self["nanos"]=nanos
            self["picos"]=picos
    #return {'hours' :hours,  'minutes' :minutes, 'seconds' :seconds, 
    #        'millis' :millis, 'micros' :micros, 'nanos' :nanos, 'picos' : picos}
    
#def Time(hours=0,  minutes=0, seconds=0, millis=0, micros=0, nanos=0, picos=0):
#    """This function wraps the xml-rpc time structure.
#    @param hours the hour of the day.
#    @param minutes the minutes.
#    @param seconds the seconds.
#    @param millis the milliseconds.
#    @param micros the microseconds.
#    @param nanos the nanoseconds.
#    @param picos the picoseconds.
#    """
#    hours=int(hours);  minutes=int(minutes); seconds=int(minutes); 
#    millis=int(minutes); micros=int(micros); nanos=int(nanos); picos=int(picos)
#    if (hours <0) or (hours >24):
#        raise ValueError, "%s is not a valid hours"%hours
#    if (minutes <0) or (minutes >60):
#        raise ValueError, "%s is not a valid minutes"%minutes
#    if (seconds <0) or (seconds >60):
#        raise ValueError, "%s is not a valid seconds"%seconds
#    return {'hours' :hours,  'minutes' :minutes, 'seconds' :seconds, 
#            'millis' :millis, 'micros' :micros, 'nanos' :nanos, 'picos' : picos}

class TimeStamp:
    """This class wraps the xml-rpc timestamp structure. """
    def __init__(self, date=Date(), time=Time(), tstamp=None):             
        if tstamp is not None: 
            self.date=Date(dict=tstamp["date"])
            self.time=Time(dict=tstamp["time"])
        else:
            self.date=date
            self.time=time
    def getParams(self):
        """Describes time stamp as a xml-rpc structure."""
        return {'date': self.date.copy() ,'time' : self.time.copy()}
    def getTimeInSeconds(self):
        import time as _time
        tpl=(self.date["year"], self.date["month"], self.date["day"],
             self.time["hours"], self.time["minutes"], self.time["seconds"],
             0,0,0,)
        return int(_time.mktime(tpl))
    def getTimeInMillis(self):
        return self.getTimeInSeconds() * 1000 + self.time["millis"]
    def getTimeInMicros(self):
        return self.getTimeInMillis()* 1000 + self.time["micros"]
    def getTimeInNanos(self):
        return self.getTimeInMicros()* 1000 + self.time["nanos"]
    def getTimeInPicos(self):
        return self.getTimeInNanos()* 1000 + self.time["picos"]
        
#def TimeStamp2(date=Date(), time=Time()):
#    """This function wraps the xml-rpc timestamp structure. """
#    return {'date': date.copy() ,'time' : time.copy()}

#class Dte(dict):
#    def __init__(self, dict=None):             
#        if dict is not None: self.update(dict) # copies all the keys and values from one dictionary to another
#        self["year"]=2003
#        self["month"]=8
#
#if __name__ == "__main__":
#
#    # simple test program
#    from xmlrpclib import  dumps
#
#    # server = ServerProxy("http://localhost:8000") # local server "http://betty.userland.com"
#    params = Dte()
#    #params=  {'month': 8, 'day': 2003}
#    tuple_params = tuple([params.copy()])
#    xmlrpccall = dumps(tuple_params, 'meerkat.getItems')
#    print(xmlrpccall)
