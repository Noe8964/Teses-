# @author Bernardo Carvalho
#@version 1.0.0
# author: $Author: bernardo $
# Revision $Revision: 174 $
# $HeadURL: svn://baco.ipfn.ist.utl.pt/sdas/trunk/python/src/sdas/tests/testSDASwx.py $
# $Id: testSDASwx.py 174 2006-06-27 16:35:31Z bernardo $
#----------------------------------------------------------------------
# A very simple wxPython example.  Just a wx.Frame, wx.Panel,
# wx.StaticText, wx.Button, and a wx.BoxSizer, but it shows the basic
# structure of any wxPython application.
#----------------------------------------------------------------------

import wx, numarray

from wx.lib.plot import PlotCanvas, PolyLine, PlotGraphics, _Numeric

from sdas.core.client.SDASClient import SDASClient
from sdas.core.SDAStime import Date, Time, TimeStamp

class MyFrame(wx.Frame):
    """
    This is MyFrame.  It just shows a few controls on a wxPanel,
    and has a simple menu.
    """
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, -1, title,
                          pos=(150, 150), size=(550, 400))

        # Create the menubar
        menuBar = wx.MenuBar()

        # and a menu 
        menu = wx.Menu()

        # add an item to the menu, using \tKeyName automatically
        # creates an accelerator, the third param is some help text
        # that will show up in the statusbar
        menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit this simple sample")

        # bind the menu event to an event handler
        self.Bind(wx.EVT_MENU, self.OnTimeToClose, id=wx.ID_EXIT)

        # and put the menu on the menubar
        menuBar.Append(menu, "&File")
        self.SetMenuBar(menuBar)

        self.CreateStatusBar()
        

        # Now create the Panel to put the other controls on.
        #panel = wx.Panel(self)
        self.plot = PlotCanvas(self)
        self.fill_Plot()


    def OnTimeToClose(self, evt):
        """Event handler for the button click."""
        print "See ya later!"
        self.Close()

    def OnFunButton(self, evt):
        """Event handler for the button click."""
        print "Having fun yet?"
    def fill_Plot(self):        
        
        
        # server = ServerProxy("http://localhost:8000") # local server "http://betty.userland.com"
        host='baco.cfn.ist.utl.pt'
        port=8888
        client = SDASClient(host,port)
        #dS=client.getData('CENTRAL_OS9_ADC.IIGBT','0x0000', 11244)
        dS=client.getData('CENTRAL_OS9_ADC_VME_I8.IF0SN','0x0000', 11244)
        isine =dS.getData()
        dS=client.getData('CENTRAL_OS9_ADC_VME_I8.IF0CS','0x0000', 11244)
        cosine =dS.getData()
        ic=Density(cosine,isine)
        sze=ic.size()
        print sze
        #ic=_Numeric.arange(ic)
         # 25,000 point line
        data1 = _Numeric.arange(2*sze)
        data2 = data1.astype(_Numeric.Float32)
        data2.shape = (-1, 2)
        data2[:,1] =ic
        line1 = PolyLine(data2, legend='Wide Line', colour='green', width=2)
        self.plot.Draw(PlotGraphics([line1], "SDAS Plot", "Time", ""))



def Density(cosine,isine):
    phase = _Numeric.arctan2(cosine,isine)
    diff= phase[:-1] - phase[1:]
    pi = _Numeric.pi
    p=_Numeric.where(diff > pi, 1, 0)
    p=_Numeric.where(diff < -pi, -1, p)
    p = 2 * pi * p
    r =_Numeric.cumsum(p)
    r= _Numeric.concatenate((_Numeric.array([0.0]), r,))
    unwraped = phase + r
    dens = -7e17*unwraped
    density = dens - _Numeric.average(dens[-100:])
    print density[4000]
    return density

#def fill_Plot():
#	
class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, "SDAS test wxPython App")
        self.SetTopWindow(frame)

        print "Print statements go to this stdout window by default."

        frame.Show(True)
        return True
        
app = MyApp(redirect=False)
app.MainLoop()

