
from sdas.core.client.SDASClient import SDASClient
import os
import pickle

def StartSdas():
    host='baco.ipfn.ist.utl.pt';
    port=8888;
    client = SDASClient(host,port);
    return client

def myStartSdas():
    if(os.path.exists('client.pkl')):
        fid = open('client.pkl', 'rb')
        client = pickle.load(fid)
        fid.close()
        try:
            maxEvent = client.searchMaxEventNumber('0x0000')
            print(maxEvent)
            if(maxEvent > 50000): return client
            else: raise
        except:
            print('Stored client did not worked, requesting a new one...')
    client = StartSdas()
    print(client)
    fid = open('client.pkl', 'wb')
    pickle.dump(client, fid, pickle.HIGHEST_PROTOCOL)
    fid.close()
    return client