from sdas.core.client.SDASClient import SDASClient

def StartSdas():
    host='baco.ipfn.ist.utl.pt';
    port=8888;
    client = SDASClient(host,port);
    return client
