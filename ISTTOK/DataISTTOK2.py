import argparse

import StartSdas

from sdas.core.client.SDASClient import SDASClient

import LoadSdasData

import matplotlib.pyplot as plt

 

def get_arguments():

    parser = argparse.ArgumentParser(description='Mirnov coils')

    parser.add_argument('-p', help='pulse (shot) number', default='46241', type=str)

    return parser.parse_args()

 

def signal_name_dict():

    sn = {}

    for n in range(12):

        sn.update({f'Mirnov coil no. {n+1}': f'MARTE_NODE_IVO3.DataCollection.Channel_{str(166+n).zfill(3)}'})

    return sn

 

def get_data(pulseNo, client=None):

    if(client == None):

        client = StartSdas.StartSdas()

    data = {}

    if(isinstance(client, SDASClient)):

        signalNames = signal_name_dict()

        for key in signalNames.keys():

            print(f'Downloading {key}...')

            data.update({key: LoadSdasData.LoadSdasData(client, signalNames[key], int(pulseNo))})

    else:

        raise TypeError('client is not of type SDASClient')

    return data

 

def plot_signals(pulseNo, data):

    NoOfColors = len(data.keys())

    cm = plt.get_cmap('jet')

    fig = plt.figure(figsize=(5, 7), tight_layout=True)

    ax = fig.add_subplot(111)

    ax.set_prop_cycle(color=[cm(1.*i/NoOfColors) for i in range(NoOfColors)])

    plt.title(f'#{pulseNo}: Mirnov coil signals')

    for n in range(len(data.keys())):

        key = f'Mirnov coil no. {n+1}'

        signal = data[key]

        if(len(signal) > 0):

            plt.plot(signal[1]*1e-3, signal[0], label=key.split('no. ')[-1])

            plt.xlabel('Time (ms)')

            plt.legend()

            plt.grid(True)

 

if(__name__=='__main__'):

    args = get_arguments()

    data = get_data(args.p)

    plot_signals(args.p, data)

   

    plt.show()