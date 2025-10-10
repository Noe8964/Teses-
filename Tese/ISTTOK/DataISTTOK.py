import argparse
import json
import numpy as np
import StartSdas
from sdas.core.client.SDASClient import SDASClient
import LoadSdasData
import matplotlib.pyplot as plt
import os


def get_arguments():
    parser = argparse.ArgumentParser(description="Mirnov coils and plasma current")
    parser.add_argument("-p", help="pulse (shot) number", default="52657", type=str)
    return parser.parse_args()


def signal_name_dict():
    sn = {f"Mirnov coil no. {n+1}": f"MARTE_NODE_IVO3.DataCollection.Channel_{str(166+n).zfill(3)}" for n in range(12)}

    # Adding plasma current signals
    sn["Rogowski Plasma Current"] = "MARTE_NODE_IVO3.DataCollection.Channel_088"
    sn["Plasma Current"] = "MARTE_NODE_IVO3.DataCollection.Channel_100"

    return sn


def get_data(pulseNo, client=None):
    if client is None:
        client = StartSdas.StartSdas()

    data = {"pulse_id": pulseNo, "signals": {}}

    if isinstance(client, SDASClient):
        signalNames = signal_name_dict()

        for key, signalName in signalNames.items():
            print(f"Downloading {key}...")
            raw_data = LoadSdasData.LoadSdasData(client, signalName, int(pulseNo))
            
            if len(raw_data) > 0:
                # Convert NumPy arrays to lists before saving
                data["signals"][key] = {
                    "time": raw_data[1].tolist(),
                    "values": raw_data[0].tolist()
                }
            else:
                data["signals"][key] = {"time": [], "values": []}
    else:
        raise TypeError("client is not of type SDASClient")

    return data


def save_data_to_file(data, folder="D:\Tese\ISTTOK\Saved_data"):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Define the full file path
    filename = os.path.join(folder, f"pulse_{data['pulse_id']}.json")

    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Data successfully saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def plot_signals(data):
    pulseNo = data["pulse_id"]
    signals = data["signals"]

    NoOfColors = len(signals)
    cm = plt.get_cmap("jet")

    fig, ax = plt.subplots(figsize=(5, 7), tight_layout=True)
    ax.set_prop_cycle(color=[cm(1.0 * i / NoOfColors) for i in range(NoOfColors)])

    plt.title(f"#{pulseNo}: Mirnov coil & Plasma Current signals")

    for key, signal in signals.items():
        if len(signal["time"]) > 0:
            plt.plot(np.array(signal["time"]) * 1e-3, signal["values"], label=key)
    
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    args = get_arguments()
    data = get_data(args.p)
    save_data_to_file(data)
    plot_signals(data)
    client = StartSdas.StartSdas()
    print(client)  # Should show a valid SDASClient object
    params = client.searchParametersByName("Channel_088")
    print(params)



