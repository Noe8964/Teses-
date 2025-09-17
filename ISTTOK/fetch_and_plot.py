import json
import numpy as np
import os
import matplotlib.pyplot as plt


# Define the folder where JSON files are stored
DATA_FOLDER = "/home/noe/Documents/ISTTOK/Saved_data"

# Define the current signals
CURRENT_SIGNALS = {
    "Plasma Current": "MARTE_NODE_IVO3.DataCollection.Channel_100",
    "Rogowski Plasma Current": "MARTE_NODE_IVO3.DataCollection.Channel_088"
}


def get_latest_pulse_file():
    """Find the most recent pulse file in the directory."""
    try:
        files = [f for f in os.listdir(DATA_FOLDER) if f.startswith("pulse_") and f.endswith(".json")]
        if not files:
            print("❌ No pulse data files found.")
            return None

        # Sort files by pulse number (assuming filenames are like 'pulse_46241.json')
        latest_file = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
        return os.path.join(DATA_FOLDER, latest_file)

    except Exception as e:
        print(f"❌ Error finding latest file: {e}")
        return None


def load_data_from_file(filepath):
    """Load JSON data from a given file path."""
    if not os.path.exists(filepath):
        print(f"❌ Error: File {filepath} not found.")
        return None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return None


def plot_currents(data):
    """Plots only the two current signals from the dataset."""
    if data is None:
        return

    pulseNo = data["pulse_id"]
    signals = data["signals"]

    # Debug: print all loaded signal keys
    print("Loaded signal keys:", list(signals.keys()))

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    plt.title(f"#{pulseNo}: Plasma Current Signals")

    for key in CURRENT_SIGNALS.keys():
        if key in signals and len(signals[key]["time"]) > 0:
            time = np.array(signals[key]["time"]) * 1e-3  # Convert to ms
            values = signals[key]["values"]
            # Debug: print shape of the signal values
            print(f"✅ '{key}' signal shape: {np.array(values).shape}")
            plt.plot(time, values, label=key)
        else:
            print(f"⚠️ Warning: Signal '{key}' not found or has no data.")
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (A)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    latest_file = get_latest_pulse_file()
    
    if latest_file:
        print(f"✅ Using latest pulse file: {latest_file}")
        data = load_data_from_file(latest_file)
        plot_currents(data)
        data["signals"]
    else:
        print("❌ No valid pulse file found.")
