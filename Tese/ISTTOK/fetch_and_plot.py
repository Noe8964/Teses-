import json
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the folder where JSON files are stored
DATA_FOLDER = r"D:\Tese\ISTTOK\Saved_data"

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

        # Sort files by pulse number (assuming filenames like 'pulse_46241.json')
        latest_file = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
        return os.path.join(DATA_FOLDER, latest_file)

    except Exception as e:
        print(f"❌ Error finding latest file: {e}")
        return None


def get_all_pulse_files():
    """Return a list of all pulse JSON files in the directory, sorted by pulse number."""
    try:
        files = [f for f in os.listdir(DATA_FOLDER) if f.startswith("pulse_") and f.endswith(".json")]
        if not files:
            print("❌ No pulse data files found.")
            return []

        sorted_files = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        return [os.path.join(DATA_FOLDER, f) for f in sorted_files]

    except Exception as e:
        print(f"❌ Error finding pulse files: {e}")
        return []


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
    if data is None:
        return

    pulseNo = data["pulse_id"]
    signals = data["signals"]

    print("Loaded signal keys:", list(signals.keys()))

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    fig.suptitle(f"#{pulseNo}: Plasma Current with Squared Simplification")

    for key in CURRENT_SIGNALS.keys():
        if key in signals and len(signals[key]["time"]) > 0:
            time = np.array(signals[key]["time"]) * 1e-3  # Convert to ms
            values = np.array(signals[key]["values"])
            print(f"✅ '{key}' signal shape: {values.shape}")

            square = three_state_square(values, threshold=500)

            ax.plot(time, values, label=f"{key} (original)")
            ax.plot(time, square * 500, "--", label=f"{key} (squared)")  
        else:
            print(f"⚠️ Warning: Signal '{key}' not found or has no data.")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current / Squared state")
    ax.legend()
    ax.grid(True)

    plt.show()


def three_state_square(signal, threshold=500):
    """Convert signal into ±1 / 0 square wave."""
    square = np.zeros_like(signal)
    square[signal > threshold] = 1
    square[signal < -threshold] = -1
    return square


if __name__ == "__main__":
    # --- Option A: Use latest file ---
    latest_file = get_latest_pulse_file()
    if latest_file:
        print(f"✅ Using latest pulse file: {latest_file}")
        data = load_data_from_file(latest_file)
        plot_currents(data)

    # --- Option B: Loop through all files ---
    all_files = get_all_pulse_files()
    if all_files:
        print(f"✅ Found {len(all_files)} pulse files.")
        for f in all_files:
            print(f"\n📂 Processing: {f}")
            data = load_data_from_file(f)
            plot_currents(data)

