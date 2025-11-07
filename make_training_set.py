import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

# 📂 Normalized data folder
NORMALIZED_FOLDER = "/home/noe/Documents/ISTTOK/Normalized_data2"


class DischargeTrainingSet:
    """
    Training set where each sample spans from one reversal to the next.
    No batching yet.
    """
    def __init__(self, folder=NORMALIZED_FOLDER):
        self.folder = folder

        # 🔍 Find all normalized pulses
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith("_normalized.json")
        ]
        if not self.files:
            raise FileNotFoundError(f"No normalized files found in {folder}")

        print(f"📁 Found {len(self.files)} normalized pulses")

        # Determine union of all features
        self.all_keys = self._collect_all_signals()
        print(f"📊 Using {len(self.all_keys)} total features:\n   {self.all_keys}")

        # Build reversal-to-reversal windows
        self.samples = []
        self.window_sizes = []
        self._build_windows()

        print(f"✅ Created {len(self.samples)} total samples")
        print(f"📏 Average window size: {np.mean(self.window_sizes):.1f} | "
              f"Max: {np.max(self.window_sizes)} | Min: {np.min(self.window_sizes)}")

    # Collect all features from all pulses
    def _collect_all_signals(self):
        all_keys = set()
        for f in self.files:
            with open(f, "r") as j:
                data = json.load(j)
            all_keys.update([k for k in data["signals"].keys() if k != "time"])
        return sorted(all_keys)

    # Build samples
    def _build_windows(self):
        for f in self.files:
            with open(f, "r") as j:
                data = json.load(j)

            pid = data["pulse_id"]
            signals = data["signals"]
            reversals = data.get("reversals", [])

            # Time reference
            time = np.array(signals["Plasma Current"]["time"], dtype=np.float32)

            # Feature matrix (fill missing signals with zeros)
            X_list = []
            for key in self.all_keys:
                if key in signals:
                    vals = np.array(signals[key]["values"], dtype=np.float32)
                else:
                    vals = np.zeros_like(time, dtype=np.float32)
                X_list.append(vals)
            X_full = np.stack(X_list, axis=1)

            # Find reversal indices
            reversal_indices = []
            for rev in reversals:
                t_rev = rev["time"]
                idx = np.abs(time - t_rev).argmin()
                reversal_indices.append(idx)
            reversal_indices = sorted(list(set(reversal_indices)))

            if len(reversal_indices) < 2:
                continue

            # Build windows
            for i in range(len(reversal_indices) - 1):
                start = reversal_indices[i]
                end = reversal_indices[i + 1]

                x_window = X_full[start:end]
                y_target = float(reversals[i + 1]["Y"]) if (i + 1) < len(reversals) else np.nan

                if len(x_window) < 5:
                    continue

                self.samples.append({
                    "pulse_id": pid,
                    "X": x_window,
                    "Y": y_target,
                    "mask": 0.0 if np.isnan(y_target) else 1.0
                })
                self.window_sizes.append(len(x_window))

    # Access a sample
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "pulse_id": s["pulse_id"],
            "X": torch.tensor(s["X"], dtype=torch.float32),
            "Y": torch.tensor(np.nan_to_num(s["Y"], nan=-1.0), dtype=torch.float32),
            "mask": torch.tensor(s["mask"], dtype=torch.float32)
        }


# Plot a few random windows
def plot_random_windows(dataset, n_samples=5, n_features_to_show=3):
    fig, axes = plt.subplots(n_samples, n_features_to_show, figsize=(14, 2.5 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_samples):
        sample = random.choice(dataset.samples)
        X = sample["X"]
        Y = sample["Y"]
        pid = sample["pulse_id"]

        t = np.arange(len(X))
        for j in range(n_features_to_show):
            ax = axes[i, j] if n_features_to_show > 1 else axes[i]
            ax.plot(t, X[:, j])
            ax.set_title(f"Pulse {pid} | {dataset.all_keys[j]}\nY={Y}")
            ax.grid(True)

    plt.tight_layout()
    plt.show()


# 🧪 Example usage
if __name__ == "__main__":
    dataset = DischargeTrainingSet()
    print(f"\nTotal samples: {len(dataset)}")
    print(f"First sample X shape: {dataset[0]['X'].shape}, Y: {dataset[0]['Y']}, mask: {dataset[0]['mask']}")

    # Plot a few random windows
    #plot_random_windows(dataset, n_samples=5, n_features_to_show=4)






