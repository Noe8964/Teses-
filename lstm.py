import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from make_training_set import DischargeTrainingSet  # your dataset builder

# ---------- Hyperparameters ----------
BATCH_SIZE = 16
HIDDEN_SIZE = 128
NUM_LAYERS = 10
NUM_EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load dataset ----------
dataset = DischargeTrainingSet()

# --- Get unique pulse IDs and split by pulses (not random samples)
pulse_ids = list({sample["pulse_id"] for sample in dataset})
pulse_ids.sort()

random.seed()
random.shuffle(pulse_ids)
split_idx = int(0.8 * len(pulse_ids))
train_pulses = set(pulse_ids[:split_idx])
val_pulses = set(pulse_ids[split_idx:])

print(f"Total pulses: {len(pulse_ids)}")
print(f"Training pulses: {len(train_pulses)}")
print(f"Validation pulses: {len(val_pulses)}")

# --- Build train/val subsets
train_indices = [i for i, s in enumerate(dataset) if s["pulse_id"] in train_pulses]
val_indices = [i for i, s in enumerate(dataset) if s["pulse_id"] in val_pulses]

train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)

print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")

# ---------- Collate function for variable-length sequences ----------
def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x["X"].shape[0], reverse=True)
    lengths = [x["X"].shape[0] for x in batch]
    max_len = max(lengths)
    n_features = batch[0]["X"].shape[1]

    padded_X = torch.zeros(len(batch), max_len, n_features)
    Y = torch.stack([x["Y"] for x in batch])
    mask = torch.stack([x["mask"] for x in batch])

    for i, b in enumerate(batch):
        seq_len = b["X"].shape[0]
        padded_X[i, :seq_len, :] = b["X"]

    return {
        "pulse_id": [b["pulse_id"] for b in batch],
        "X": padded_X.to(DEVICE),
        "Y": Y.to(DEVICE),
        "mask": mask.to(DEVICE),
        "lengths": lengths
    }

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------- LSTM Model ----------
class ReversalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.lstm(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)

# ---------- Initialize model and optimizer ----------
sample = dataset[0]
input_size = sample["X"].shape[1]
model = ReversalLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss(reduction='none')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# ---------- Training loop ----------
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- Training ----
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        X, Y, mask, lengths = batch["X"], batch["Y"], batch["mask"], batch["lengths"]
        optimizer.zero_grad()
        outputs = model(X, lengths)

        last_idx = [l-1 for l in lengths]
        preds = torch.stack([outputs[i, last_idx[i]] for i in range(len(last_idx))])
        labels = Y
        mask_last = mask

        loss = (criterion(preds, labels) * mask_last).sum() / mask_last.sum()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        y_pred.extend((preds[mask_last>0].detach().cpu().numpy() > 0.5).astype(int))
        y_true.extend(labels[mask_last>0].detach().cpu().numpy())

    train_losses.append(total_loss / len(y_true))
    train_f1s.append(f1_score(y_true, y_pred))

    # ---- Validation ----
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            X, Y, mask, lengths = batch["X"], batch["Y"], batch["mask"], batch["lengths"]
            outputs = model(X, lengths)

            last_idx = [l-1 for l in lengths]
            preds = torch.stack([outputs[i, last_idx[i]] for i in range(len(last_idx))])
            labels = Y
            mask_last = mask

            loss = (criterion(preds, labels) * mask_last).sum() / mask_last.sum()
            total_loss += loss.item() * len(labels)

            y_true.extend(labels[mask_last>0].cpu().numpy())
            y_pred.extend((preds[mask_last>0].cpu().numpy() > 0.5).astype(int))

    val_losses.append(total_loss / len(y_true))
    val_f1s.append(f1_score(y_true, y_pred))

    print(f"Epoch {epoch}/{NUM_EPOCHS} | "
          f"Train Loss: {train_losses[-1]:.4f}, F1: {train_f1s[-1]:.3f} | "
          f"Val Loss: {val_losses[-1]:.4f}, F1: {val_f1s[-1]:.3f} | LR: {optimizer.param_groups[0]['lr']:.5f}")

    scheduler.step(val_f1s[-1])

# ---------- Plot training curves ----------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_f1s, label='Train F1')
plt.plot(val_f1s, label='Val F1')
plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Example prediction ----------
model.eval()
sample_idx = random.randint(0, len(dataset)-1)
sample = dataset[sample_idx]
X_sample = sample["X"].unsqueeze(0).to(DEVICE)
lengths = [X_sample.shape[1]]
with torch.no_grad():
    output_seq = model(X_sample, lengths)
    pred = output_seq[0, -1].item()
print(f"\nPulse {sample['pulse_id']} | True Y: {sample['Y'].item()} | Predicted: {pred:.3f}")

# ---------- Confusion Matrices ----------
def evaluate_confusion(loader, name=""):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            X, Y, mask, lengths = batch["X"], batch["Y"], batch["mask"], batch["lengths"]
            outputs = model(X, lengths)
            last_idx = [l-1 for l in lengths]
            preds = torch.stack([outputs[i, last_idx[i]] for i in range(len(last_idx))])
            labels = Y
            mask_last = mask
            y_true.extend(labels[mask_last>0].cpu().numpy())
            y_pred.extend((preds[mask_last>0].cpu().numpy() > 0.5).astype(int))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0 (no reversal)", "1 (reversal)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"{name} Confusion Matrix:")
        print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
        print(f"  Precision: {tp / (tp + fp + 1e-8):.3f} | Recall: {tp / (tp + fn + 1e-8):.3f} | F1: {2*tp / (2*tp + fp + fn + 1e-8):.3f}")
    else:
        print(f"{name} Confusion Matrix:\n{cm}")

evaluate_confusion(train_loader, name="Training")
evaluate_confusion(val_loader, name="Validation")




