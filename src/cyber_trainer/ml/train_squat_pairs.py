import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class PairWindowDS(Dataset):
    def __init__(self, npz_path: str, window: int = 45, stride: int = 15):
        d = np.load(npz_path, allow_pickle=True)
        self.Xf = d["X_front"]
        self.Xs = d["X_side"]
        self.y = d["y_form"].astype(np.int64)

        self.window = int(window)
        self.stride = int(stride)
        self.index = []

        for i in range(len(self.Xf)):
            T = min(self.Xf[i].shape[0], self.Xs[i].shape[0])
            if T < self.window:
                self.index.append((i, 0))
            else:
                for s in range(0, T - self.window + 1, self.stride):
                    self.index.append((i, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        xf = self.Xf[i].astype(np.float32)  # (T,33,5)
        xs = self.Xs[i].astype(np.float32)

        T = min(xf.shape[0], xs.shape[0])
        xf = xf[:T]
        xs = xs[:T]

        W = self.window
        if T >= W:
            wf = xf[s:s+W]
            ws = xs[s:s+W]
        else:
            wf = np.zeros((W, 33, 5), dtype=np.float32)
            ws = np.zeros((W, 33, 5), dtype=np.float32)
            wf[:T] = xf
            ws[:T] = xs

        wf = wf.reshape(W, -1)
        ws = ws.reshape(W, -1)
        x = np.concatenate([wf, ws], axis=1)  # (W,F)
        x = torch.from_numpy(x)               # (W,F)
        y = torch.tensor(int(self.y[i]), dtype=torch.float32)  # 0/1
        return x, y

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, dilation=1, p_drop=0.15):
        super().__init__()
        pad = (k - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation, padding=pad)
        self.drop = nn.Dropout(p_drop)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = y[..., :-self.pad] if self.pad > 0 else y
        y = F.relu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = y[..., :-self.pad] if self.pad > 0 else y
        y = F.relu(y)
        y = self.drop(y)
        return y + self.res(x)

class SquatFormTCN(nn.Module):
    def __init__(self, in_dim: int, width: int = 192, levels: int = 5):
        super().__init__()
        layers = []
        c = in_dim
        for i in range(levels):
            layers.append(TCNBlock(c, width, k=5, dilation=2**i, p_drop=0.15))
            c = width
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        # x: (B,W,F)
        x = x.transpose(1, 2)     # (B,F,W)
        h = self.tcn(x)           # (B,width,W)
        g = self.pool(h).squeeze(-1)
        return self.head(g).squeeze(-1)  # (B,) logits

def train(npz_path: str, out_pt: str, window: int = 45, stride: int = 15, epochs: int = 20, bs: int = 64, lr: float = 2e-3):
    ds = PairWindowDS(npz_path, window=window, stride=stride)
    n = len(ds)
    n_train = int(0.9 * n)
    n_val = n - n_train
    tr, va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)

    x0, _ = ds[0]
    in_dim = x0.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SquatFormTCN(in_dim=in_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        m = 0

        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            tr_acc += (pred == y).float().mean().item()
            tr_loss += loss.item()
            m += 1

        tr_loss /= max(m, 1)
        tr_acc /= max(m, 1)

        model.eval()
        va_acc = 0.0
        va_loss = 0.0
        k = 0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)

                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                va_acc += (pred == y).float().mean().item()
                va_loss += loss.item()
                k += 1

        va_acc /= max(k, 1)
        va_loss /= max(k, 1)

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best:
            best = va_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_dim": in_dim,
                    "window": window,
                },
                out_pt,
            )
            print(f"[SAVED] {out_pt} (best val acc={best:.3f})")

    print("Done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    train(args.npz, args.out, args.window, args.stride, args.epochs, args.bs, args.lr)
