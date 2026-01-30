import numpy as np
import torch

from cyber_trainer.ml.train_squat_pairs import SquatFormTCN, PairWindowDS

def main(npz_path: str, ckpt_path: str, n: int = 40):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    window = int(ckpt["window"])

    ds = PairWindowDS(npz_path, window=window, stride=window)
    x0, _ = ds[0]
    in_dim = x0.shape[-1]

    model = SquatFormTCN(in_dim=in_dim)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    idxs = np.linspace(0, len(ds) - 1, num=min(n, len(ds)), dtype=int)
    ok = 0

    for i in idxs:
        x, y = ds[i]
        with torch.no_grad():
            logit = model(x.unsqueeze(0)).item()
        p = 1.0 / (1.0 + np.exp(-logit))
        pred = 1 if p >= 0.5 else 0
        ok += int(pred == int(y.item()))
        print(f"#{i:04d} y={int(y.item())}  p(correct)={p:.3f}  pred={pred}")

    print(f"sample-acc: {ok}/{len(idxs)} = {ok/len(idxs):.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    main(args.npz, args.ckpt, args.n)
