from collections import deque
import numpy as np
import torch

from cyber_trainer.ml.train_squat_pairs import SquatFormTCN

class RealtimeSquatForm:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.window = int(ckpt["window"])
        self.in_dim = int(ckpt["in_dim"])

        self.model = SquatFormTCN(in_dim=self.in_dim)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

        self.buf_front = deque(maxlen=self.window)
        self.buf_side = deque(maxlen=self.window)

    def push(self, feat_front_33x5: np.ndarray, feat_side_33x5: np.ndarray):
        self.buf_front.append(feat_front_33x5.reshape(-1).astype(np.float32))
        self.buf_side.append(feat_side_33x5.reshape(-1).astype(np.float32))

    def ready(self) -> bool:
        return len(self.buf_front) == self.window and len(self.buf_side) == self.window

    @torch.no_grad()
    def predict_p_correct(self):
        if not self.ready():
            return None
        wf = np.stack(list(self.buf_front), axis=0)  # (W,Ff)
        ws = np.stack(list(self.buf_side), axis=0)   # (W,Fs)
        x = np.concatenate([wf, ws], axis=1).astype(np.float32)  # (W,F)
        x = torch.from_numpy(x).unsqueeze(0)  # (1,W,F)
        logit = self.model(x).squeeze(0).item()
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(p)

    def reset(self):
        self.buf_front.clear()
        self.buf_side.clear()
