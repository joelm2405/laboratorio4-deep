from pathlib import Path

import numpy as np
import torch

from walker_lab.agent import BaseAgent
from .architecture import PolicyNetwork


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        model_dir = Path(self.model_dir)

        weights_path = model_dir / "actor_weights.pt"
        norm_path = model_dir / "normalization.npz"
        for path in (weights_path, norm_path):
            if not path.is_file():
                raise FileNotFoundError(f"No se encontró {path}")

        # mean/var guardados del VecNormalize de SB3
        norm = np.load(norm_path)
        self.mean = norm["mean"].astype(np.float32)
        self.std = np.sqrt(norm["var"].astype(np.float32) + 1e-8)

        self.policy = PolicyNetwork()
        self.policy.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.policy.to(self.device)
        self.policy.eval()

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        obs = np.clip((obs - self.mean) / self.std, -10.0, 10.0)

        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action = self.policy(x)[0].cpu().numpy()

        return np.clip(np.nan_to_num(action), -1.0, 1.0).astype(np.float32)
