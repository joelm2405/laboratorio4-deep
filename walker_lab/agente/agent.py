"""Adaptador de inferencia para el bípedo (Laboratorio 4)."""

from pathlib import Path
import numpy as np
import torch

from walker_lab.agent import BaseAgent
from .architecture import PolicyNetwork


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        model_path = Path(self.model_dir)
        
        # Cargar parámetros de normalización (RunningMeanStd)
        norm_file = model_path / "normalization.npz"
        if norm_file.exists():
            data = np.load(norm_file)
            self.mean = data["mean"].astype(np.float32)
            self.var = data["var"].astype(np.float32)
            self.epsilon = 1e-8
            self.clip_obs = 10.0
        else:
            self.mean = None
            self.var = None

        # Cargar arquitectura y pesos de la política (Actor)
        self.device = torch.device(device)
        self.policy = PolicyNetwork(obs_dim=24, act_dim=4, hidden_dims=(256, 256))
        
        weights_file = model_path / "actor_weights.pt"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(state_dict)
            
        self.policy.to(self.device)
        self.policy.eval()

    def reset(self):
        """Reiniciar el estado recurrente o memoria (no requerido para política feed-forward)."""
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Inferencia determinista de la política entrenada."""
        obs = np.asarray(observation, dtype=np.float32)
        
        # Aplicar la misma normalización usada en el entrenamiento
        if self.mean is not None and self.var is not None:
            obs = np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)
            
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action = self.policy(obs_tensor)
            # Devuelve numpy.ndarray en CPU de shape (4,)
            return action.squeeze(0).cpu().numpy().astype(np.float32)
