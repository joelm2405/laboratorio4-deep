"""Ajuste fino especializado en obstáculos difíciles (bloques, huecos y maratones)."""

import os
import sys
from pathlib import Path
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from walker_lab.environment import WalkerEnv
from agente.architecture import PolicyNetwork

ROOT = Path(__file__).resolve().parent
AGENTE_DIR = ROOT / "agente"
VALIDATION_DIR = ROOT / "escenarios" / "validacion"
CHECKPOINTS_DIR = ROOT / "checkpoints"


def make_env_fn(profile):
    def _init():
        return WalkerEnv(profile=profile)
    return _init


def build_actor_from_model(model):
    actor = PolicyNetwork(obs_dim=24, act_dim=4, hidden_dims=(256, 256))
    policy = model.policy
    state_dict = {
        "net.0.weight": policy.mlp_extractor.policy_net[0].weight.data.clone(),
        "net.0.bias": policy.mlp_extractor.policy_net[0].bias.data.clone(),
        "net.2.weight": policy.mlp_extractor.policy_net[2].weight.data.clone(),
        "net.2.bias": policy.mlp_extractor.policy_net[2].bias.data.clone(),
        "net.4.weight": policy.action_net.weight.data.clone(),
        "net.4.bias": policy.action_net.bias.data.clone(),
    }
    actor.load_state_dict(state_dict)
    return actor


def evaluate_val_tracks(actor, mean, var, device="cpu"):
    actor.eval()
    actor.to(device)
    epsilon = 1e-8
    clip_obs = 10.0
    val_files = sorted(VALIDATION_DIR.glob("maraton_*.json"))
    
    successes = 0
    total_time = 0.0
    completed_times = []
    total_distance = 0.0
    
    for f in val_files:
        env = WalkerEnv(scenario=f)
        obs, _ = env.reset(seed=env.fixed_scenario["simulation_seed"])
        terminated = truncated = False
        while not (terminated or truncated):
            norm_obs = np.clip((obs - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs)
            with torch.no_grad():
                t_obs = torch.from_numpy(norm_obs.astype(np.float32)).unsqueeze(0).to(device)
                action = actor(t_obs).squeeze(0).cpu().numpy().astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()
        total_distance += info["distance"]
        if info["success"]:
            successes += 1
            completed_times.append(info["simulated_seconds"])
        total_time += info["simulated_seconds"]
        
    avg_dist = total_distance / len(val_files)
    avg_time = (sum(completed_times) / len(completed_times)) if completed_times else (total_time / len(val_files))
    return successes, avg_time, avg_dist


class ObstacleFineTuneCallback(BaseCallback):
    def __init__(self, eval_freq=24000, initial_best_success=1, initial_best_dist=177.9, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.best_successes = initial_best_success
        self.best_time = 84.08
        self.best_distance = initial_best_dist
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            mean = self.training_env.obs_rms.mean.copy()
            var = self.training_env.obs_rms.var.copy()
            actor = build_actor_from_model(self.model)
            
            successes, avg_time, avg_dist = evaluate_val_tracks(actor, mean, var, device="cpu")
            if self.verbose:
                print(f"\n>>> [FINETUNE EVAL @ {self.num_timesteps}] Éxitos: {successes}/5 | Tiempo: {avg_time:.2f}s | Dist: {avg_dist:.2f}m", flush=True)
            
            is_better = False
            if successes > self.best_successes:
                is_better = True
            elif successes == self.best_successes and successes > 0:
                if avg_time < self.best_time:
                    is_better = True
            elif successes == self.best_successes and successes == 0:
                if avg_dist > self.best_distance:
                    is_better = True
                    
            if is_better:
                self.best_successes = successes
                self.best_time = avg_time
                self.best_distance = avg_dist
                print(f"--> ¡NUEVO RECORD SUPERADO! Éxitos: {successes}/5, Tiempo: {avg_time:.2f}s, Dist: {avg_dist:.2f}m", flush=True)
                AGENTE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez(AGENTE_DIR / "normalization.npz", mean=mean, var=var)
                torch.save(actor.state_dict(), AGENTE_DIR / "actor_weights.pt")
                self.model.save(CHECKPOINTS_DIR / "finetuned_sb3_model")
                self.training_env.save(CHECKPOINTS_DIR / "finetuned_vec_normalize.pkl")
                
        return True


def finetune():
    profiles = ["bloques", "huecos", "bloques", "huecos", "mixto", "maraton", "mixto", "maraton"]
    print("Creando 8 entornos mixtos orientados a obstáculos:", profiles)
    
    env = SubprocVecEnv([make_env_fn(p) for p in profiles])
    
    # Cargar normalización previa
    best_norm_path = str(CHECKPOINTS_DIR / "best_vec_normalize.pkl")
    if os.path.exists(best_norm_path):
        print("Cargando normalizador previo...")
        env = VecNormalize.load(best_norm_path, env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
    best_model_path = str(CHECKPOINTS_DIR / "best_sb3_model.zip")
    print("Cargando modelo PPO preentrenado...")
    model = PPO.load(
        best_model_path,
        env=env,
        device="cpu",
        learning_rate=lambda _: 8e-5,
        clip_range=lambda _: 0.15,
        ent_coef=0.002
    )
    
    callback = ObstacleFineTuneCallback(eval_freq=24000, initial_best_success=1, initial_best_dist=177.9, verbose=1)
    
    print("\nIniciando ajuste fino de 400,000 pasos en obstáculos...")
    model.learn(total_timesteps=400000, callback=callback)
    env.close()
    print("Ajuste fino finalizado.")


if __name__ == "__main__":
    finetune()
