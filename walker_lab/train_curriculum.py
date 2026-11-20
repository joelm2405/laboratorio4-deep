"""Pipeline de entrenamiento con Curriculum Learning para BipedalWalker (Laboratorio 4)."""

import os
import sys
import time
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


def make_env_fn(profile="suave", difficulty="normal"):
    def _init():
        return WalkerEnv(profile=profile, difficulty=difficulty)
    return _init


def build_actor_from_model(model):
    """Extraer la red del Actor desde la política de Stable-Baselines3 a PolicyNetwork."""
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
    """Evaluar en las pistas de maratón oficiales de validación."""
    actor.eval()
    actor.to(device)
    epsilon = 1e-8
    clip_obs = 10.0
    
    val_files = sorted(VALIDATION_DIR.glob("maraton_*.json"))
    if not val_files:
        return 0, 0.0, 0.0
        
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


class MarathonEvalCallback(BaseCallback):
    """Callback para evaluar en las pistas de maratón y guardar el mejor modelo."""
    def __init__(self, eval_freq=32000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.best_successes = -1
        self.best_time = float("inf")
        self.best_distance = -1.0
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            mean = self.training_env.obs_rms.mean.copy()
            var = self.training_env.obs_rms.var.copy()
            actor = build_actor_from_model(self.model)
            
            successes, avg_time, avg_dist = evaluate_val_tracks(actor, mean, var, device="cpu")
            if self.verbose:
                print(f"\n>>> [EVAL @ {self.num_timesteps} pasos] Éxitos: {successes}/5 | Tiempo: {avg_time:.2f}s | Dist: {avg_dist:.2f}m", flush=True)
            
            # Ranking de la competencia:
            # 1. Mayor número de llegadas (successes)
            # 2. Menor tiempo promedio en llegadas (avg_time)
            # 3. Mayor distancia promedio acumulada (avg_dist)
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
                print(f"--> ¡GUARDANDO MEJOR MODELO! Éxitos: {successes}/5, Tiempo: {avg_time:.2f}s, Dist: {avg_dist:.2f}m", flush=True)
                AGENTE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez(AGENTE_DIR / "normalization.npz", mean=mean, var=var)
                torch.save(actor.state_dict(), AGENTE_DIR / "actor_weights.pt")
                self.model.save(ROOT / "checkpoints" / "best_sb3_model")
                self.training_env.save(ROOT / "checkpoints" / "best_vec_normalize.pkl")
                
        return True


def train():
    num_envs = 8
    print(f"Iniciando entrenamiento PPO con {num_envs} entornos concurrentes...")
    
    checkpoints_dir = ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    eval_callback = MarathonEvalCallback(eval_freq=32000, verbose=1)
    
    # ----------------------------------------------------
    # ETAPA 1: Locomoción básica y equilibrio en terreno suave
    # ----------------------------------------------------
    print("\n" + "="*65)
    print("ETAPA 1: Locomoción y estabilidad bípeda (profile='suave')")
    print("="*65, flush=True)
    env1 = SubprocVecEnv([make_env_fn(profile="suave") for _ in range(num_envs)])
    env1 = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    model = PPO(
        "MlpPolicy",
        env1,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh
        ),
        device="cpu",
        verbose=1
    )
    
    model.learn(total_timesteps=600000, callback=eval_callback, reset_num_timesteps=False)
    
    rms_mean = env1.obs_rms.mean.copy()
    rms_var = env1.obs_rms.var.copy()
    env1.close()
    
    # ----------------------------------------------------
    # ETAPA 2: Terreno variado con obstáculos (profile='mixto')
    # ----------------------------------------------------
    print("\n" + "="*65)
    print("ETAPA 2: Obstáculos mixtos: escalones, pendientes, bloques, huecos")
    print("="*65, flush=True)
    env2 = SubprocVecEnv([make_env_fn(profile="mixto") for _ in range(num_envs)])
    env2 = VecNormalize(env2, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env2.obs_rms.mean = rms_mean
    env2.obs_rms.var = rms_var
    
    model.set_env(env2)
    model.lr_schedule = lambda _: 2e-4
    model.ent_coef = 0.002
    
    model.learn(total_timesteps=800000, callback=eval_callback, reset_num_timesteps=False)
    
    rms_mean = env2.obs_rms.mean.copy()
    rms_var = env2.obs_rms.var.copy()
    env2.close()
    
    # ----------------------------------------------------
    # ETAPA 3: Recorridos completos de Maratón (profile='maraton')
    # ----------------------------------------------------
    print("\n" + "="*65)
    print("ETAPA 3: Maratones completos de 240 metros con obstáculos procedurales")
    print("="*65, flush=True)
    env3 = SubprocVecEnv([make_env_fn(profile="maraton") for _ in range(num_envs)])
    env3 = VecNormalize(env3, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env3.obs_rms.mean = rms_mean
    env3.obs_rms.var = rms_var
    
    model.set_env(env3)
    model.lr_schedule = lambda _: 1e-4
    model.ent_coef = 0.001
    model.clip_range = lambda _: 0.15
    
    model.learn(total_timesteps=1000000, callback=eval_callback, reset_num_timesteps=False)
    env3.close()
    
    print("\n" + "="*65)
    print("¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print(f"Mejor récord en validación:")
    print(f"  - Carreras completadas: {eval_callback.best_successes} / 5")
    print(f"  - Tiempo promedio: {eval_callback.best_time:.2f} s")
    print(f"  - Distancia promedio: {eval_callback.best_distance:.2f} m")
    print("Pesos y normalización listos en 'agente/'.")
    print("="*65, flush=True)


if __name__ == "__main__":
    train()
