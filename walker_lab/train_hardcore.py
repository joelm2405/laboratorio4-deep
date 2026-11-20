"""Entrenamiento especializado en dificultad HARDCORE / HARDCORE-PLUS (Laboratorio 4)."""

import os
import shutil
import zipfile
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
CKPT_DIR = ROOT / "checkpoints"
HARDCORE_DIR = ROOT / "escenarios" / "hardcore"
VAL_DIR = ROOT / "escenarios" / "validacion"

# 8 entornos paralelos combinando los perfiles de mayor dificultad
ENV_CONFIGS = [
    ("escalones", "hardcore"),
    ("bloques", "hardcore"),
    ("huecos", "hardcore"),
    ("mixto", "hardcore"),
    ("maraton", "hardcore"),
    ("maraton", "hardcore"),
    ("maraton", "hardcore-plus"),
    ("escalones", "hardcore-plus"),
]


def make_env(profile, difficulty):
    def _thunk():
        return WalkerEnv(profile=profile, difficulty=difficulty)
    return _thunk


def extract_actor(model):
    pi = model.policy
    actor = PolicyNetwork()
    actor.load_state_dict({
        "net.0.weight": pi.mlp_extractor.policy_net[0].weight.data.clone(),
        "net.0.bias": pi.mlp_extractor.policy_net[0].bias.data.clone(),
        "net.2.weight": pi.mlp_extractor.policy_net[2].weight.data.clone(),
        "net.2.bias": pi.mlp_extractor.policy_net[2].bias.data.clone(),
        "net.4.weight": pi.action_net.weight.data.clone(),
        "net.4.bias": pi.action_net.bias.data.clone(),
    })
    return actor.eval()


def update_agente_zip():
    zip_path = ROOT / "agente.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in AGENTE_DIR.iterdir():
            if file.is_file() and not file.name.endswith(".pyc"):
                zipf.write(file, arcname=f"agente/{file.name}")
    print(f"--> [ZIP ACTUALIZADO] {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)", flush=True)


def evaluate_hardcore_and_val(actor, mean, var):
    std = np.sqrt(var + 1e-8)
    
    # 5 pistas hardcore + 5 pistas validacion = 10 pistas maratónicas
    hardcore_tracks = sorted(HARDCORE_DIR.glob("maraton_*.json"))
    val_tracks = sorted(VAL_DIR.glob("maraton_*.json"))
    all_tracks = hardcore_tracks + val_tracks
    
    wins = 0
    dists = []
    times = []
    win_times = []

    for track in all_tracks:
        env = WalkerEnv(scenario=track)
        obs, _ = env.reset(seed=env.fixed_scenario["simulation_seed"])
        done = False
        while not done:
            x = np.clip((obs - mean) / std, -10.0, 10.0).astype(np.float32)
            with torch.no_grad():
                action = actor(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        env.close()

        dists.append(info["distance"])
        times.append(info["simulated_seconds"])
        if info["success"]:
            wins += 1
            win_times.append(info["simulated_seconds"])

    avg_time = float(np.mean(win_times)) if win_times else float(np.mean(times))
    avg_dist = float(np.mean(dists))
    return wins, len(all_tracks), avg_time, avg_dist


class HardcoreSaveBestCallback(BaseCallback):
    def __init__(self, eval_freq=32000, best_score=None, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval = 0
        
        # Cargar baseline inicial
        if best_score is not None:
            self.best_wins, self.best_time, self.best_dist = best_score
        else:
            self.best_wins = 1
            self.best_time = 76.68
            self.best_dist = 160.0

    def is_better(self, wins, t, dist):
        # 1. Mayor número de llegadas
        # 2. Menor tiempo promedio en victorias
        # 3. Mayor distancia promedio acumulada
        if wins != self.best_wins:
            return wins > self.best_wins
        if wins > 0:
            return t < self.best_time
        return dist > self.best_dist

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps

        mean = self.training_env.obs_rms.mean.copy()
        var = self.training_env.obs_rms.var.copy()
        actor = extract_actor(self.model)

        wins, total_evals, avg_t, avg_dist = evaluate_hardcore_and_val(actor, mean, var)
        print(f"\n[EVAL HC @ {self.num_timesteps}] Llegadas: {wins}/{total_evals} | Tiempo: {avg_t:.2f}s | Dist: {avg_dist:.2f}m", flush=True)

        if self.is_better(wins, avg_t, avg_dist):
            self.best_wins = wins
            self.best_time = avg_t
            self.best_dist = avg_dist
            print(f"--> ¡NUEVO RECORD HARDCORE SUPERADO! Llegadas: {wins}/{total_evals}, Tiempo: {avg_t:.2f}s, Dist: {avg_dist:.2f}m", flush=True)

            # 1. Guardar en checkpoints seguros
            CKPT_DIR.mkdir(exist_ok=True)
            self.model.save(CKPT_DIR / "hardcore_best_model")
            self.training_env.save(CKPT_DIR / "hardcore_best_vec_normalize.pkl")
            torch.save(actor.state_dict(), CKPT_DIR / "hardcore_actor_weights.pt")
            np.savez(CKPT_DIR / "hardcore_normalization.npz", mean=mean, var=var)

            # 2. Guardar en agente/
            AGENTE_DIR.mkdir(exist_ok=True)
            torch.save(actor.state_dict(), AGENTE_DIR / "actor_weights.pt")
            np.savez(AGENTE_DIR / "normalization.npz", mean=mean, var=var)

            # 3. Re-generar agente.zip inmediatamente
            update_agente_zip()

        return True


def train_hardcore():
    print("="*65)
    print("INICIANDO ENTRENAMIENTO EN MODO HARDCORE / HARDCORE-PLUS")
    print("="*65)
    print("Configuración de los 8 entornos:", ENV_CONFIGS)
    
    env = SubprocVecEnv([make_env(p, diff) for p, diff in ENV_CONFIGS])
    
    # Cargar normalizador previo si existe
    norm_path = CKPT_DIR / "best_vec_normalize.pkl"
    if norm_path.exists():
        print(f"Cargando estadísticas previas de normalización desde {norm_path.name}...")
        env = VecNormalize.load(str(norm_path), env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Cargar modelo base previo
    model_path = CKPT_DIR / "best_sb3_model.zip"
    if model_path.exists():
        print(f"Cargando modelo PPO preentrenado desde {model_path.name}...")
        model = PPO.load(
            str(model_path),
            env=env,
            device="cpu",
            learning_rate=lambda _: 1e-4,
            clip_range=lambda _: 0.15,
            ent_coef=0.003
        )
    else:
        print("Iniciando nuevo modelo PPO...")
        model = PPO(
            "MlpPolicy", env,
            n_steps=1024, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.15,
            ent_coef=0.003, learning_rate=1e-4,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh),
            device="cpu", verbose=1
        )

    # Evaluar baseline antes de entrenar
    initial_mean = env.obs_rms.mean.copy()
    initial_var = env.obs_rms.var.copy()
    initial_actor = extract_actor(model)
    init_wins, total_evals, init_t, init_dist = evaluate_hardcore_and_val(initial_actor, initial_mean, initial_var)
    print(f"\nBaseline inicial (10 pistas HC + Val): Llegadas {init_wins}/{total_evals} | Tiempo: {init_t:.2f}s | Dist: {init_dist:.2f}m")

    callback = HardcoreSaveBestCallback(eval_freq=32000, best_score=(init_wins, init_t, init_dist), verbose=1)

    # 1,500,000 pasos en hardcore (~12-14 minutos en esta máquina)
    total_steps = 1500000
    print(f"\nComenzando entrenamiento de {total_steps:,} pasos...")
    model.learn(total_timesteps=total_steps, callback=callback)
    env.close()

    print("\n" + "="*65)
    print("¡ENTRENAMIENTO HARDCORE FINALIZADO!")
    print(f"Mejor marca alcanzada: {callback.best_wins}/{total_evals} llegadas, {callback.best_time:.2f}s, {callback.best_dist:.2f}m")
    print("="*65)


if __name__ == "__main__":
    train_hardcore()
