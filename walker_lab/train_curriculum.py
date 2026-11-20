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
VAL_DIR = ROOT / "escenarios" / "validacion"
N_ENVS = 8


def make_env(profile):
    return lambda: WalkerEnv(profile=profile)


def make_vec_env(profiles, rms=None):
    env = VecNormalize(SubprocVecEnv([make_env(p) for p in profiles]),
                       norm_obs=True, norm_reward=False, clip_obs=10.0)
    if rms is not None:
        env.obs_rms.mean, env.obs_rms.var = rms
    return env


def extract_actor(model):
    # copia el actor de SB3 a PolicyNetwork para poder usarlo sin SB3
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


def run_validation(actor, mean, var):
    std = np.sqrt(var + 1e-8)
    tracks = sorted(VAL_DIR.glob("maraton_*.json"))
    wins, dists, times, win_times = 0, [], [], []

    for track in tracks:
        env = WalkerEnv(scenario=track)
        obs, _ = env.reset(seed=env.fixed_scenario["simulation_seed"])
        done = False
        while not done:
            x = np.clip((obs - mean) / std, -10, 10).astype(np.float32)
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

    avg_time = np.mean(win_times) if win_times else np.mean(times)
    return wins, avg_time, np.mean(dists)


class SaveBest(BaseCallback):
    """Cada eval_freq pasos corre las 5 maratones y guarda si mejora el ranking."""

    def __init__(self, eval_freq=32000, best=(-1, float("inf"), -1.0), tag="best"):
        super().__init__()
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.best_wins, self.best_time, self.best_dist = best
        self.tag = tag

    def is_better(self, wins, t, dist):
        # llegadas > menor tiempo > mayor distancia
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
        wins, t, dist = run_validation(actor, mean, var)
        print(f"[{self.num_timesteps}] llegadas {wins}/5  tiempo {t:.2f}s  dist {dist:.2f}m", flush=True)

        if self.is_better(wins, t, dist):
            self.best_wins, self.best_time, self.best_dist = wins, t, dist
            print("  guardando modelo", flush=True)
            AGENTE_DIR.mkdir(exist_ok=True)
            np.savez(AGENTE_DIR / "normalization.npz", mean=mean, var=var)
            torch.save(actor.state_dict(), AGENTE_DIR / "actor_weights.pt")
            self.model.save(CKPT_DIR / f"{self.tag}_sb3_model")
            self.training_env.save(CKPT_DIR / f"{self.tag}_vec_normalize.pkl")
        return True


def train():
    CKPT_DIR.mkdir(exist_ok=True)
    callback = SaveBest(eval_freq=32000)

    # etapa 1: aprender a caminar en terreno suave
    env = make_vec_env(["suave"] * N_ENVS)
    model = PPO(
        "MlpPolicy", env,
        n_steps=1024, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh),
        device="cpu", verbose=1,
    )
    model.learn(600_000, callback=callback, reset_num_timesteps=False)
    rms = (env.obs_rms.mean.copy(), env.obs_rms.var.copy())
    env.close()

    # etapa 2: pista mixta (escalones, pendientes, bloques, huecos)
    env = make_vec_env(["mixto"] * N_ENVS, rms)
    model.set_env(env)
    model.lr_schedule = lambda _: 2e-4
    model.ent_coef = 0.002
    model.learn(800_000, callback=callback, reset_num_timesteps=False)
    rms = (env.obs_rms.mean.copy(), env.obs_rms.var.copy())
    env.close()

    # etapa 3: maratones completas de 240 m
    env = make_vec_env(["maraton"] * N_ENVS, rms)
    model.set_env(env)
    model.lr_schedule = lambda _: 1e-4
    model.ent_coef = 0.001
    model.clip_range = lambda _: 0.15
    model.learn(1_000_000, callback=callback, reset_num_timesteps=False)
    env.close()

    print(f"mejor: {callback.best_wins}/5 llegadas, {callback.best_time:.2f}s, {callback.best_dist:.2f}m")


if __name__ == "__main__":
    train()
