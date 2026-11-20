from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from train_curriculum import CKPT_DIR, SaveBest, make_env, make_vec_env

# mas entornos con bloques y huecos, que es donde mas se cae
PROFILES = ["bloques", "huecos", "bloques", "huecos", "mixto", "maraton", "mixto", "maraton"]


def finetune():
    norm_path = CKPT_DIR / "best_vec_normalize.pkl"
    if norm_path.exists():
        env = VecNormalize.load(norm_path, SubprocVecEnv([make_env(p) for p in PROFILES]))
        env.training = True
        env.norm_reward = False
    else:
        env = make_vec_env(PROFILES)

    model = PPO.load(
        CKPT_DIR / "best_sb3_model.zip", env=env, device="cpu",
        learning_rate=lambda _: 8e-5, clip_range=lambda _: 0.15, ent_coef=0.002,
    )
    # solo guarda si supera al modelo base (1/5 llegadas en 84.08 s)
    callback = SaveBest(eval_freq=24000, best=(1, 84.08, 177.9), tag="finetuned")
    model.learn(400_000, callback=callback)
    env.close()


if __name__ == "__main__":
    finetune()
