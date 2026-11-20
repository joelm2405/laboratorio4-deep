import numpy as np
import torch
from pathlib import Path
from walker_lab.environment import WalkerEnv
from walker_lab.scenarios import load_scenario
from agente.agent import Agent

ROOT = Path(__file__).resolve().parent
agent = Agent(model_dir=ROOT / "agente", device="cpu")

def test_group(track_files, n_seeds=6, max_steps=None):
    total_runs = len(track_files) * n_seeds
    wins = 0
    dists = []
    times = []
    for f in track_files:
        base_scenario = load_scenario(f)
        for s_idx in range(n_seeds):
            seed = base_scenario.get("simulation_seed", 0) + s_idx * 17
            env = WalkerEnv(scenario=f)
            obs, info = env.reset(seed=seed)
            agent.reset()
            done = False
            while not done:
                action = agent.act(obs)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
            env.close()
            dists.append(info["distance"])
            times.append(info["simulated_seconds"])
            if info["success"]:
                wins += 1
    return wins, total_runs, min(dists), max(dists), np.mean(dists), np.mean(times)

if __name__ == "__main__":
    val_tracks = sorted((ROOT / "escenarios" / "validacion").glob("maraton_*.json"))
    hc_tracks = sorted((ROOT / "escenarios" / "hardcore").glob("maraton_*.json"))
    
    print("Midiendo validacion (01-05)...")
    val_wins, val_total, val_min, val_max, val_mean, val_t = test_group(val_tracks, n_seeds=6)
    
    print("Midiendo hardcore (01-05)...")
    hc_wins, hc_total, hc_min, hc_max, hc_mean, hc_t = test_group(hc_tracks, n_seeds=6)
    
    print("Midiendo practica...")
    practica_dir = ROOT / "escenarios" / "practica"
    basic_tracks = [practica_dir / f"{name}.json" for name in ["plano", "suave", "pendientes", "escalones", "mixto"]]
    b_wins, b_total, b_min, b_max, b_mean, b_t = test_group(basic_tracks, n_seeds=6)
    
    bloques_wins, _, _, _, bloques_dist, _ = test_group([practica_dir / "bloques.json"], n_seeds=6)
    huecos_wins, _, _, _, huecos_dist, _ = test_group([practica_dir / "huecos.json"], n_seeds=6)
    
    print("\n" + "="*70)
    print("TABLA COMPARATIVA DE RENDIMIENTO MEDIDO (6 semillas por pista)")
    print("="*70)
    print(f"Validación 01-05:   Llegadas: {val_wins}/{val_total} | Rango: {val_min:.1f}-{val_max:.1f} m | Media: {val_mean:.1f} m | Tiempo prom: {val_t:.2f}s")
    print(f"Hardcore 01-05:     Llegadas: {hc_wins}/{hc_total} | Rango: {hc_min:.1f}-{hc_max:.1f} m | Media: {hc_mean:.1f} m | Tiempo prom: {hc_t:.2f}s")
    print(f"Práctica (5 base):  Llegadas: {b_wins}/{b_total} | Media: {b_mean:.1f}/60 m | Tiempo prom: {b_t:.2f}s")
    print(f"Práctica bloques:   Llegadas: {bloques_wins}/6 | Distancia media: {bloques_dist:.1f}/60 m")
    print(f"Práctica huecos:    Llegadas: {huecos_wins}/6 | Distancia media: {huecos_dist:.1f}/60 m")
    print("="*70)
