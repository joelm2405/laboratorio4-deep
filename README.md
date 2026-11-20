# Laboratorio 4 - BipedalWalker con PPO

Deep Learning, UTEC 2026-2.

Entrenamos el bípedo de `walker_lab` con PPO (stable-baselines3) y un currículo de pistas cada vez más difíciles. Para la entrega solo se usa el actor, exportado a una red de PyTorch sin depender de SB3.

## Modelo

- Entrada: las 24 observaciones del entorno, normalizadas con la media y varianza del `VecNormalize` del entrenamiento (`normalization.npz`), recortadas a [-10, 10].
- MLP 256-256 con Tanh.
- Salida: 4 acciones (caderas y rodillas) recortadas a [-1, 1]. En evaluación se usa la media de la política, sin muestrear.

## Entrenamiento

`train_curriculum.py`, 8 entornos en paralelo:

| Etapa | Perfil | Pasos | lr | ent_coef |
|---|---|---:|---:|---:|
| 1 | `suave` | 600k | 3e-4 | 0.005 |
| 2 | `mixto` | 800k | 2e-4 | 0.002 |
| 3 | `maraton` | 1M | 1e-4 | 0.001 |

Cada 32k pasos se corren las 5 maratones de `escenarios/validacion/` y se guarda el modelo si mejora según el criterio del ranking (llegadas, luego tiempo, luego distancia).

`finetune_obstacles.py` sigue desde el mejor checkpoint con más entornos de bloques y huecos.

## Resultados

Con `python evaluar.py` y la semilla de cada pista:

| Práctica (60 m) | Distancia | Tiempo |
|---|---:|---:|
| plano | 60.00 | 19.62 s |
| suave | 60.00 | 19.42 s |
| pendientes | 60.00 | 20.26 s |
| escalones | 60.00 | 19.04 s |
| mixto | 60.00 | 18.96 s |
| bloques | 22.37 (cae) | 6.84 s |
| huecos | 13.51 (cae) | 7.46 s |

| Maratón (240 m) | validacion | hardcore |
|---|---:|---:|
| 01 | 152.06 m | 161.03 m |
| 02 | 170.78 m | 111.76 m |
| 03 | 217.67 m | 124.84 m |
| 04 | 116.15 m | 152.29 m |
| 05 | 220.41 m | 150.95 m |

Todavía no termina ninguna maratón. Probando 6 semillas por pista, la distancia media va de unos 90 a 175 m. Casi todas las caídas son en bloques y huecos.

## Uso

```bash
cd walker_lab
python evaluar.py
python evaluar.py --render
python evaluar.py --scenario escenarios/validacion/maraton_02.json
```

Para entrenar hace falta además `stable-baselines3`:

```bash
python train_curriculum.py
```

## Entrega

`walker_lab/agente/` tiene `agent.py`, `architecture.py`, `actor_weights.pt`, `normalization.npz` y `requirements.txt`. El zip se arma desde `walker_lab/`:

```bash
python -c "import zipfile,pathlib; z=zipfile.ZipFile('agente.zip','w',zipfile.ZIP_DEFLATED); [z.write(p, p.as_posix()) for p in sorted(pathlib.Path('agente').glob('*')) if p.is_file()]; z.close()"
```
