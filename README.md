# Laboratorio 4: Reinforcement Learning (BipedalWalker)

Solución desarrollada para el Laboratorio 4 de Deep Learning (UTEC 2026-2).
Este proyecto implementa y entrena un agente bípedo de control continuo utilizando **Proximal Policy Optimization (PPO)** con **Curriculum Learning** sobre pistas procedurales del entorno Box2D `WalkerEnv`.

---

## 1. Arquitectura del Agente

* **Espacio de Acciones (4D continuo en $[-1, 1]$):**
  * Motores de cadera 1, rodilla 1, cadera 2 y rodilla 2.
* **Espacio de Observaciones (24D continuo):**
  * Ángulo y velocidad angular del torso.
  * Velocidades lineales horizontal y vertical.
  * Ángulos y velocidades de las 4 articulaciones.
  * Sensores de contacto de las piernas con el suelo.
  * 10 haces de lidar que detectan la elevación y obstáculos del terreno próximo.
* **Red Neuronal (Actor):**
  * Capa de entrada: 24 neuronas con normalización adaptativa (`RunningMeanStd`).
  * Capas ocultas: MLP `[256, 256]` con activaciones `Tanh`.
  * Capa de salida: 4 neuronas con acotamiento `clamp(-1.0, 1.0)`.
  * Inferencia pura en PyTorch sin dependencias pesadas en tiempo de ejecución.

---

## 2. Estrategia de Entrenamiento (Curriculum Learning)

Para superar el reto de las pistas maratónicas con relieve variado y obstáculos:
1. **Fase 1 (Locomoción y balance en `suave`):** 600,000 pasos. El agente aprende el balance vertical y el paso coordinado bípedo.
2. **Fase 2 (Terreno variado en `mixto`):** 800,000 pasos. El agente aprende a utilizar las lecturas del lidar para superar pendientes y escalones.
3. **Fase 3 (Resistencia en `maraton`):** 1,000,000 pasos. Recorridos de 240 metros con generación procedural de obstáculos.
4. **Fase 4 (Ajuste fino en obstáculos):** Refinamiento para optimizar velocidad y tiempo de carrera.

El sistema monitorizó en tiempo real las 5 pistas oficiales de maratón (`maraton_01.json` a `maraton_05.json`) y conservó el punto de control óptimo según las reglas de la competencia:
1. Mayor número de llegadas a la meta.
2. Menor tiempo promedio en llegadas.
3. Mayor distancia promedio acumulada.

---

## 3. Resultados de Evaluación

### Pruebas de Práctica (`escenarios/practica/`):
* `plano.json`: **60.00 m / 21.22 s** (Llegada a la meta - 100%)
* `suave.json`: **60.00 m / 20.90 s** (Llegada a la meta - 100%)
* `pendientes.json`: **60.00 m / 21.70 s** (Llegada a la meta - 100%)
* `escalones.json`: **60.00 m / 22.44 s** (Llegada a la meta - 100%)
* `mixto.json`: **60.00 m / 21.34 s** (Llegada a la meta - 100%)

### Pistas de Maratón (`escenarios/validacion/`):
* `maraton_01.json`: **164.92 m / 52.82 s**
* `maraton_02.json`: **173.12 m / 54.44 s**
* `maraton_03.json`: **157.81 m / 49.94 s**
* `maraton_04.json`: **114.94 m / 38.58 s**
* `maraton_05.json`: **143.46 m / 46.56 s**

---

## 4. Instrucciones de Ejecución

Para evaluar el modelo entrenado contenido en `agente/`:

```bash
cd walker_lab
python evaluar.py
```

Para inferencia acelerada por GPU:
```bash
python evaluar.py --device cuda
```

Para visualizar la carrera del robot en tiempo real:
```bash
python evaluar.py --render
```

Para evaluar cualquier otra pista:
```bash
python evaluar.py --scenario escenarios/validacion/maraton_02.json
```

---

## 5. Estructura de la Carpeta de Entrega (`agente/`)

* `agente/agent.py`: Adaptador con la clase `Agent(BaseAgent)`, implementando `__init__`, `reset` y `act(observation)`.
* `agente/architecture.py`: Definición de la red `PolicyNetwork` en PyTorch.
* `agente/actor_weights.pt`: Pesos entrenados del modelo.
* `agente/normalization.npz`: Medias y varianzas del normalizador de estados.
* `agente/requirements.txt`: Dependencias mínimas requeridas (`torch>=2.0.0`, `numpy>=1.20.0`).
