## Training Collapse After Warm-up — What Actually Happened

During BERT training, everything looked fine before warm-up finished, but right after that the estimated training time exploded (from ~16 hours to ~3000+ hours).
This was not a bug in BERT and not normal slowdown — it was a system-level failure.

---

### What caused it

- Warm-up ended → learning rate reached its peak
    - Optimizer updates became much heavier
    - FP16 + AdamW became numerically unstable
- FP16 instability triggered extra safety mechanisms
    - Gradient scaling
    - Step skipping
    - GPU ↔ CPU synchronization
    > These dramatically increased step time.
- GPU power logging inside the training loop
    - NVML calls (pynvml) force GPU synchronization
    - This blocks CUDA execution
    - Cost was small early, but catastrophic once steps became heavier
- All of this happened at the same time
    - LR peak
    - FP16 pressure
    - Optimizer cost spike
    - GPU sync from callbacks

This pushed training into a broken compute regime where each step became extremely slow.

---

### Why it never recovered

- Nothing automatically reduces optimizer cost
- FP16 overflow handling does not “heal itself”
- GPU synchronization overhead persists every step
- ETA keeps increasing because per-step time stays huge

Once crossed, training will not speed up again unless stopped and reconfigured.

---

### Key lessons

- Throughput is a first-class metric, not just loss
- Warm-up completion is a phase transition, not a formality
- Heavy logging / telemetry inside the training loop can destroy performance
- FP16 + AdamW + high effective batch on consumer GPUs is fragile
- If ETA jumps massively → stop immediately

---

### Mental model to remember

The model didn’t fail — the training entered a hardware-unstable regime.

---