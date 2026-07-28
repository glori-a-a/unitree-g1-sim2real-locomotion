# Unitree G1 Sim2Sim-to-Sim2Real Locomotion

> **Status: Active development**  
> Velocity-tracking locomotion for Unitree G1 using Isaac Lab, RSL-RL/PPO, ONNX, MuJoCo, and Unitree SDK2 LowCmd.

## Overview

This repository develops an end-to-end locomotion deployment pipeline for the Unitree G1 humanoid robot:

1. train a velocity-tracking policy in Isaac Lab with RSL-RL/PPO;
2. export the trained policy to ONNX;
3. validate policy consistency in MuJoCo as a cross-engine Sim2Sim stage;
4. deploy the same low-level policy through Unitree SDK2 `LowCmd` for closed-loop hardware control.

The project intentionally avoids Unitree's high-level locomotion API. The deployment target is low-level joint command execution with explicit observation construction, action scaling, joint mapping, safety limits, and control-loop timing.

## Technical Pipeline

```text
Isaac Lab
   ↓
RSL-RL / PPO velocity-tracking policy
   ↓
PyTorch checkpoint
   ↓
ONNX export
   ↓
MuJoCo Sim2Sim validation
   ↓
Unitree SDK2 LowCmd
   ↓
G1 low-level closed-loop deployment
```

## Current Progress

- [x] Repository architecture and deployment interface design
- [x] Initial velocity-tracking configuration
- [x] ONNX export entry point
- [x] MuJoCo inference-loop scaffold
- [ ] Isaac Lab environment integration
- [ ] PPO training and checkpoint selection
- [ ] Cross-engine observation/action alignment
- [ ] Sim2Sim quantitative validation
- [ ] Unitree SDK2 LowCmd integration
- [ ] G1 hardware validation

## Repository Structure

```text
.
├── configs/
│   └── g1_velocity_tracking.yaml
├── docs/
│   └── architecture.md
├── scripts/
│   └── export_onnx.py
├── sim2sim/
│   └── mujoco_runner.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Core Design

### Observation

The deployment observation vector is designed to contain:

- base angular velocity;
- projected gravity;
- commanded linear and yaw velocity;
- normalized joint positions;
- joint velocities;
- previous policy action.

The final ordering and normalization parameters must remain identical across Isaac Lab, MuJoCo, and the real robot.

### Action

The policy outputs normalized joint-position targets. These are scaled and added to the default standing pose before being passed to a low-level PD controller.

```text
q_target = q_default + action_scale × policy_action
```

### Sim2Sim Validation

MuJoCo is used to verify that the exported ONNX policy preserves stable behavior outside the training engine. Validation focuses on:

- observation ordering and normalization;
- joint-index mapping;
- control frequency and decimation;
- action clipping and scaling;
- velocity-command response;
- base stability and joint-limit violations.

## Planned Evaluation

- commanded vs. measured linear velocity;
- commanded vs. measured yaw rate;
- base roll/pitch stability;
- joint-limit violation count;
- policy inference latency;
- Isaac Lab vs. MuJoCo trajectory consistency.

## Environment

Expected environment:

- Ubuntu 22.04
- Python 3.10+
- Isaac Lab
- RSL-RL
- PyTorch
- ONNX / ONNX Runtime
- MuJoCo
- Unitree SDK2

Install the lightweight Python dependencies used by the current scaffold:

```bash
pip install -r requirements.txt
```

## Usage

Export a trained PyTorch policy to ONNX:

```bash
python scripts/export_onnx.py \
  --checkpoint path/to/policy.pt \
  --output artifacts/g1_policy.onnx \
  --obs-dim 96 \
  --action-dim 23
```

Run the MuJoCo Sim2Sim scaffold:

```bash
python sim2sim/mujoco_runner.py \
  --model path/to/g1.xml \
  --policy artifacts/g1_policy.onnx \
  --config configs/g1_velocity_tracking.yaml
```

## Notes

This repository is under active development. Initial files define the software architecture and deployment contracts; trained checkpoints, validated robot parameters, quantitative results, and hardware demonstrations will be added as each stage is completed.

## License

This project is intended for research and educational use. Third-party robot assets, SDKs, and model files remain subject to their original licenses.
