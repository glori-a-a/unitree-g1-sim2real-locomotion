# System Architecture

## 1. Training Stage

The velocity-tracking policy is trained in Isaac Lab using RSL-RL/PPO. The task provides commanded planar velocity and yaw rate and optimizes tracking performance while maintaining balance, smooth actions, and physically valid joint behavior.

Key interfaces that must be frozen before deployment:

- observation order and dimensions;
- observation normalization;
- action order and scaling;
- default joint pose;
- policy frequency and simulator decimation;
- G1 joint names and indices.

## 2. Policy Export

The deterministic actor is exported from PyTorch to ONNX. The exported graph accepts a batch of normalized observations and returns normalized joint-position actions.

Export validation should compare PyTorch and ONNX outputs using identical random and recorded observations. Numerical differences should be reported before beginning Sim2Sim testing.

## 3. MuJoCo Sim2Sim

MuJoCo reconstructs the deployment-time inference loop independently from Isaac Lab:

```text
MuJoCo state
   ↓
observation construction and normalization
   ↓
ONNX Runtime inference
   ↓
action clipping and scaling
   ↓
low-level PD joint targets
   ↓
MuJoCo actuators
```

The MuJoCo implementation must not silently depend on Isaac Lab task internals. This stage is intended to expose inconsistent coordinate frames, joint mapping, normalization, actuator semantics, and control timing before hardware execution.

## 4. G1 Hardware Deployment

The real-robot stage follows the same observation/action contract used by MuJoCo. Unitree SDK2 provides robot state feedback and low-level `LowCmd` publication. No high-level walking command is required.

The expected loop is:

```text
Unitree LowState
   ↓
state validation and observation construction
   ↓
ONNX Runtime inference
   ↓
action clipping, interpolation, and safety checks
   ↓
PD targets encoded into LowCmd
   ↓
G1 actuators
```

## 5. Safety Requirements

Before sending policy output to hardware:

- verify the robot variant and exact joint count;
- verify every joint index and sign convention;
- start from a supported standing pose;
- enforce joint-position, velocity, and torque limits;
- reject NaN or infinite observations/actions;
- provide an independent emergency stop;
- test with reduced action scale and conservative gains;
- log observations, actions, commands, and state feedback.

## 6. Validation Plan

The final project will compare:

- PyTorch and ONNX inference outputs;
- Isaac Lab and MuJoCo command tracking;
- base orientation and joint trajectories across engines;
- inference latency and control-loop jitter;
- simulated and real command-tracking behavior.

The quantitative thresholds and final results will be added after the trained checkpoint, MuJoCo asset, and G1 hardware configuration are fixed.
