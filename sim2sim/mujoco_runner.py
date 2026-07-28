#!/usr/bin/env python3
"""Run an exported G1 locomotion policy in MuJoCo.

This is an integration scaffold. Joint ordering, default pose, observation
construction, actuator mapping, and gains must be replaced with values verified
against the selected Isaac Lab task and MuJoCo G1 asset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ControlConfig:
    simulation_frequency_hz: int
    policy_frequency_hz: int
    action_scale: float
    action_clip: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    return parser.parse_args()


def load_config(path: Path) -> ControlConfig:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    control = raw["control"]
    return ControlConfig(
        simulation_frequency_hz=int(control["simulation_frequency_hz"]),
        policy_frequency_hz=int(control["policy_frequency_hz"]),
        action_scale=float(control["action_scale"]),
        action_clip=float(control["action_clip"]),
    )


def build_observation(
    qpos: np.ndarray,
    qvel: np.ndarray,
    previous_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    """Construct a placeholder observation vector.

    Replace this function with the exact Isaac Lab observation order,
    normalization, frame conventions, and joint subset.
    """
    joint_position = qpos[7:].astype(np.float32, copy=False)
    joint_velocity = qvel[6:].astype(np.float32, copy=False)
    base_angular_velocity = qvel[3:6].astype(np.float32, copy=False)
    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return np.concatenate(
        [
            base_angular_velocity,
            projected_gravity,
            command.astype(np.float32, copy=False),
            joint_position,
            joint_velocity,
            previous_action,
        ]
    )[None, :]


def main() -> None:
    args = parse_args()
    for path in (args.model, args.policy, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.duration <= 0:
        raise ValueError("--duration must be positive.")

    import mujoco
    import onnxruntime as ort

    cfg = load_config(args.config)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / cfg.simulation_frequency_hz

    session = ort.InferenceSession(
        str(args.policy), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    action_dim = int(session.get_outputs()[0].shape[-1])
    previous_action = np.zeros(action_dim, dtype=np.float32)
    velocity_command = np.array([0.4, 0.0, 0.0], dtype=np.float32)

    policy_decimation = round(
        cfg.simulation_frequency_hz / cfg.policy_frequency_hz
    )
    if policy_decimation <= 0:
        raise ValueError("Invalid simulation/policy frequency combination.")

    total_steps = int(args.duration * cfg.simulation_frequency_hz)
    for step in range(total_steps):
        if step % policy_decimation == 0:
            observation = build_observation(
                data.qpos, data.qvel, previous_action, velocity_command
            )
            expected_dim = session.get_inputs()[0].shape[-1]
            if isinstance(expected_dim, int) and observation.shape[-1] != expected_dim:
                raise ValueError(
                    f"Observation dimension mismatch: built {observation.shape[-1]}, "
                    f"policy expects {expected_dim}. Align build_observation() with training."
                )
            action = session.run([output_name], {input_name: observation})[0][0]
            previous_action = np.clip(
                action, -cfg.action_clip, cfg.action_clip
            ).astype(np.float32)

        # Placeholder actuator mapping. Replace with verified G1 actuator indices,
        # default joint pose, action scale, and low-level PD target conversion.
        control_count = min(model.nu, previous_action.size)
        data.ctrl[:control_count] = (
            cfg.action_scale * previous_action[:control_count]
        )
        mujoco.mj_step(model, data)

    print(
        f"Completed {args.duration:.1f}s MuJoCo rollout. "
        "Validate state/action mapping before interpreting locomotion results."
    )


if __name__ == "__main__":
    main()
