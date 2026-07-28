#!/usr/bin/env python3
"""Export a PyTorch locomotion policy to ONNX.

The script accepts either a TorchScript module or a serialized nn.Module. Exact
checkpoint loading may need adaptation to the selected RSL-RL runner format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def load_policy(checkpoint: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for ONNX export.") from exc

    try:
        policy = torch.jit.load(str(checkpoint), map_location="cpu")
    except RuntimeError:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if hasattr(loaded, "eval"):
            policy = loaded
        elif isinstance(loaded, dict) and hasattr(loaded.get("policy"), "eval"):
            policy = loaded["policy"]
        else:
            raise ValueError(
                "Unsupported checkpoint format. Adapt load_policy() to the "
                "RSL-RL runner/checkpoint used by the training project."
            )
    return policy.eval()


def main() -> None:
    args = parse_args()
    if args.obs_dim <= 0 or args.action_dim <= 0:
        raise ValueError("Observation and action dimensions must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    import torch

    policy = load_policy(args.checkpoint)
    dummy_observation = torch.zeros(1, args.obs_dim, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        sample_output = policy(dummy_observation)
    if sample_output.shape[-1] != args.action_dim:
        raise ValueError(
            f"Policy output dimension {sample_output.shape[-1]} does not match "
            f"--action-dim={args.action_dim}."
        )

    torch.onnx.export(
        policy,
        dummy_observation,
        str(args.output),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported policy to {args.output}")


if __name__ == "__main__":
    main()
