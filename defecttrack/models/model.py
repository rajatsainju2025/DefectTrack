"""Model construction and checkpoint loading."""

import logging

import torch

from .hrnet import get_pose_net as get_hrnet

logger = logging.getLogger(__name__)

_MODEL_FACTORY = {
    "hrnet": get_hrnet,
}


def create_model(arch, heads, head_conv=256):
    """Build a model from an ``<name>_<depth>`` string, e.g. ``hrnet_18``."""
    name, _, depth = arch.partition("_")
    if name not in _MODEL_FACTORY:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Choose from: {', '.join(_MODEL_FACTORY)}")
    num_layers = int(depth) if depth else 0
    return _MODEL_FACTORY[name](num_layers=num_layers, heads=heads, head_conv=head_conv)


def load_model(model, model_path):
    """Load weights from a checkpoint, skipping any mismatched tensors."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip the "module." prefix left by nn.DataParallel.
    state_dict = {
        (k[7:] if k.startswith("module") and not k.startswith("module_list") else k): v
        for k, v in state_dict.items()
    }

    model_state = model.state_dict()
    for k in list(state_dict):
        if k not in model_state:
            logger.warning("Dropping unexpected parameter %s", k)
            del state_dict[k]
        elif state_dict[k].shape != model_state[k].shape:
            logger.warning("Skipping %s: checkpoint shape %s != model shape %s",
                           k, tuple(state_dict[k].shape), tuple(model_state[k].shape))
            state_dict[k] = model_state[k]
    missing = [k for k in model_state if k not in state_dict]
    if missing:
        logger.warning("%d parameters missing from checkpoint (check --arch)", len(missing))

    model.load_state_dict(state_dict, strict=False)
    return model
