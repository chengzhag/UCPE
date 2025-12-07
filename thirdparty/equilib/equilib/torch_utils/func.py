#!/usr/bin/env python3

import torch


def sizeof(tensor: torch.Tensor) -> float:
    """Get the size of a tensor"""
    assert torch.is_tensor(tensor), "ERR: is not tensor"
    return tensor.element_size() * tensor.nelement()


def get_device(a: torch.Tensor) -> torch.device:
    """Get device of a Tensor"""
    return torch.device(a.get_device() if a.get_device() >= 0 else "cpu")
