import os
import random
import numpy as np
import scipy.io
import torch


def load_digit_master(base_dir=None):
    """Load digit templates from digitMaster.mat."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'digitMaster.mat')
    if not os.path.exists(file_path):
        raise FileNotFoundError('digitMaster.mat not found: {}'.format(file_path))
    temp = scipy.io.loadmat(file_path)
    return temp['digitMaster']


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if state is None:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_checkpoint(path, rnn, optimizer=None, step=0, best_metric=None, hp=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model_state': rnn.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'step': int(step),
        'best_metric': None if best_metric is None else float(best_metric),
        'hp': hp,
        'rng_state': get_rng_state(),
    }
    torch.save(payload, path)


def _move_optimizer_state_to_device(optimizer, device):
    if optimizer is None or device is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def load_checkpoint(path, rnn, optimizer=None, load_optimizer=True, strict=True):
    # PyTorch >=2.6 defaults weights_only=True; our checkpoint contains python/numpy RNG state.
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions without weights_only argument.
        payload = torch.load(path, map_location='cpu')
    rnn.load_state_dict(payload['model_state'], strict=strict)
    if optimizer is not None and load_optimizer and payload.get('optimizer_state') is not None:
        optimizer.load_state_dict(payload['optimizer_state'])
        _move_optimizer_state_to_device(optimizer, getattr(rnn, 'device', None))
    set_rng_state(payload.get('rng_state'))
    return payload
