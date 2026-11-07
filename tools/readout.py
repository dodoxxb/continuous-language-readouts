"""Apply an ELF, AURORA or LangFlow fitted affine probe to saved states.

The probe predicts native token IDs at supplied positions. It does not generate
states, determine the valid lexical mask, stop generation, or establish quality.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Fitted affine state dict, loaded with weights_only=True.')
    parser.add_argument('--states', type=Path, required=True, help='Numpy .npy array of shape (..., feature_dim).')
    parser.add_argument('--output', type=Path, required=True, help='Numpy .npy token-ID array of shape (...).')
    parser.add_argument('--device', default='cpu', help='Torch device (default: cpu).')
    parser.add_argument('--chunk-size', type=int, default=128, help='Maximum positions per forward pass (default: 128).')
    parser.add_argument('--threads', type=int, default=2, help='PyTorch CPU threads (default: 2).')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.chunk_size < 1 or args.threads < 1:
        parser.error('chunk size and thread count must be positive')
    if args.output.resolve() in {args.states.resolve(), args.checkpoint.resolve()}:
        parser.error('output must not replace an input')
    if args.output.exists() and not args.overwrite:
        parser.error('output exists; use --overwrite to replace it')
    if not args.output.parent.is_dir():
        parser.error('output parent directory does not exist')

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or set(state) != {'weight', 'bias'}:
        raise ValueError('Expected a fitted affine checkpoint with exactly weight and bias; adapters use a different format.')
    weight, bias = state['weight'], state['bias']
    if weight.ndim != 2 or bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError('Invalid affine checkpoint dimensions')
    x = np.load(args.states, mmap_mode='r', allow_pickle=False)
    if x.ndim < 2 or x.shape[-1] != weight.shape[1] or not np.issubdtype(x.dtype, np.floating):
        raise ValueError(f'States must be a real floating-point array of shape (..., {weight.shape[1]}); got {x.shape} and {x.dtype}')
    if not x.flags.c_contiguous or any(dim == 0 for dim in x.shape):
        raise ValueError('States must be a nonempty, C-contiguous .npy array')
    flat = x.reshape(-1, x.shape[-1])
    device = torch.device(args.device)
    w = weight.to(device=device, dtype=torch.float32)
    b = bias.to(device=device, dtype=torch.float32)
    fd, tmpname = tempfile.mkstemp(prefix=f'.{args.output.name}.', suffix='.tmp', dir=args.output.parent)
    os.close(fd)
    try:
        out = np.lib.format.open_memmap(tmpname, mode='w+', dtype=np.int32, shape=x.shape[:-1])
        y = out.reshape(-1)
        with torch.inference_mode():
            for start in range(0, len(flat), args.chunk_size):
                batch = np.array(flat[start:start + args.chunk_size], dtype=np.float32, copy=True)
                if not np.isfinite(batch).all():
                    raise ValueError(f'Non-finite state at or after flattened position {start}')
                y[start:start + len(batch)] = F.linear(torch.from_numpy(batch).to(device), w, b).argmax(-1).cpu().numpy()
        out.flush()
        del y, out
        os.replace(tmpname, args.output)
    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)
    print(f'Saved {len(flat)} token decisions to {args.output}; apply the appropriate native mask and stopping protocol separately.')


if __name__ == '__main__':
    main()
