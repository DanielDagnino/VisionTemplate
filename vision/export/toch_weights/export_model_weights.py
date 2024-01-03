import argparse
import os
import sys

import torch
from path import Path

from vision.utils.torch.dataparallel import add_module_dataparallel


def run(fn_in: Path, fn_out: Path):
    checkpoint = torch.load(fn_in, map_location='cpu')
    print(f'type(checkpoint) = {type(checkpoint)}')
    print(f'state_dict.keys() = {checkpoint.keys()}')
    state_dict = checkpoint['model']
    state_dict = add_module_dataparallel(state_dict)

    args = {
        "n_history": 28,
        "n_features_in": 9,
        "n_hidden_feat": [128, 1],
        "n_hidden_forecast": [128, 128],
        "verbose": False
    }

    checkpoint = dict(state_dict=state_dict, args=args)

    torch.save(checkpoint, fn_out)
    print(f'Model size = {os.path.getsize(fn_out) / 1024:.2f} Kb')


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn_in", type=Path, help="Input file checkpoint")
    parser.add_argument("fn_out", type=Path, help="Output file checkpoint")
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
