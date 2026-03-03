#!/usr/bin/env python3
'''
Simple line plot y=x with matplotlib.
Saves figure to figure/ directory with configurable filename.
'''
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main(save_filename):
    # create figure directory if it doesn't exist
    figure_dir = Path('figure')
    figure_dir.mkdir(exist_ok=True)

    # generate data
    x = np.linspace(0, 10, 100)
    y = x

    # create plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='y = x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Function: y = x')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # save figure
    output_path = figure_dir / save_filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot y=x and save to file')
    parser.add_argument(
        '--save',
        type=str,
        required=True,
        help='Filename to save under figure/ directory (e.g., plot.png)'
    )
    args = parser.parse_args()

    main(args.save)
