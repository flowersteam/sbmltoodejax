"""Plot exported trajectories: python -m validation.plot_comparison outputs/oracles."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, case, title in zip(axes, ["time_dependent", "state_dependent"],
                                ["Time-dependent rate: k = 1 + t", "Feedback rate: k = A"]):
        data = np.genfromtxt(args.directory / f"{case}_10.csv", delimiter=",", names=True)
        for name, label, style in [
            ("exact", "Exact solution", "k-"),
            ("roadrunner", "libRoadRunner", "o"),
            ("copasi", "COPASI", "x"),
            ("jax", "Corrected JAX", "+"),
            ("previous_sample_and_hold", "Previous sampled rule", "--"),
        ]:
            ax.plot(data["time"], data[f"{name}_A"], style, label=label, markersize=6)
        ax.set(title=title, xlabel="Time (s)", ylabel="A amount (mol)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Assignment rules evaluated continuously; output interval 0.1 s")
    output = args.output or args.directory / "comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
