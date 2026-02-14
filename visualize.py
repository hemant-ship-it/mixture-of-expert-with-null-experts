import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_telemetry(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def plot_training_loss(data: list[dict], output_dir: str):
    """Plot total loss, LM loss, balance loss, and z-loss over steps."""
    steps = [d["step"] for d in data]
    total = [d["total_loss"] for d in data]
    lm = [d["lm_loss"] for d in data]
    bal = [d["balance_loss"] for d in data]
    zl = [d["z_loss"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, total, label="Total Loss", linewidth=1.5, alpha=0.9)
    ax.plot(steps, lm, label="LM Loss", linewidth=1.5, alpha=0.9)
    ax.plot(steps, bal, label="Balance Loss", linewidth=1, alpha=0.7)
    ax.plot(steps, zl, label="Z-Loss", linewidth=1, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.show()
    plt.close(fig)


def plot_perplexity(data: list[dict], output_dir: str):
    """Plot perplexity over training steps."""
    ppl_data = [(d["step"], d["perplexity"]) for d in data if "perplexity" in d]
    if not ppl_data:
        print("No perplexity data found, skipping plot.")
        return

    steps, ppls = zip(*ppl_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, ppls, marker="o", markersize=4, linewidth=1.5, color="tab:red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity Over Training")
    ax.grid(True, alpha=0.3)
    if max(ppls) > 10 * min(ppls):
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "perplexity.png"), dpi=150)
    plt.show()
    plt.close(fig)


def plot_expert_heatmap(data: list[dict], output_dir: str):
    """Heatmap of expert utilization over time (steps x experts)."""
    entries = [(d["step"], d["expert_counts"]) for d in data if d.get("expert_counts")]
    if not entries:
        print("No expert count data found, skipping heatmap.")
        return

    steps, counts = zip(*entries)
    n_experts = len(counts[0])
    matrix = np.array(counts)  # (n_steps, n_experts)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Expert")
    ax.set_yticks(range(n_experts))
    ax.set_yticklabels([f"E{i}" for i in range(n_experts)])
    ax.set_title("Expert Utilization Heatmap")

    # Set x-tick labels to actual step numbers (sparse)
    n_ticks = min(10, len(steps))
    tick_positions = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(steps[i]) for i in tick_positions])

    fig.colorbar(im, ax=ax, label="Token Count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "expert_heatmap.png"), dpi=150)
    plt.show()
    plt.close(fig)


def plot_null_routing(data: list[dict], output_dir: str):
    """Line plot of null routing ratio over time."""
    steps = [d["step"] for d in data]
    null_ratios = [d["null_ratio"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [r * 100 for r in null_ratios], linewidth=1.5, color="tab:purple")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Target ρ=0.5")
    ax.set_xlabel("Step")
    ax.set_ylabel("Null Routing (%)")
    ax.set_title("Null Routing Ratio Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "null_routing.png"), dpi=150)
    plt.show()
    plt.close(fig)


def plot_expert_token_distribution(data: list[dict], output_dir: str):
    """Bar chart of total tokens per expert across training."""
    entries = [d["expert_counts"] for d in data if d.get("expert_counts")]
    if not entries:
        print("No expert count data found, skipping bar chart.")
        return

    totals = np.array(entries).sum(axis=0)
    n_experts = len(totals)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(n_experts), totals, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Total Tokens Processed")
    ax.set_title("Per-Expert Token Distribution (Cumulative)")
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "expert_token_dist.png"), dpi=150)
    plt.show()
    plt.close(fig)


def plot_zero_compute(data: list[dict], output_dir: str):
    """Plot fraction of zero-compute tokens over time."""
    steps = [d["step"] for d in data]
    zc = [d["zero_compute_ratio"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [r * 100 for r in zc], linewidth=1.5, color="tab:orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("Zero-Compute Tokens (%)")
    ax.set_title("Zero-Compute Token Ratio Over Training")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "zero_compute.png"), dpi=150)
    plt.show()
    plt.close(fig)


def plot_gate_weights(data: list[dict], output_dir: str):
    """Average gate weights per expert (final snapshot and over time)."""
    entries = [(d["step"], d["gate_weights"]) for d in data if d.get("gate_weights")]
    if not entries:
        print("No gate weight data found, skipping plot.")
        return

    steps, weights = zip(*entries)
    n_experts = len(weights[0])

    # Bar chart of final gate weights
    final_weights = weights[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: final snapshot bar chart
    ax = axes[0]
    ax.bar(range(n_experts), final_weights, color="tab:green", alpha=0.8)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Average Gate Weight")
    ax.set_title(f"Gate Weights at Step {steps[-1]}")
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax.grid(True, alpha=0.3, axis="y")

    # Right: gate weights over time (per expert)
    ax = axes[1]
    weight_matrix = np.array(weights)  # (n_steps, n_experts)
    for e in range(n_experts):
        ax.plot(list(steps), weight_matrix[:, e], label=f"E{e}", linewidth=1, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Gate Weight")
    ax.set_title("Gate Weights Over Training")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gate_weights.png"), dpi=150)
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize MoE Null Expert telemetry")
    parser.add_argument("--input", type=str, default="telemetry.json",
                        help="Path to telemetry JSON file")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save plot PNGs")
    args = parser.parse_args()

    print(f"Loading telemetry from {args.input}...")
    data = load_telemetry(args.input)
    print(f"Loaded {len(data)} steps of telemetry data.")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Plotting training loss curves...")
    plot_training_loss(data, args.output_dir)

    print("Plotting perplexity...")
    plot_perplexity(data, args.output_dir)

    print("Plotting expert utilization heatmap...")
    plot_expert_heatmap(data, args.output_dir)

    print("Plotting null routing ratio...")
    plot_null_routing(data, args.output_dir)

    print("Plotting expert token distribution...")
    plot_expert_token_distribution(data, args.output_dir)

    print("Plotting zero-compute token ratio...")
    plot_zero_compute(data, args.output_dir)

    print("Plotting gate weight distributions...")
    plot_gate_weights(data, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
