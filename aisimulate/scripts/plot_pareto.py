import json
import sys
import matplotlib.pyplot as plt
import pandas as pd

def plot_pareto(json_path, output_png):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    candidates = data.get("candidates", [])
    if not candidates:
        print(f"No candidates found in {json_path}")
        return

    records = []
    for idx, c in enumerate(candidates):
        objs = c.get("objectives", {})
        config = c.get("config", {})
        metrics = c.get("metrics", {})
        records.append({
            "idx": idx,
            "throughput_per_gpu": objs.get("throughput_per_gpu", c.get("score")),
            "throughput_per_user": objs.get("throughput_per_user", metrics.get("mean_output_token_throughput_per_user")),
            "used_gpus": c.get("used_gpus", config.get("used_gpus")),
            "concurrency": config.get("concurrency"),
            "kv_load_ratio": config.get("kv_load_ratio"),
            "deployment_mode": config.get("deployment_mode")
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="throughput_per_user").reset_index(drop=True)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot Pareto curve line
    plt.plot(df["throughput_per_user"], df["throughput_per_gpu"], 
             color='#1f77b4', linestyle='--', alpha=0.7, linewidth=1.5, label='Pareto Frontier')

    # Scatter points grouped by used_gpus
    for gpus, group in df.groupby("used_gpus"):
        plt.scatter(group["throughput_per_user"], group["throughput_per_gpu"], 
                    s=80, label=f'{gpus} GPUs', zorder=5)

    # Annotate points with index & concurrency
    for _, row in df.iterrows():
        label = f"#{int(row['idx'])}\n({int(row['concurrency']) if row['concurrency'] else 'N/A'} c)"
        plt.annotate(label, 
                     (row["throughput_per_user"], row["throughput_per_gpu"]),
                     textcoords="offset points", 
                     xytext=(0, 8), 
                     ha='center', 
                     fontsize=8, 
                     color='#333333')

    plt.title("Sweeper Pareto Frontier (Throughput/GPU vs Throughput/User)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Throughput per User (tokens/s/user)", fontsize=12)
    plt.ylabel("Throughput per GPU (tokens/s/GPU)", fontsize=12)
    plt.legend(title="Hardware Budget", loc="best", frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_png)
    print(f"Pareto plot successfully saved to {output_png}")

if __name__ == "__main__":
    json_file = sys.argv[1] if len(sys.argv) > 1 else "/workspace/dynosim_aic_sweep/results/pareto_gb300_deepseek_v4.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/workspace/dynosim_aic_sweep/results/pareto_gb300_deepseek_v4.png"
    plot_pareto(json_file, output_file)
