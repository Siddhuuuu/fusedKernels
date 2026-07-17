import argparse
import re
import subprocess
import sys
import statistics

RESULT_RE = re.compile(
    r"tokens/sec:\s*([\d,]+)\s*"
    r".*?peak VRAM:\s*([\d.]+)\s*GB"
    r".*?final loss:\s*([\d.]+)",
    re.DOTALL,
)

def run_once(mode, args):
    cmd = [
        sys.executable, "examples/train_mini_gpt.py",
        "--mode", mode,
        "--steps", str(args.steps),
        "--warmup", str(args.warmup),
        "--dim", str(args.dim),
        "--n-layers", str(args.n_layers),
        "--n-heads", str(args.n_heads),
        "--hidden-dim", str(args.hidden_dim),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--vocab-size", str(args.vocab_size),
        "--dtype", args.dtype,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"--- {mode} run FAILED ---")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    match = RESULT_RE.search(result.stdout)
    if not match:
        print(f"--- could not parse output for {mode} ---")
        print(result.stdout)
        sys.exit(1)

    tokens_per_sec = float(match.group(1).replace(",", ""))
    peak_vram_gb = float(match.group(2))
    final_loss = float(match.group(3))
    return tokens_per_sec, peak_vram_gb, final_loss

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=1408)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--dtype", default="bf16")
    args = p.parse_args()

    print(f"Config: dim={args.dim} layers={args.n_layers} heads={args.n_heads} "
          f"hidden_dim={args.hidden_dim} batch={args.batch_size} seq_len={args.seq_len} "
          f"vocab={args.vocab_size} dtype={args.dtype}")
    print(f"Running {args.trials} trial(s) per mode, {args.steps} steps each "
          f"({args.warmup} warmup steps not timed)...\n")

    results = {}
    for mode in ["native", "fused"]:
        tps_list, vram_list, loss_list = [], [], []
        for t in range(args.trials):
            print(f"[{mode}] trial {t + 1}/{args.trials}...", flush=True)
            tps, vram, loss = run_once(mode, args)
            tps_list.append(tps)
            vram_list.append(vram)
            loss_list.append(loss)
        results[mode] = {
            "tps_mean": statistics.mean(tps_list),
            "tps_std": statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0,
            "vram_mean": statistics.mean(vram_list),
            "loss_mean": statistics.mean(loss_list),
        }

    n = results["native"]
    f = results["fused"]
    speedup = f["tps_mean"] / n["tps_mean"]
    mem_reduction = (1 - f["vram_mean"] / n["vram_mean"]) * 100

    print("\n" + "=" * 60)
    print(f"{'':15}{'native':>15}{'fused':>15}{'delta':>15}")
    print("-" * 60)
    print(f"{'tokens/sec':15}{n['tps_mean']:>15,.0f}{f['tps_mean']:>15,.0f}{speedup:>14.2f}x")
    print(f"{'  (std dev)':15}{n['tps_std']:>15,.0f}{f['tps_std']:>15,.0f}")
    print(f"{'peak VRAM (GB)':15}{n['vram_mean']:>15.2f}{f['vram_mean']:>15.2f}{mem_reduction:>13.1f}%")
    print(f"{'final loss':15}{n['loss_mean']:>15.4f}{f['loss_mean']:>15.4f}")
    print("=" * 60)
    print(f"\nResult: fused is {speedup:.2f}x faster and uses {mem_reduction:.1f}% less peak VRAM,")
    print(f"averaged over {args.trials} trial(s) x {args.steps} steps each.")

if __name__ == "__main__":
    main()
