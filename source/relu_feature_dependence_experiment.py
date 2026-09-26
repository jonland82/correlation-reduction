"""Synthetic ReLU experiment: per-feature output dependence and update energy."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import win32pdh
except ImportError:
    win32pdh = None


FEATURES = [
    "linear strong",
    "U-shaped",
    "linear proxy",
    "interaction A",
    "interaction B",
    "linear weak",
    "noise A",
    "noise B",
]
PENALTIES = [0.1, 0.3, 1.0, 3.0, 10.0]


class EnergyMeter:
    """Windows RAPL PKG + DRAM counters; counter energy is in picowatt-hours."""

    def __init__(self):
        self.query = None
        self.counters = {}
        if win32pdh is not None:
            try:
                self.query = win32pdh.OpenQuery()
                for name in ("PKG", "DRAM"):
                    path = rf"\Energy Meter(RAPL_Package0_{name})\Energy"
                    self.counters[name] = win32pdh.AddCounter(self.query, path)
                self.read()
            except Exception:
                if self.query is not None:
                    win32pdh.CloseQuery(self.query)
                self.query = None
                self.counters = {}

    def read(self):
        if self.query is None:
            return None
        win32pdh.CollectQueryData(self.query)
        return {
            name: win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_LARGE)[1]
            for name, counter in self.counters.items()
        }

    @staticmethod
    def joules(before, after):
        if before is None or after is None:
            return None
        return sum((after[key] - before[key]) * 3.6e-9 for key in ("PKG", "DRAM"))

    def close(self):
        if self.query is not None:
            win32pdh.CloseQuery(self.query)


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.empty((n, 8), dtype=np.float32)
    for j in (0, 3, 4, 5, 6, 7):
        x[:, j] = rng.choice([-1.0, 1.0], size=n)
    x[:, 1] = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.25, 0.5, 0.25])
    x[:, 2] = x[:, 0] * rng.choice([-1.0, 1.0], size=n, p=[0.1, 0.9])
    logit = (
        1.4 * x[:, 0]
        + 1.5 * (np.abs(x[:, 1]) - 0.5)
        + 1.3 * x[:, 3] * x[:, 4]
        + 0.6 * x[:, 5]
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p).astype(np.float32)
    return x, y


def tensors(n: int, seed: int):
    x, y = make_data(n, seed)
    return torch.from_numpy(x), torch.from_numpy(y).reshape(-1, 1)


def model_from_seed(seed: int):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def binary_kl(p, q):
    eps = 1e-7
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def mi_torch(probs, feature, values):
    q = probs.mean()
    result = q.new_zeros(())
    for value in values:
        mask = feature == value
        result = result + mask.float().mean() * binary_kl(probs[mask].mean(), q)
    return result


def metrics(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        p = torch.sigmoid(logits).reshape(-1)
        label = y.reshape(-1)
        ce = F.binary_cross_entropy_with_logits(logits, y).item()
        hard_accuracy = ((p >= 0.5).float() == label).float().mean().item()
        q = p.mean()
        mi = []
        corr = []
        for j in range(x.shape[1]):
            values = torch.unique(x[:, j])
            mi.append(mi_torch(p, x[:, j], values).item())
            xx = x[:, j]
            covariance = ((xx - xx.mean()) * (p - q)).mean()
            denom = xx.std(unbiased=False) * torch.sqrt(q * (1 - q))
            corr.append((covariance / denom).item())
    return {"ce": ce, "accuracy": hard_accuracy, "mi": mi, "corr": corr}


def train_base(model, x, y, steps):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        ce = F.binary_cross_entropy_with_logits(model(x), y)
        ce.backward()
        opt.step()


def finetune(
    base,
    xtrain,
    ytrain,
    xval,
    yval,
    feature,
    penalty,
    target_mi,
    max_ce,
    steps,
    check_every,
    meter,
    idle_watts,
):
    model = model_from_seed(0)
    model.load_state_dict(base.state_dict())
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-4)
    values = torch.unique(xtrain[:, feature])
    start_energy = meter.read()
    start_time = time.perf_counter()
    hit = False
    val = None
    for step in range(1, steps + 1):
        opt.zero_grad()
        logits = model(xtrain)
        p = torch.sigmoid(logits).reshape(-1)
        ce = F.binary_cross_entropy_with_logits(logits, ytrain)
        mi = mi_torch(p, xtrain[:, feature], values)
        (ce + penalty * mi).backward()
        opt.step()
        if step % check_every == 0 or step == steps:
            val = metrics(model, xval, yval)
            if val["mi"][feature] <= target_mi and val["ce"] <= max_ce:
                hit = True
                break
    elapsed = time.perf_counter() - start_time
    raw_j = meter.joules(start_energy, meter.read())
    net_j = None if raw_j is None else raw_j - idle_watts * elapsed
    return {
        "model": model,
        "hit": hit,
        "step": step,
        "seconds": elapsed,
        "raw_j": raw_j,
        "net_j": net_j,
        "val": val,
        "penalty": penalty,
    }


def permutation_loss(model, x, y, seed):
    rng = np.random.default_rng(seed)
    baseline = metrics(model, x, y)["ce"]
    result = []
    for j in range(x.shape[1]):
        perturbed = x.clone()
        perturbed[:, j] = x[torch.as_tensor(rng.permutation(len(x))), j]
        result.append(metrics(model, perturbed, y)["ce"] - baseline)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-steps", type=int, default=250)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-n", type=int, default=12000)
    parser.add_argument("--val-n", type=int, default=12000)
    parser.add_argument("--test-n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-mi", type=float, default=0.002)
    parser.add_argument("--ce-tolerance", type=float, default=0.015)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--energy-repeats", type=int, default=12)
    parser.add_argument("--outdir", type=Path, default=Path("source/data/relu_dependence"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    xtrain, ytrain = tensors(args.train_n, args.seed)
    xval, yval = tensors(args.val_n, args.seed + 1)
    xtest, ytest = tensors(args.test_n, args.seed + 2)
    meter = EnergyMeter()
    idle_before = meter.read()
    idle_start = time.perf_counter()
    time.sleep(4)
    idle_elapsed = time.perf_counter() - idle_start
    idle_j = meter.joules(idle_before, meter.read())
    idle_watts = 0.0 if idle_j is None else idle_j / idle_elapsed
    print(f"idle package+DRAM watts: {idle_watts:.3f}", flush=True)

    base = model_from_seed(args.seed)
    energy_before = meter.read()
    base_start = time.perf_counter()
    train_base(base, xtrain, ytrain, args.base_steps)
    base_seconds = time.perf_counter() - base_start
    base_raw_j = meter.joules(energy_before, meter.read())
    baseline_val = metrics(base, xval, yval)
    baseline_test = metrics(base, xtest, ytest)
    permutation = permutation_loss(base, xtest, ytest, args.seed + 3)
    print(f"base val CE {baseline_val['ce']:.4f}, test accuracy {baseline_test['accuracy']:.4f}", flush=True)

    rows = []
    all_candidates = []
    for j, feature_name in enumerate(FEATURES):
        base_mi = baseline_val["mi"][j]
        row = {
            "feature": j,
            "name": feature_name,
            "baseline_mi": baseline_test["mi"][j],
            "abs_corr": abs(baseline_test["corr"][j]),
            "permutation_ce_increase": permutation[j],
            "hit": False,
            "penalty": None,
            "steps": None,
            "seconds": None,
            "raw_j": None,
            "net_j": None,
            "net_j_sd": None,
            "test_mi_after": None,
            "test_ce_increase": None,
        }
        if base_mi < args.min_mi:
            row["status"] = "below MI threshold"
            print(f"{j} {feature_name}: baseline MI {base_mi:.5f}, skipped", flush=True)
            rows.append(row)
            continue
        candidates = []
        for penalty in PENALTIES:
            candidate = finetune(
                base, xtrain, ytrain, xval, yval, j, penalty,
                base_mi * args.target_fraction,
                baseline_val["ce"] + args.ce_tolerance,
                args.steps, args.check_every, meter, idle_watts,
            )
            test_result = metrics(candidate["model"], xtest, ytest)
            record = {
                "feature": j,
                "penalty": penalty,
                "hit": candidate["hit"],
                "steps": candidate["step"],
                "seconds": candidate["seconds"],
                "raw_j": candidate["raw_j"],
                "net_j": candidate["net_j"],
                "val_mi": candidate["val"]["mi"][j],
                "val_ce": candidate["val"]["ce"],
                "test_mi": test_result["mi"][j],
                "test_ce": test_result["ce"],
            }
            candidates.append(record)
            all_candidates.append(record)
        hits = [c for c in candidates if c["hit"]]
        if hits:
            best = min(hits, key=lambda c: (c["steps"], c["val_ce"]))
            energy_repeats = []
            for _ in range(args.energy_repeats):
                replay = finetune(
                    base, xtrain, ytrain, xval, yval, j, best["penalty"],
                    -1.0, -1.0, best["steps"], args.check_every, meter, idle_watts,
                )
                energy_repeats.append(replay)
            row.update({
                "hit": True,
                "status": "target met",
                "penalty": best["penalty"],
                "steps": best["steps"],
                "seconds": float(np.mean([r["seconds"] for r in energy_repeats])),
                "raw_j": float(np.mean([r["raw_j"] for r in energy_repeats])),
                "net_j": float(np.mean([r["net_j"] for r in energy_repeats])),
                "net_j_sd": float(np.std([r["net_j"] for r in energy_repeats], ddof=1)),
                "test_mi_after": best["test_mi"],
                "test_ce_increase": best["test_ce"] - baseline_test["ce"],
            })
        else:
            row["status"] = "no feasible candidate"
            feasible = [c for c in candidates if c["val_ce"] <= baseline_val["ce"] + args.ce_tolerance]
            if feasible:
                best = min(feasible, key=lambda c: c["val_mi"])
                row["test_mi_after"] = best["test_mi"]
                row["test_ce_increase"] = best["test_ce"] - baseline_test["ce"]
        rows.append(row)
        print(
            f"{j} {feature_name}: MI {row['baseline_mi']:.5f}, "
            f"|r| {row['abs_corr']:.4f}, {row['status']}, "
            f"steps {row['steps']}, net J {row['net_j']}",
            flush=True,
        )

    with (args.outdir / "features.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.outdir / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_candidates[0].keys())
        writer.writeheader()
        writer.writerows(all_candidates)
    with (args.outdir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": vars(args) | {"outdir": str(args.outdir)},
                "idle_watts": idle_watts,
                "base_seconds": base_seconds,
                "base_raw_j": base_raw_j,
                "baseline_val": baseline_val,
                "baseline_test": baseline_test,
                "features": rows,
            },
            fh, indent=2,
        )
    torch.save(base.state_dict(), args.outdir / "baseline_model.pt")
    meter.close()


if __name__ == "__main__":
    main()
