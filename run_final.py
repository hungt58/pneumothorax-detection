"""Run every end-of-term experiment in one go (Kaggle GPU) and collect the numbers.

Stages (each is skipped if its result already exists, so a crashed run can resume):
  leak      duplicate-image check across train/val/test
  base      E1-E4 at 256: reuse checkpoints found under --ckpt-search (or train them),
            tune thresholds on VAL, score TEST once
  cascade   E2 + E4 (screen then confirm), tuned on VAL, scored on TEST once
  e4b       ResNet18 segmentation only (train, tune, test)
  e2_512    Attention U-Net at 512x512 (train, tune, test)
  viz       prediction figures for the slides

Outputs: results/final_summary.json, results/final_summary.md,
configs/thresholds.json (updated with all VAL-tuned thresholds) and /kaggle/working/final_results.zip.

    python run_final.py --images IMAGES --masks MASKS --ckpt-search /kaggle/input
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
CONFIG = ROOT / "configs" / "thresholds.json"

# key -> (model name, image size, batch size, display name)
EXPERIMENTS = {
    "e1_unet": ("unet", 256, 8, "E1 U-Net"),
    "e2_attention_unet": ("attention_unet", 256, 8, "E2 Attention U-Net"),
    "e3_attention_multiscale": ("attention_multiscale_unet", 256, 8, "E3 Attention + Multi-scale"),
    "e4_resnet18_multitask": ("resnet18_multitask", 256, 8, "E4 ResNet18 multi-task"),
    "e4b_resnet18_seg": ("resnet18_seg", 256, 8, "E4b ResNet18 seg-only"),
    "e2_attention_unet_512": ("attention_unet", 512, 4, "E2 Attention U-Net @512"),
}
OLD_NAMES = {"unet_balanced_best.pt": "e1_unet"}
METRICS = ("dice_positive", "iou_positive", "image_sensitivity", "image_specificity",
           "selection_score", "tp", "fn", "fp", "tn")


def log(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(args: list[str]):
    log("$ python " + " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def best_row(path: Path) -> dict:
    return max(read_csv(path), key=lambda r: r["selection_score"])


def update_config(key: str, values: dict):
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.is_file() else {}
    config[key] = {**config.get(key, {}), **values}
    CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_checkpoints(search: list[Path]):
    """Copy *_best.pt files found under the search roots into checkpoints/."""
    CKPT.mkdir(exist_ok=True)
    for root in search:
        if not root.exists():
            continue
        for path in root.rglob("*_best.pt"):
            key = OLD_NAMES.get(path.name, path.name.removesuffix("_best.pt"))
            target = CKPT / f"{key}_best.pt"
            if key in EXPERIMENTS and not target.exists():
                shutil.copy2(path, target)
                log(f"found checkpoint {path} -> {target.name}")


def ensure_trained(key: str, common: list[str], epochs: int):
    model, size, batch, _ = EXPERIMENTS[key]
    ckpt = CKPT / f"{key}_best.pt"
    if ckpt.exists():
        return ckpt
    run(["train.py", "--model", model, "--image-size", str(size), "--batch-size", str(batch),
         "--epochs", str(epochs), "--checkpoint", str(ckpt),
         "--history", str(RESULTS / key / "history.csv"), *common])
    return ckpt


def tune_and_test(key: str, common: list[str]) -> dict:
    """VAL tuning then one TEST run; returns the summary row for this experiment."""
    model, size, batch, name = EXPERIMENTS[key]
    ckpt = CKPT / f"{key}_best.pt"
    out = RESULTS / key
    test_csv = out / "final_test_metrics.csv"
    bs = ["--batch-size", str(batch)]
    if model == "resnet18_multitask":
        tune_csv = out / "seg_cls_threshold_tuning.csv"
        if not tune_csv.exists():
            run(["evaluate_multitask_thresholds.py", "--checkpoint", str(ckpt), "--output", str(tune_csv), *bs, *common])
        best = best_row(tune_csv)
        thr = {"seg_threshold": round(best["seg_threshold"], 4), "cls_threshold": round(best["cls_threshold"], 4)}
        if not test_csv.exists():
            run(["evaluate_multitask_thresholds.py", "--split", "test", "--checkpoint", str(ckpt),
                 "--seg-threshold", str(thr["seg_threshold"]), "--cls-threshold", str(thr["cls_threshold"]),
                 "--output", str(test_csv), *bs, *common])
    else:
        tune_csv = out / "threshold_tuning.csv"
        if not tune_csv.exists():
            run(["evaluate_thresholds.py", "--checkpoint", str(ckpt), "--threshold-start", "0.30",
                 "--threshold-end", "0.98", "--threshold-step", "0.01", "--output", str(tune_csv), *bs, *common])
        best = best_row(tune_csv)
        thr = {"seg_threshold": round(best["threshold"], 4)}
        if not test_csv.exists():
            run(["evaluate_test.py", "--checkpoint", str(ckpt), "--threshold", str(thr["seg_threshold"]),
                 "--output", str(test_csv), *bs, *common])
    update_config(key, {**thr, "min_pred_pixels": 64})
    test = read_csv(test_csv)[0]
    if "selection_score" not in test:
        _d, _sp = float(test["dice_positive"]), float(test["image_specificity"])
        test["selection_score"] = 2 * _d * _sp / (_d + _sp + 1e-8)
    return {"key": key, "name": name, "image_size": size, "thresholds": thr,
            "val": {m: best[m] for m in METRICS}, "test": {m: test[m] for m in METRICS}}


def stage_cascade(common: list[str]) -> dict:
    out = RESULTS / "cascade_e2_e4"
    tune_csv, test_csv = out / "threshold_tuning.csv", out / "final_test_metrics.csv"
    if not tune_csv.exists():
        run(["evaluate_cascade.py", "--split", "val", "--write-config", "--output", str(tune_csv), *common])
    best = best_row(tune_csv)
    update_config("cascade_e2_e4", {"e2_seg_threshold": best["e2_threshold"],
                                    "e4_cls_threshold": best["e4_cls_threshold"], "tuned": True})
    if not test_csv.exists():
        run(["evaluate_cascade.py", "--split", "test", "--e2-threshold", str(best["e2_threshold"]),
             "--e4-cls-threshold", str(best["e4_cls_threshold"]), "--output", str(test_csv), *common])
    test = read_csv(test_csv)[0]
    if "selection_score" not in test:
        _d, _sp = float(test["dice_positive"]), float(test["image_specificity"])
        test["selection_score"] = 2 * _d * _sp / (_d + _sp + 1e-8)
    return {"key": "cascade_e2_e4", "name": "E2 + E4 cascade", "image_size": 256,
            "thresholds": {"e2_seg_threshold": best["e2_threshold"], "e4_cls_threshold": best["e4_cls_threshold"]},
            "val": {m: best[m] for m in METRICS}, "test": {m: test[m] for m in METRICS}}


def stage_viz(rows: list[dict], common: list[str]):
    for row in rows:
        key = row["key"]
        ckpt = CKPT / f"{key}_best.pt"
        out = RESULTS / key / "predictions_val"
        if key not in EXPERIMENTS or not ckpt.exists() or out.exists():
            continue
        extra = ["--threshold", str(row["thresholds"]["seg_threshold"])]
        if "cls_threshold" in row["thresholds"]:
            extra += ["--cls-threshold", str(row["thresholds"]["cls_threshold"])]
        try:
            run(["visualize_predictions.py", "--checkpoint", str(ckpt), "--num-samples", "12",
                 "--output-dir", str(out), "--batch-size", str(EXPERIMENTS[key][2]), *extra, *common])
        except subprocess.CalledProcessError as exc:  # figures are optional
            log(f"viz failed for {key}: {exc}")


def write_summary(summary: dict):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["| Model | Size | Thresholds (VAL) | Dice+ | IoU+ | Sens | Spec | Score | TP | FN | FP | TN |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["models"]:
        t = r["test"]
        thr = ", ".join(f"{k}={v:g}" for k, v in r["thresholds"].items())
        lines.append(f"| {r['name']} | {r['image_size']} | {thr} | {t['dice_positive']:.3f} | {t['iou_positive']:.3f} | "
                     f"{t['image_sensitivity']:.3f} | {t['image_specificity']:.3f} | {t['selection_score']:.3f} | "
                     f"{int(t['tp'])} | {int(t['fn'])} | {int(t['fp'])} | {int(t['tn'])} |")
    if summary.get("leakage"):
        lines += ["", "Leakage check:", "```", summary["leakage"], "```"]
    if summary.get("errors"):
        lines += ["", "Stages that failed:", *[f"- {e}" for e in summary["errors"]]]
    (RESULTS / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--ckpt-search", nargs="*", type=Path, default=[Path("/kaggle/input")])
    p.add_argument("--stages", nargs="*", default=["leak", "base", "cascade", "e4b", "e2_512", "viz"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--zip", type=Path, default=Path("/kaggle/working/final_results.zip"))
    a = p.parse_args()
    common = ["--images", a.images, "--masks", a.masks, "--num-workers", str(a.num_workers)]
    train_common = ["--images", a.images, "--masks", a.masks, "--num-workers", str(a.num_workers)]

    find_checkpoints(a.ckpt_search)
    summary = {"models": [], "errors": [], "leakage": None}

    def attempt(label, fn):
        try:
            return fn()
        except Exception as exc:  # keep going: later stages are independent
            log(f"STAGE {label} FAILED: {exc}")
            summary["errors"].append(f"{label}: {exc}")
            return None
        finally:
            write_summary(summary)

    if "leak" in a.stages:
        def leak():
            report = RESULTS / "leakage" / "report.txt"
            if not report.exists():
                run(["check_split_leakage.py", "--images", a.images, "--masks", a.masks,
                     "--output-dir", str(report.parent)])
            summary["leakage"] = report.read_text(encoding="utf-8").strip()
        attempt("leak", leak)

    plan = []
    if "base" in a.stages:
        plan += ["e1_unet", "e2_attention_unet", "e3_attention_multiscale", "e4_resnet18_multitask"]
    for key in plan:
        def base(key=key):
            ensure_trained(key, train_common, a.epochs)
            summary["models"].append(tune_and_test(key, common))
        attempt(key, base)

    if "cascade" in a.stages:
        attempt("cascade", lambda: summary["models"].append(stage_cascade(common)))
    for stage, key in (("e4b", "e4b_resnet18_seg"), ("e2_512", "e2_attention_unet_512")):
        if stage in a.stages:
            def extra(key=key):
                ensure_trained(key, train_common, a.epochs)
                summary["models"].append(tune_and_test(key, common))
            attempt(stage, extra)
    if "viz" in a.stages:
        attempt("viz", lambda: stage_viz(summary["models"], common))

    write_summary(summary)
    if a.zip:
        a.zip.parent.mkdir(parents=True, exist_ok=True)
        staging = ROOT / "_final_export"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(RESULTS, staging / "results")
        shutil.copytree(ROOT / "configs", staging / "configs")
        archive = shutil.make_archive(str(a.zip.with_suffix("")), "zip", staging)
        log(f"Download this file: {archive}")


if __name__ == "__main__":
    main()
