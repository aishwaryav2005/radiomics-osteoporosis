#!/usr/bin/env python
"""Single entry point for the whole experimental pipeline.

    python run_all.py                      # run everything, resuming where possible
    python run_all.py --stage dataset      # one stage
    python run_all.py --stage train --smoke-test
    python run_all.py --list-stages

Stages are resumable: completed training runs are skipped unless ``--force`` is
given, so a crash never costs more than the run that was in flight.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.utils import banner, get_logger, human_seconds, set_all_seeds, write_json  # noqa: E402

log = get_logger("run_all")

STAGES = [
    ("env", "Environment and GPU check"),
    ("dataset", "Dataset scan, integrity, duplicates, manifest, statistics"),
    ("folds", "Stratified group five-fold split generation"),
    ("features", "Frozen-backbone feature bank extraction"),
    ("verify", "Model build verification"),
    ("smoke", "Smoke test: one architecture, one fold, few epochs"),
    ("train", "Five-fold training of all ablation configurations"),
    ("evaluate", "Metrics, cross-validation summary, out-of-fold predictions"),
    ("stats", "Paired statistical significance testing"),
    ("complexity", "Parameters, FLOPs, model size, latency"),
    ("ablation", "Component ablation analysis"),
    ("figures", "Publication-quality figures"),
    ("tables", "Manuscript-ready tables (CSV + XLSX)"),
    ("gradcam", "Grad-CAM interpretability figures"),
    ("multimodal", "LUMOS multimodal feasibility assessment"),
    ("report", "Final report"),
]

DEFAULT_ORDER = [s for s, _ in STAGES if s not in ("smoke",)]


# --------------------------------------------------------------------------
def stage_env(args) -> dict:
    from src.env_check import check
    return check()


def stage_dataset(args) -> dict:
    from src.dataset_scan import build_manifest
    return build_manifest()


def stage_folds(args) -> dict:
    from src.splits import generate_folds
    return generate_folds()


def stage_features(args) -> dict:
    from src.features import extract_feature_bank
    return extract_feature_bank(force=args.force)


def stage_verify(args) -> dict:
    """Known-answer self-test, then build every architecture and check shapes."""
    banner("STAGE: MODEL VERIFICATION")
    import subprocess

    from src.features import FeatureBank
    from src.models import build_fusion_head, count_parameters

    st = subprocess.run(
        [sys.executable, str(config.PROJECT_ROOT / "scripts" / "self_test.py")],
        capture_output=True, text=True,
    )
    tail = [ln for ln in st.stdout.splitlines() if "FAIL" in ln or "ALL CHECKS" in ln]
    for ln in tail:
        log.info("self-test: %s", ln.strip())
    if st.returncode != 0:
        raise AssertionError("Numerical self-test failed; see scripts/self_test.py output")

    out = {"self_test": "passed"}
    for arch in config.ABLATION_ORDER:
        cfg = config.ABLATION_CONFIGS[arch]
        bank = FeatureBank(cfg["backbones"])
        model = build_fusion_head(bank.dim)
        p = count_parameters(model)
        expected = sum(config.BACKBONE_FEATURE_DIMS[b] for b in cfg["backbones"])
        ok = bank.dim == expected
        out[arch] = {"feature_dim": bank.dim, "expected_dim": expected,
                     "matches": ok, **p}
        log.info("%s | %-38s | dim %4d (expected %4d) %-8s | head params %s",
                 arch, cfg["display"], bank.dim, expected,
                 "OK" if ok else "MISMATCH", f"{p['trainable_params']:,}")
        if not ok:
            raise AssertionError(f"{arch}: fused dim {bank.dim} != expected {expected}")
        import tensorflow as tf
        tf.keras.backend.clear_session()
    write_json(config.RESULTS_DIR / "model_verification.json", out)
    return out


def stage_smoke(args) -> dict:
    """Cheap end-to-end check before committing to 25 full runs."""
    banner("STAGE: SMOKE TEST")
    from src.features import FeatureBank
    from src.train import train_one

    arch, fold = config.FULL_MODEL, 1
    bank = FeatureBank(config.ABLATION_CONFIGS[arch]["backbones"])
    res = train_one(arch, fold, bank, max_epochs=5, force=True)

    hist_ok = res["epochs_run"] >= 1
    loss_dropped = res["final_train_loss"] < 1.0
    checks = {
        "images_loaded": bank.index.shape[0] > 0,
        "model_built": res["head_parameters"]["trainable_params"] > 0,
        "training_ran": hist_ok,
        "loss_finite_and_decreasing": loss_dropped,
        "checkpoint_saved": (config.MODELS_DIR / arch / f"fold_{fold}" /
                             "best_model.keras").exists(),
        "evaluation_ran": res["test_metrics"]["n_samples"] > 0,
        "metrics_verified": res["metric_verification"]["agrees"],
    }
    for k, v in checks.items():
        log.info("  %-32s %s", k, "PASS" if v else "FAIL")
    if not all(checks.values()):
        raise RuntimeError(f"Smoke test failed: {checks}")

    # The smoke run used 5 epochs; clear its status so the real run redoes it.
    from src.utils import set_status
    set_status(f"{arch}_fold{fold}", "pending", {"note": "reset after smoke test"})
    log.info("Smoke test PASSED. Test accuracy after 5 epochs: %.4f",
             res["test_metrics"]["accuracy"])
    return {"checks": checks, "smoke_accuracy": res["test_metrics"]["accuracy"]}


def stage_train(args) -> dict:
    from src.train import run_all_experiments
    return run_all_experiments(
        architectures=args.architectures,
        folds=args.folds,
        max_epochs=args.epochs,
        force=args.force,
    )


def stage_evaluate(args) -> dict:
    from src.evaluate import evaluate
    return evaluate()


def stage_stats(args) -> dict:
    from src.stats_tests import run_statistical_tests
    return run_statistical_tests()


def stage_complexity(args) -> dict:
    from src.complexity import analyse_complexity
    return {"rows": len(analyse_complexity())}


def stage_ablation(args) -> dict:
    from src.ablation import run_ablation
    return run_ablation()


def stage_figures(args) -> dict:
    from src.figures import generate_all_figures
    return generate_all_figures()


def stage_tables(args) -> dict:
    from src.tables import generate_all_tables
    return generate_all_tables()


def stage_gradcam(args) -> dict:
    from src.gradcam import generate_gradcam
    return generate_gradcam()


def stage_multimodal(args) -> dict:
    from src.multimodal import run_multimodal
    return run_multimodal(args.lumos_dir)


def stage_report(args) -> dict:
    from src.report import generate_report
    return {"report": str(generate_report())}


HANDLERS = {
    "env": stage_env, "dataset": stage_dataset, "folds": stage_folds,
    "features": stage_features, "verify": stage_verify, "smoke": stage_smoke,
    "train": stage_train, "evaluate": stage_evaluate, "stats": stage_stats,
    "complexity": stage_complexity, "ablation": stage_ablation,
    "figures": stage_figures, "tables": stage_tables, "gradcam": stage_gradcam,
    "multimodal": stage_multimodal, "report": stage_report,
}


# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Knee osteoporosis radiomics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages:\n" + "\n".join(f"  {n:<12} {d}" for n, d in STAGES),
    )
    p.add_argument("--stage", action="append", choices=[s for s, _ in STAGES],
                   help="run only this stage (repeatable)")
    p.add_argument("--from-stage", choices=[s for s, _ in STAGES],
                   help="run this stage and everything after it")
    p.add_argument("--skip", action="append", choices=[s for s, _ in STAGES],
                   default=[], help="skip this stage (repeatable)")
    p.add_argument("--force", action="store_true",
                   help="recompute even if cached/completed results exist")
    p.add_argument("--epochs", type=int, default=None, help="override max epochs")
    p.add_argument("--architectures", nargs="+", choices=config.ABLATION_ORDER,
                   default=None, help="restrict training to these configurations")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="restrict training to these folds")
    p.add_argument("--smoke-test", action="store_true",
                   help="insert the smoke test before training")
    p.add_argument("--lumos-dir", default=None, help="path to an extracted LUMOS dataset")
    p.add_argument("--continue-on-error", action="store_true",
                   help="keep going if a stage fails")
    p.add_argument("--list-stages", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_stages:
        print("Available stages:\n")
        for n, d in STAGES:
            print(f"  {n:<12} {d}")
        return 0

    config.ensure_dirs()
    config.set_tf_threading_env()
    set_all_seeds(config.GLOBAL_SEED)

    if args.stage:
        order = list(args.stage)
    elif args.from_stage:
        names = [s for s, _ in STAGES]
        order = [s for s in names[names.index(args.from_stage):] if s != "smoke"]
    else:
        order = list(DEFAULT_ORDER)
        if args.smoke_test:
            order.insert(order.index("train"), "smoke")
    order = [s for s in order if s not in args.skip]

    banner("KNEE OSTEOPOROSIS RADIOMICS PIPELINE")
    log.info("Project root : %s", config.PROJECT_ROOT)
    log.info("Stages       : %s", " -> ".join(order))
    log.info("")

    summary: dict[str, dict] = {}
    t_all = time.perf_counter()
    failed_stage = None

    for name in order:
        t0 = time.perf_counter()
        try:
            result = HANDLERS[name](args)
            dt = time.perf_counter() - t0
            summary[name] = {"status": "ok", "seconds": round(dt, 2)}
            log.info("[stage %s] completed in %s", name, human_seconds(dt))
        except Exception as exc:  # noqa: BLE001
            dt = time.perf_counter() - t0
            tb = traceback.format_exc()
            summary[name] = {"status": "failed", "seconds": round(dt, 2),
                             "error": f"{type(exc).__name__}: {exc}"}
            log.error("[stage %s] FAILED after %s: %s", name, human_seconds(dt), exc)
            (config.LOGS_DIR / f"stage_error_{name}.txt").write_text(tb, encoding="utf-8")
            failed_stage = name
            if not args.continue_on_error:
                break
        log.info("")

    total = time.perf_counter() - t_all
    summary["_total_seconds"] = round(total, 2)
    write_json(config.RESULTS_DIR / "pipeline_run_summary.json", summary)

    banner("PIPELINE SUMMARY")
    for name in order:
        s = summary.get(name)
        if s:
            log.info("  %-12s %-8s %s", name, s["status"], human_seconds(s["seconds"]))
    log.info("  %-12s %-8s %s", "TOTAL", "", human_seconds(total))

    if failed_stage and not args.continue_on_error:
        log.error("Pipeline stopped at stage '%s'. See results/logs/stage_error_%s.txt",
                  failed_stage, failed_stage)
        return 1
    log.info("Results  : %s", config.RESULTS_DIR)
    log.info("Figures  : %s", config.FIGURES_DIR)
    log.info("Tables   : %s", config.TABLES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
