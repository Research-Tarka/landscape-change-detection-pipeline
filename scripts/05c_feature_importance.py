#!/usr/bin/env python3
"""Stage 5c -- which input features does the model actually need?

Supported models: ``unet`` and ``catboost`` ONLY. Any other ``model.type``
(threshold, random_forest, lightgbm, deeplabv3plus, segformer) is refused:
this analysis has not been set up or tuned for them.

Methods (each switched by ``feature_importance.<method>.enabled``)
-----------------------------------------------
permutation   Score drop when one feature (or a group) is destroyed, repeated
              ``--repeats`` times (mean +/- std).
                unet     : the channel is swapped with the same channel of
                           another val scene (keeps the channel's real
                           distribution, unlike mean-substitution; spatial
                           structure of the other scene, no link to the labels).
                catboost : the column is shuffled over a pixel sample.
shap          Per-class mean |SHAP| (CatBoost TreeSHAP, exact), on a
              class-balanced pixel sample -- shows WHICH feature separates
              WHICH class. For a unet run the CatBoost is a spectral PROXY
              trained here on a pixel sample (it ignores spatial context).
dropcolumn    Retrain the CatBoost without each feature and compare val mIoU:
              the only method that answers "is it as good without it?" and is
              immune to redundancy masking (a correlated twin takes over).
              Same proxy remark for unet. Slowest (one fit per feature).

A feature is a drop candidate when drop-column (mIoU change within the noise
threshold) AND SHAP (<= 10% of the top feature) agree; the permutation is
informative only (a trained net degrades whenever any channel is broken).
separability  Model-free: Jeffries-Matusita distance between every pair of annotated
              classes (0 = identical .. 2 = separable) and the best single features
              for the weak pairs. Tells whether classes are separable from the
              features at all (use it before annotating a new class).

Always confirm by retraining the real model without it.

Everything is configured in ``feature_importance:`` of config.yaml (no
options here besides the config/class file paths). Results: one ``.npz`` of
``tbl_*`` tables under ``feature_importance.output_root``, exportable as CSV
from ``scripts/export_gui.py``, plus a ``feature_importance.json`` summary.

Usage
-----
    python scripts/05c_feature_importance.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # live logs (long runs)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import (  # noqa: E402
    label_remap_table,
    load_class_config,
)
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.features.scene_context import record_context  # noqa: E402
from landscape_change_detection_pipeline.features.training_cache import load_scene_cache  # noqa: E402
from landscape_change_detection_pipeline.training.dataset import (  # noqa: E402
    collect_class_pixel_counts_by_scene,
    discover_scene_records,
    split_scenes,
    with_label_remap,
)
from landscape_change_detection_pipeline.training.losses import IGNORE_INDEX  # noqa: E402
from landscape_change_detection_pipeline.training.metrics import compute_confusion_metrics  # noqa: E402

SUPPORTED = ("unet", "catboost")
METHODS = ("permutation", "shap", "dropcolumn", "separability")

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--env-file", default=None, help="Path to a .env file")
    p.add_argument("--classes", default=None, help="Path to classes.yaml")
    return p.parse_args(argv)


# --------------------------------------------------------------------- data


def _sample_pixels(records, per_scene: int, seed: int):
    """(X (N,C), y (N,)) of labelled pixels, <= per_scene per scene."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for r in records:
        c = load_scene_cache(r.cache_dir, label_remap=r.label_remap)
        labels = c["labels"]
        yy, xx = np.nonzero(labels != IGNORE_INDEX)
        if len(yy) == 0:
            continue
        pick = rng.choice(len(yy), min(per_scene, len(yy)), replace=False)
        x = c["features"][:, yy[pick], xx[pick]].astype(np.float32).T
        ok = np.all(np.isfinite(x), axis=1)
        xs.append(x[ok])
        ys.append(labels[yy[pick], xx[pick]].astype(np.int64)[ok])
    return np.concatenate(xs), np.concatenate(ys)


def _miou(y_true, y_pred, num_classes, class_names) -> tuple[float, dict]:
    conf = np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)
    m = compute_confusion_metrics(conf, num_classes, class_names=class_names)
    return float(m["macro"]["miou"]), {e["class_name"]: e["iou"] for e in m["per_class"]}


def _targets(names, feature_groups: dict):
    t = [(n, [i]) for i, n in enumerate(names)]
    if feature_groups:
        for g, members in feature_groups.items():
            idx = [names.index(m) for m in members if m in names]
            if len(idx) > 1:
                t.append((f"[group] {g}", idx))
    return t


# ------------------------------------------------------------- permutation


def _build_patches(subset, patch: int, per_scene: int, min_labelled: float, seed: int):
    """Random (features (C,p,p) float32, labels (p,p)) patches holding at least
    ``min_labelled`` labelled pixels -- the same size the model trained on."""
    rng = np.random.default_rng(seed)
    feats, labs, ctxs = [], [], []
    for r in subset:
        c = load_scene_cache(r.cache_dir, label_remap=r.label_remap)
        ctx = record_context(r)
        f, lab = c["features"], c["labels"]
        h, w = lab.shape
        if h < patch or w < patch:
            continue
        got = 0
        for _ in range(per_scene * 6):
            y0, x0 = int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1))
            sl = lab[y0:y0 + patch, x0:x0 + patch]
            if (sl != IGNORE_INDEX).mean() >= min_labelled:
                feats.append(f[:, y0:y0 + patch, x0:x0 + patch].astype(np.float32))
                labs.append(sl.astype(np.int64))
                ctxs.append(ctx)
                got += 1
                if got >= per_scene:
                    break
    return feats, labs, ctxs


def _unet_permutation(loaded, subset, args, config, num_classes, class_names, targets):
    """Patch-based and GPU-resident: patches are cut and normalised once, then each
    ablation swaps the channel with the same channel of another patch (shift k) and
    is a single batched forward pass -- minutes instead of hours vs full scenes."""
    import torch

    device = next(loaded.model.parameters()).device
    patch = config.training.patch_size
    feats, labs, ctxs = _build_patches(subset, patch, args.patches_per_scene, 0.2, seed=2)
    n = len(feats)
    if n < 2:
        raise RuntimeError("not enough labelled patches for the permutation")
    mean = loaded.mean.astype(np.float32).reshape(1, -1, 1, 1)
    std = np.maximum(loaded.std.astype(np.float32), 1e-6).reshape(1, -1, 1, 1)
    x = np.nan_to_num((np.stack(feats) - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(x).to(device=device, dtype=torch.float16)
    y = torch.from_numpy(np.stack(labs)).to(device)
    ctx_all = torch.from_numpy(np.stack(ctxs)).to(device)
    uses_context = bool(getattr(loaded.model, "uses_scene_context", False))
    remap = subset[0].label_remap
    remap_t = torch.as_tensor(np.asarray(remap, dtype=np.int64), device=device) if remap is not None else None
    loaded.model.eval()
    print(f"[importance] unet permutation on {n} patches of {patch}px from {len(subset)} scenes")

    def score(channels: list[int], shift: int):
        conf = torch.zeros(num_classes * num_classes, dtype=torch.int64, device=device)
        with torch.inference_mode():
            for i in range(0, n, args.batch_size):
                xb = x[i:i + args.batch_size]
                if channels:
                    donor = x[(torch.arange(i, i + len(xb), device=device) + shift) % n]
                    xb = xb.clone()
                    xb[:, channels] = donor[:, channels]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    xin = xb.float() if device.type != "cuda" else xb
                    out = loaded.model(xin, context=ctx_all[i:i + args.batch_size]) if uses_context else loaded.model(xin)
                    pred = out.argmax(1)
                if remap_t is not None:
                    pred = remap_t[pred]
                t = y[i:i + args.batch_size]
                ok = t != IGNORE_INDEX
                conf += torch.bincount(t[ok] * num_classes + pred[ok], minlength=num_classes**2)
        c = conf.reshape(num_classes, num_classes).cpu().numpy()
        m = compute_confusion_metrics(c, num_classes, class_names=class_names)
        return float(m["macro"]["miou"]), {e["class_name"]: e["iou"] for e in m["per_class"]}

    base, base_per = score([], 1)
    print(f"[importance] unet baseline miou_macro={base:.4f}")
    out = {}
    for i, (label, idx) in enumerate(targets, 1):
        drops, per_drops = [], []
        for k in range(1, max(1, args.repeats) + 1):
            m, per = score(idx, 7 * k + 1)
            drops.append(base - m)
            per_drops.append({cn: base_per[cn] - v for cn, v in per.items()
                              if not (np.isnan(v) or np.isnan(base_per[cn]))})
        out[label] = {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "per_class": per_drops}
        print(f"[importance] [{i}/{len(targets)}] permutation {label:<34} d_miou={out[label]['mean']:+.4f} +/- {out[label]['std']:.4f}")
    return base, out


def _catboost_permutation(predict, x, y, args, num_classes, class_names, targets):
    base, base_per = _miou(y, predict(x), num_classes, class_names)
    print(f"[importance] catboost baseline pixel miou_macro={base:.4f} on {len(y)} px")
    rng = np.random.default_rng(0)
    out = {}
    for label, idx in targets:
        drops, per_drops = [], []
        for _ in range(max(1, args.repeats)):
            xp = x.copy()
            perm = rng.permutation(len(x))
            for ch in idx:  # joint permutation keeps a group's internal consistency
                xp[:, ch] = x[perm, ch]
            m, per = _miou(y, predict(xp), num_classes, class_names)
            drops.append(base - m)
            per_drops.append({cn: base_per[cn] - v for cn, v in per.items()
                              if not (np.isnan(v) or np.isnan(base_per[cn]))})
        out[label] = {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "per_class": per_drops}
        print(f"[importance] permutation {label:<34} d_miou={out[label]['mean']:+.4f} +/- {out[label]['std']:.4f}")
    return base, out


# ------------------------------------------------------------ catboost side


def _catboost_params(config, args):
    cfg = config.model.catboost
    return dict(
        iterations=args.proxy_iterations, learning_rate=max(cfg.learning_rate, 0.1), depth=min(cfg.depth, 8),
        task_type=cfg.task_type, random_state=cfg.random_state, loss_function="MultiClass", verbose=0,
    )


def _fit_catboost(config, args, x, y, xv, yv, num_classes, drop: list[int] | None = None):
    from catboost import CatBoostClassifier

    keep = [i for i in range(x.shape[1]) if not drop or i not in drop]
    model = CatBoostClassifier(classes_count=num_classes, **_catboost_params(config, args))
    model.fit(x[:, keep], y, eval_set=(xv[:, keep], yv), early_stopping_rounds=30)
    return model, keep


def _full_proba_pred(model, keep, x, num_classes):
    p = np.asarray(model.predict(x[:, keep])).reshape(-1).astype(np.int64)
    return p


def _shap(model, x, y, names, class_names, per_class: int, trained_classes=None):
    from catboost import Pool

    rng = np.random.default_rng(0)
    pick = []
    for k in np.unique(y):
        ids = np.nonzero(y == k)[0]
        pick.append(rng.choice(ids, min(per_class, len(ids)), replace=False))
    pick = np.concatenate(pick)
    sv = model.get_feature_importance(Pool(x[pick], y[pick]), type="ShapValues")  # (N, K, C+1)
    sv = np.asarray(sv)[:, :, :-1]
    if sv.ndim == 2:  # binary fallback
        sv = sv[:, None, :]
    by_class = {}
    classes = [int(k) for k in np.asarray(getattr(model, "classes_", [])).reshape(-1)]
    if len(classes) != sv.shape[1]:  # trained on fewer classes than classes_count: use the ones it saw
        classes = [int(k) for k in (trained_classes if trained_classes is not None else np.unique(y))]
    for j, k in enumerate(classes):
        rows = y[pick] == k
        if rows.any():
            by_class[class_names[int(k)]] = np.abs(sv[rows, j, :]).mean(axis=0)
    overall = np.mean([v for v in by_class.values()], axis=0)  # class-balanced mean |SHAP|
    return overall, by_class


# ------------------------------------------------------------ separability


def _jm(a: np.ndarray, b: np.ndarray) -> float:
    """Jeffries-Matusita distance, Gaussian approximation: 2 (1 - exp(-B))."""
    d = a.shape[1]
    ca = np.atleast_2d(np.cov(a, rowvar=False)) + 1e-6 * np.eye(d)
    cb = np.atleast_2d(np.cov(b, rowvar=False)) + 1e-6 * np.eye(d)
    c = (ca + cb) / 2
    diff = (a.mean(0) - b.mean(0)).reshape(-1, 1)
    _, ld_c = np.linalg.slogdet(c)
    _, ld_a = np.linalg.slogdet(ca)
    _, ld_b = np.linalg.slogdet(cb)
    bhat = float((diff.T @ np.linalg.solve(c, diff)).item()) / 8 + 0.5 * (ld_c - 0.5 * (ld_a + ld_b))
    return float(2 * (1 - np.exp(-max(bhat, 0.0))))


def _separability(records, names, class_names, args):
    rng = np.random.default_rng(0)
    num_classes = len(class_names)
    buckets: dict[int, list[np.ndarray]] = {k: [] for k in range(num_classes)}
    per_scene = max(50, args.sep_per_class // max(1, len(records) // 4))
    for r in records:
        c = load_scene_cache(r.cache_dir, label_remap=r.label_remap)
        labels = c["labels"]
        for k in np.unique(labels):
            if k == IGNORE_INDEX or k >= num_classes:
                continue
            yy, xx = np.nonzero(labels == k)
            pick = rng.choice(len(yy), min(per_scene, len(yy)), replace=False)
            x = c["features"][:, yy[pick], xx[pick]].astype(np.float32).T
            buckets[int(k)].append(x[np.all(np.isfinite(x), axis=1)])
    data = {}
    for k, parts in buckets.items():
        if parts:
            x = np.concatenate(parts)
            if len(x) > args.sep_per_class:
                x = x[rng.choice(len(x), args.sep_per_class, replace=False)]
            if len(x) >= 500:
                data[k] = x
    present = sorted(data)
    skipped = [class_names[k] for k in range(num_classes) if k not in data]
    allx = np.concatenate([data[k] for k in present])
    mu, sd = allx.mean(0), np.maximum(allx.std(0), 1e-6)
    z = {k: (v - mu) / sd for k, v in data.items()}

    pairs, mat = [], np.full((len(present), len(present)), np.nan)
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            v = _jm(z[present[i]], z[present[j]])
            mat[i, j] = mat[j, i] = v
            pairs.append((class_names[present[i]], class_names[present[j]], round(v, 3)))
    pairs.sort(key=lambda t: t[2])
    print(f"[importance] separability: classes {[class_names[k] for k in present]}; skipped (absent/<500 px): {skipped}")
    print("[importance] weakest class pairs (JM 0 = identical .. 2 = separable; Gaussian, in-sample: optimistic):")
    best = {}
    for a, b, v in pairs[:15]:
        print(f"  {a:<26} vs {b:<26} {v:.3f}" + ("  <-- weak" if v < args.sep_weak_below else ""))
        if v < args.sep_weak_below:
            ia, ib = class_names.index(a), class_names.index(b)
            sc = [_jm(z[ia][:, [f]], z[ib][:, [f]]) for f in range(len(names))]
            top = np.argsort(sc)[::-1][:3]
            best[f"{a}|{b}"] = {names[f]: round(sc[f], 3) for f in top}
            print("      best features: " + ", ".join(f"{names[f]}={sc[f]:.2f}" for f in top))
    return {"classes": [class_names[k] for k in present], "skipped": skipped, "pairs": pairs, "best_features_weak_pairs": best}


# --------------------------------------------------------------------- main




def _write_tables(path: Path, names, targets, results, verdict, pairs, sep) -> Path:
    """One ``.npz`` of ``tbl_<table>__<column>`` 1-D columns (the export_gui / csv_export convention)."""
    out: dict[str, np.ndarray] = {"product": np.array("feature_importance")}
    labels = [lbl for lbl, _ in targets]
    out["tbl_importance__feature"] = np.array(labels)
    for m in ("permutation", "dropcolumn"):
        if m in results:
            out[f"tbl_importance__{m}_d_miou"] = np.array([results[m][l]["mean"] for l in labels], dtype=np.float32)
            out[f"tbl_importance__{m}_std"] = np.array([results[m][l]["std"] for l in labels], dtype=np.float32)
    if "shap" in results:
        out["tbl_importance__shap_mean_abs"] = np.array(
            [results["shap"]["overall"].get(l, np.nan) for l in labels], dtype=np.float32)
    out["tbl_importance__drop_candidate"] = np.array([int(verdict.get(l, False)) for l in labels], dtype=np.int8)
    for m in ("permutation", "dropcolumn"):  # per-class drop of every feature
        if m in results:
            cls = sorted({c for l in labels for d in results[m][l]["per_class"] for c in d})
            out[f"tbl_{m}_by_class__feature"] = np.array(labels)
            for c in cls:
                out[f"tbl_{m}_by_class__{c}"] = np.array([
                    np.nanmean([d.get(c, np.nan) for d in results[m][l]["per_class"]]) for l in labels], dtype=np.float32)
    if "shap" in results:
        by = results["shap"]["by_class"]
        out["tbl_shap_by_class__class"] = np.array(list(by))
        for n in names:
            out[f"tbl_shap_by_class__{n}"] = np.array([by[c][n] for c in by], dtype=np.float32)
    if pairs:
        out["tbl_correlated_pairs__feature_a"] = np.array([a for a, _, _ in pairs])
        out["tbl_correlated_pairs__feature_b"] = np.array([b for _, b, _ in pairs])
        out["tbl_correlated_pairs__r"] = np.array([r for _, _, r in pairs], dtype=np.float32)
    if sep and sep["pairs"]:
        out["tbl_class_separability__class_a"] = np.array([a for a, _, _ in sep["pairs"]])
        out["tbl_class_separability__class_b"] = np.array([b for _, b, _ in sep["pairs"]])
        out["tbl_class_separability__jm"] = np.array([v for _, _, v in sep["pairs"]], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(cli.config, cli.env_file)
    fi = config.feature_importance
    model_type = config.model.type
    if not fi.enabled:
        print("[importance] feature_importance.enabled is false -- nothing to do")
        return 0
    if model_type not in SUPPORTED:
        print(
            f"[importance] model.type={model_type!r} is not supported: this analysis was only set up and tuned for "
            f"{' / '.join(SUPPORTED)}. It has NOT been adjusted to work with {model_type}."
        )
        return 1
    methods = [m for m in METHODS if getattr(fi, m).enabled]
    if not methods:
        print("[importance] every feature_importance method is disabled -- nothing to do")
        return 0
    args = SimpleNamespace(
        methods=methods, max_scenes=fi.max_scenes, patches_per_scene=fi.patches_per_scene, batch_size=fi.batch_size, repeats=fi.permutation.repeats,
        pixels_per_scene=fi.pixels_per_scene, shap_per_class=fi.shap.per_class,
        proxy_iterations=fi.proxy_iterations, drop_threshold=fi.drop_threshold, corr_threshold=fi.corr_threshold,
        sep_per_class=fi.separability.pixels_per_class, sep_weak_below=fi.separability.weak_below,
        device=fi.device,
    )
    print(f"[importance] model={model_type} methods={methods} split={fi.split}")

    class_config = load_class_config(cli.classes)
    class_names = tuple(c.name for c in class_config.classes)
    num_classes = len(class_names)

    remap = label_remap_table(class_names, config.training.class_merge)
    records = with_label_remap(discover_scene_records(config.features.train_root), remap)
    counts = collect_class_pixel_counts_by_scene(records, num_classes)
    parts = split_scenes(
        records, counts, num_classes, ratios=tuple(config.split.ratios),
        split_seed=config.split.split_seed, split_by=config.split.split_by,
    ).scenes_by_partition
    train_r, val_r = list(parts["train"]), list(parts[fi.split])
    if not val_r:
        print(f"[importance] no scenes in split {fi.split!r}")
        return 1

    ckpt_dir = Path(config.training.checkpoint_dir)
    suffix = "pt" if model_type == "unet" else "joblib"
    ckpt = Path(fi.checkpoint_path) if fi.checkpoint_path else ckpt_dir / f"{model_type}_best.{suffix}"

    loaded = None
    if model_type == "unet":
        from landscape_change_detection_pipeline.inference.model_loader import load_model_for_inference

        loaded = load_model_for_inference(ckpt, args.device)
        names = list(loaded.feature_names)
    else:
        from landscape_change_detection_pipeline.config import all_resolved_feature_names

        names = list(all_resolved_feature_names(config.features))
    print(f"[importance] {model_type}: {len(names)} features: {names}")

    targets = _targets(names, fi.feature_groups)
    rng = np.random.default_rng(0)

    print("[importance] sampling pixels...")
    xv, yv = _sample_pixels(val_r, args.pixels_per_scene, seed=1)
    need_train_px = model_type == "catboost" or "shap" in methods or "dropcolumn" in methods
    xt = yt = None
    if need_train_px:
        xt, yt = _sample_pixels(train_r, args.pixels_per_scene, seed=0)
        print(f"[importance] pixel sample: {len(yt)} train / {len(yv)} {fi.split}")

    results: dict[str, dict] = {}
    baseline = {}

    if "permutation" in methods:
        print("[importance] --- permutation ---")
        if model_type == "unet":
            subset = val_r
            if args.max_scenes and len(subset) > args.max_scenes:
                sel = sorted(rng.choice(len(subset), args.max_scenes, replace=False))
                subset = [subset[i] for i in sel]
            base, perm = _unet_permutation(loaded, subset, args, config, num_classes, class_names, targets)
        else:
            import joblib

            cb = joblib.load(ckpt)
            base, perm = _catboost_permutation(
                lambda x: np.asarray(cb.model.predict(x)).reshape(-1).astype(np.int64),
                xv, yv, args, num_classes, class_names, targets,
            )
        baseline["permutation"] = base
        results["permutation"] = perm

    proxy = None
    keep_all = list(range(len(names)))
    if "shap" in methods or "dropcolumn" in methods:
        if model_type == "catboost":
            import joblib

            proxy = joblib.load(ckpt).model
        else:
            print("[importance] unet run: fitting a CatBoost PROXY on the pixel sample (spectral signal only)")
            proxy, keep_all = _fit_catboost(config, args, xt, yt, xv, yv, num_classes)

    if "shap" in methods:
        print("[importance] --- shap ---")
        sx, sy = (xv, yv) if model_type == "catboost" else (xt, yt)
        overall, by_class = _shap(proxy, sx, sy, names, class_names, args.shap_per_class,
                                  trained_classes=np.unique(yt) if yt is not None else None)
        results["shap"] = {"overall": dict(zip(names, map(float, overall))),
                           "by_class": {c: dict(zip(names, map(float, v))) for c, v in by_class.items()}}
        print("[importance] SHAP (class-balanced mean |value|):")
        for i in np.argsort(-overall):
            print(f"  {names[i]:<24} {overall[i]:.4f}")

    if "dropcolumn" in methods:
        print("[importance] --- drop-column ---")
        m0, p0 = _miou(yv, _full_proba_pred(proxy, keep_all, xv, num_classes), num_classes, class_names)
        baseline["dropcolumn"] = m0
        print(f"[importance] drop-column baseline miou={m0:.4f}")
        out = {}
        for i, (label, idx) in enumerate(targets, 1):
            model, keep = _fit_catboost(config, args, xt, yt, xv, yv, num_classes, drop=idx)
            m, per = _miou(yv, _full_proba_pred(model, keep, xv, num_classes), num_classes, class_names)
            out[label] = {"mean": float(m0 - m), "std": 0.0,
                          "per_class": [{cn: p0[cn] - v for cn, v in per.items()
                                         if not (np.isnan(v) or np.isnan(p0[cn]))}]}
            print(f"[importance] [{i}/{len(targets)}] drop-column {label:<34} d_miou={out[label]['mean']:+.4f}")
        results["dropcolumn"] = out

    sep = None
    if "separability" in methods:
        print("[importance] --- separability ---")
        sep = _separability(records, names, class_names, args)

    corr = np.corrcoef(xv, rowvar=False)
    pairs = [(names[i], names[j], round(float(corr[i, j]), 3))
             for i in range(len(names)) for j in range(i + 1, len(names)) if abs(corr[i, j]) >= args.corr_threshold]

    verdict = {}
    for n in names:
        # A trained net always degrades when any channel is broken, so the permutation
        # alone cannot call a feature useless: it only votes when it is the sole method
        # run. drop-column (retrained without it) and SHAP decide.
        decisive = [m for m in ("dropcolumn",) if m in results] or (["permutation"] if "shap" not in results else [])
        votes = [results[m][n]["mean"] <= args.drop_threshold for m in decisive]
        if "shap" in results:
            sh = results["shap"]["overall"]
            votes.append(sh[n] <= 0.1 * max(sh.values()))
        verdict[n] = bool(votes) and all(votes)
    drops = [n for n, v in verdict.items() if v]

    out_dir = Path(fi.output_root)
    npz = _write_tables(out_dir / "feature_importance.npz", names, targets, results, verdict, pairs, sep)
    (out_dir / "feature_importance.json").write_text(json.dumps({
        "model_type": model_type, "checkpoint": str(ckpt), "split": fi.split, "baseline": baseline,
        "proxy": model_type == "unet" and ("shap" in results or "dropcolumn" in results),
        "drop_candidates": drops, "correlated_pairs": pairs, "results": results, "separability": sep,
    }, indent=2, default=float), encoding="utf-8")

    print(f"\n[importance] drop candidates (low on every method run): {drops}")
    print(f"[importance] correlated pairs (|r| >= {args.corr_threshold}): {pairs}")
    print("[importance] remove ONE candidate (or one of a correlated pair) at a time, retrain the real model, compare mIoU.")
    print(f"[importance] tables written to {npz} (CSV via scripts/export_gui.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
