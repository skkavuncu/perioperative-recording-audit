from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

__all__ = [
    "CHECKLIST",
    "coarsen_to_grid",
    "label_agreement",
    "resample_median",
    "screen_sampling_features",
    "paired_bootstrap_change",
    "share_closed",
    "suppress_small",
]

CHECKLIST = """\
Before attributing a cross-dataset performance loss to the model or to the patients:

A. Outcome labels
  A1. Is the outcome derived from measured values (laboratory results, vital signs)?
  A2. Are those values rounded, binned, truncated or aggregated in either dataset?
  A3. If linked raw records exist, recompute the label from them and report the
      share of raw-defined events that the shared label captures.
  A4. If not, apply the coarser recording to the finer dataset and recompute the label.

B. Predictors
  B1. Is the time resolution of repeated measurements the same in both datasets?
  B2. Does any predictor count measurements, or summarise a window (for example its minimum
      or maximum) whose number of measurements differs between datasets?
  B3. Do predictors whose distributions differ most between datasets also carry a
      large share of the model's output in the target dataset?

C. Population definition
  C1. Are there procedures in which the outcome is an expected result of the procedure?
  C2. Is their share of events similar in the two datasets?

D. Re-estimation
  D1. Change one recording difference at a time and keep the rest of the pipeline fixed.
  D2. Re-estimate the loss on the same test patients with a paired bootstrap.
  D3. Report the share of the loss closed by each step, and what remains unexplained.
"""


def coarsen_to_grid(values, grid):
    v = np.asarray(values, dtype=float)
    g = np.unique(np.asarray(grid, dtype=float))
    g = g[np.isfinite(g)]
    if g.size == 0:
        raise ValueError("grid has no finite values")
    out = np.full_like(v, np.nan)
    ok = np.isfinite(v)
    pos = np.searchsorted(g, v[ok])
    lo = np.clip(pos - 1, 0, g.size - 1)
    hi = np.clip(pos, 0, g.size - 1)
    pick_hi = np.abs(g[hi] - v[ok]) < np.abs(v[ok] - g[lo])
    out[ok] = np.where(pick_hi, g[hi], g[lo])
    return out


def label_agreement(y_reference, y_shared, min_cell=10):
    a = np.asarray(y_reference).astype(int)
    b = np.asarray(y_shared).astype(int)
    if a.shape != b.shape:
        raise ValueError("labels must have the same length")
    n = a.size
    tp = int(((a == 1) & (b == 1)).sum())
    fn = int(((a == 1) & (b == 0)).sum())
    fp = int(((a == 0) & (b == 1)).sum())
    tn = n - tp - fn - fp
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return {
        "n": n,
        "rate_reference": (tp + fn) / n,
        "rate_shared": (tp + fp) / n,
        "capture": tp / (tp + fn) if tp + fn else np.nan,
        "events_missed": suppress_small(fn, min_cell),
        "events_added": suppress_small(fp, min_cell),
        "kappa": kappa,
    }


def resample_median(df, step_min, time="t_min", value="value",
                    group="case_id", source=None, preferred=None):
    d = df[[group, time, value] + ([source] if source else [])].dropna(subset=[time, value]).copy()
    d["_bin"] = np.floor(d[time] / step_min).astype(int)
    if source and preferred is not None:
        has_pref = (d[source] == preferred).groupby([d[group], d["_bin"]]).transform("any")
        d = d[~has_pref | (d[source] == preferred)]
    out = d.groupby([group, "_bin"], as_index=False)[value].median()
    if source and preferred is not None:
        src = d.groupby([group, "_bin"])[source].first().reset_index(drop=True)
        out[source] = src.values
    out[time] = (out["_bin"] + 1) * step_min
    return out.drop(columns="_bin")[[group, time, value] + ([source] if source and preferred is not None else [])]


def screen_sampling_features(X_source, X_target, coef=None, top=10):
    Xs = pd.DataFrame(X_source).astype(float)
    Xt = pd.DataFrame(X_target, columns=Xs.columns).astype(float)
    mu = Xs.mean()
    sd = Xs.std(ddof=0).replace(0, np.nan)
    pooled = np.sqrt((Xs.var(ddof=0) + Xt.var(ddof=0)) / 2).replace(0, np.nan)
    res = pd.DataFrame({
        "mean_source": mu,
        "mean_target": Xt.mean(),
        "smd": (Xt.mean() - mu) / pooled,
    })
    if coef is not None:
        c = pd.Series(np.asarray(coef, dtype=float), index=Xs.columns)
        zs = (Xs - mu) / sd
        zt = (Xt - mu) / sd
        res["contrib_source"] = (zs.mul(c)).abs().mean()
        res["contrib_target"] = (zt.mul(c)).abs().mean()
        res["contrib_ratio"] = res["contrib_target"] / res["contrib_source"]
        order = res["contrib_target"].abs()
    else:
        order = res["smd"].abs()
    return res.loc[order.sort_values(ascending=False).index].head(top)


def paired_bootstrap_change(y, p_within, p_cross, groups=None, B=1000, seed=0,
                            metric=roc_auc_score):
    y = np.asarray(y)
    p1 = np.asarray(p_within, dtype=float)
    p2 = np.asarray(p_cross, dtype=float)
    est = metric(y, p2) - metric(y, p1)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(y.size)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(uniq.size)]
    reps = []
    for _ in range(B):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([members[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        reps.append(metric(y[idx], p2[idx]) - metric(y[idx], p1[idx]))
    reps = np.asarray(reps)
    return {"change": est,
            "ci_low": float(np.quantile(reps, 0.025)),
            "ci_high": float(np.quantile(reps, 0.975)),
            "replicates": int(reps.size)}


def share_closed(change_before, change_after):
    b = float(change_before)
    a = float(change_after)
    if b >= 0:
        return np.nan
    return (b - a) / b


def suppress_small(count, min_cell=10):
    count = int(count)
    if 0 < count < min_cell:
        return f"<{min_cell}"
    return count
