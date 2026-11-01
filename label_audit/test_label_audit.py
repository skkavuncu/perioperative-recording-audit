import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import label_audit as la


def test_coarsen_to_grid():
    g = [0.5, 0.8, 1.2, 1.68, 3.03]
    out = la.coarsen_to_grid([0.6, 0.66, 1.0, 2.0, 2.5, np.nan, 9], g)
    assert np.allclose(out[:5], [0.5, 0.8, 0.8, 1.68, 3.03])
    assert np.isnan(out[5]) and out[6] == 3.03


def test_coarsening_hides_small_rises():
    grid = [0.6, 0.9, 1.3, 1.68, 3.03, 5.54]
    base, post = np.array([1.70, 1.80]), np.array([2.05, 2.20])
    raw = (post - base) >= 0.3
    rb, rp = la.coarsen_to_grid(base, grid), la.coarsen_to_grid(post, grid)
    coarse = (rp - rb) >= 0.3
    assert raw.tolist() == [True, True]
    assert coarse.tolist() == [False, False]


def test_label_agreement():
    a = np.array([1] * 50 + [0] * 950)
    b = a.copy(); b[:10] = 0; b[60:63] = 1
    r = la.label_agreement(a, b)
    assert abs(r["capture"] - 0.8) < 1e-9
    assert r["events_missed"] == 10 and r["events_added"] == "<10"
    assert 0.8 < r["kappa"] < 0.9


def test_resample_median_end_stamp_and_priority():
    df = pd.DataFrame({"case_id": 1, "t_min": [0, 1, 2, 3, 4, 5, 6],
                       "value": [70, 60, 80, 90, 100, 50, 55],
                       "src": ["nibp", "art", "art", "nibp", "nibp", "nibp", "nibp"]})
    r = la.resample_median(df, 5, source="src", preferred="art")
    assert r["t_min"].tolist() == [5, 10]
    assert r["value"].tolist() == [70.0, 52.5]
    assert r["src"].tolist() == ["art", "nibp"]


def test_screen_flags_count_feature():
    rng = np.random.default_rng(1)
    n = 4000
    Xs = pd.DataFrame({"map": rng.normal(80, 10, n), "n_meas": rng.poisson(3, n)})
    y = (rng.random(n) < 1 / (1 + np.exp((Xs["map"] - 70) / 4) )).astype(int)
    y[Xs["n_meas"] > 5] = 1
    m = LogisticRegression(max_iter=1000).fit((Xs - Xs.mean()) / Xs.std(ddof=0), y)
    Xt = pd.DataFrame({"map": rng.normal(80, 10, n), "n_meas": rng.poisson(14, n)})
    r = la.screen_sampling_features(Xs, Xt, coef=m.coef_[0])
    assert r.index[0] == "n_meas" and r.loc["n_meas", "contrib_ratio"] > 3


def test_paired_bootstrap_and_share():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.6, y.size)
    worse = y + rng.normal(0, 1.2, y.size)
    r = la.paired_bootstrap_change(y, good, worse, groups=np.arange(y.size) // 2, B=200)
    assert r["change"] < 0 and r["ci_high"] < 0
    assert abs(la.share_closed(-0.129, -0.056) - 0.566) < 1e-3
    assert np.isnan(la.share_closed(0.01, 0.0))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
