# Perioperative recording audit

A checklist and a small Python package for checking whether a drop in model performance between two clinical datasets comes from how outcomes and predictors were recorded, before the drop is attributed to the model or to the patients.

They accompany the paper *A linked-record audit shows that recording and de-identification explain part of the cross-hospital loss of perioperative prediction models*.

## Contents

- `CHECKLIST.md`: twelve questions on outcome labels, predictors, the population and re-estimation
- `label_audit/label_audit.py`: functions used in each step
- `label_audit/test_label_audit.py`: tests on synthetic data

## The three steps

1. Audit the shared label against linked raw records. Recompute the outcome from raw values for cases present in both sources and report how many raw-defined events the shared label captures (`coarsen_to_grid`, `label_agreement`).
2. Harmonise recording. Apply the coarser recording of one dataset to the other: round values to the released grid, resample repeated measurements to the same time resolution, and screen predictors that carry measurement frequency (`coarsen_to_grid`, `resample_median`, `screen_sampling_features`).
3. Re-estimate the loss on the same test patients with a paired, patient-level bootstrap and report the share of the loss closed by each step (`paired_bootstrap_change`, `share_closed`).

## Requirements

Python 3.9 or later with numpy, pandas and scikit-learn.

```
pip install -r requirements.txt
```

## Example

```python
import label_audit as la

grid = released_creatinine_values
y_raw = aki_from(raw_creatinine)
y_coarse = aki_from(la.coarsen_to_grid(raw_creatinine, grid))
print(la.label_agreement(y_raw, y_coarse))

bp5 = la.resample_median(bp, 5, time="t_min", value="map",
                         group="case_id", source="site", preferred="arterial")

r = la.paired_bootstrap_change(y_test, p_within, p_cross, groups=patient_id, B=1000)
print(r, la.share_closed(change_before, r["change"]))
```

Here `aki_from` stands for your own label function, and `released_creatinine_values` for the distinct values found in the coarser dataset.

All functions return aggregate numbers. `label_agreement` shows counts between 1 and 9 as `<10`, so the output can be shared under data use agreements that forbid releasing small cells.

## Tests

```
cd label_audit
python test_label_audit.py
```

## License

MIT
