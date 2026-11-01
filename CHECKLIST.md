# Checklist

Questions to answer before a cross-dataset performance loss is attributed to the model or to the patients.

## A. Outcome labels

1. Is the outcome derived from measured values (laboratory results, vital signs)?
2. Are those values rounded, binned, truncated or aggregated in either dataset?
3. If linked raw records exist, recompute the label from them and report the share of raw-defined events that the shared label captures.
4. If not, apply the coarser recording to the finer dataset and recompute the label.

## B. Predictors

1. Is the time resolution of repeated measurements the same in both datasets?
2. Does any predictor count measurements, or summarise a window (for example its minimum or maximum) whose number of measurements differs between datasets?
3. Do predictors whose distributions differ most between datasets also carry a large share of the model's output in the target dataset?

## C. Population definition

1. Are there procedures in which the outcome is an expected result of the procedure?
2. Is their share of events similar in the two datasets?

## D. Re-estimation

1. Change one recording difference at a time and keep the rest of the pipeline fixed.
2. Re-estimate the loss on the same test patients with a paired bootstrap.
3. Report the share of the loss closed by each step, and what remains unexplained.
