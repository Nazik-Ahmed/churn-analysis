# Customer churn analysis

Statistical analysis of 2,400 monthly subscriptions (223 churned, 9.3%) to find the factors most associated with cancellation, with effects and confidence intervals, plus a Power BI dashboard design.

**العربية:** تحليل إحصائي لانسحاب 2,400 مشترك (223 ألغوا، أي 9.3%). الاستخدام المنخفض وكثرة تذاكر الدعم وعدم حضور التهيئة هي أقوى العوامل المرتبطة بالإلغاء. التوصية: تجربة عشوائية لتدخّل استباقي على الشريحة المعرّضة (923 مشتركًا، إلغاؤهم 15.6% مقابل 5.3% لغيرهم). التفاصيل أدناه بالإنجليزية.

## Key results

| Factor | Churn with / without | Adjusted effect (pts, 95% CI) | Odds ratio (95% CI) |
|---|---|---|---|
| 12 or fewer sessions per month | 17.3% / 7.3% | +10.8 (7.2 to 14.4) | 2.94 (2.17 to 3.98) |
| 5 or more support tickets | 16.3% / 7.2% | +9.5 (6.4 to 12.9) | 2.69 (2.00 to 3.62) |
| Did not attend onboarding | 11.5% / 7.5% | +4.1 (1.8 to 6.4) | 1.66 (1.25 to 2.21) |

- The first two factors are in the top three in 100% of 2,000 bootstrap resamples. The third is not settled: onboarding was top three in 49% of resamples, the Basic plan in 32% and monthly billing in 20%.
- Payment method, discount and tenure add nothing together with the five factors (likelihood-ratio p = 0.62). Absence of evidence, not proof of no effect.
- The two usage and support effects combine: churn is 5.3% with neither, 14.2% with one, 27.6% with both.

## Why accuracy is not enough
Predicting "nobody churns" is 90.7% accurate and misses every churner. Metrics used instead, from 5-fold × 10 repeated stratified cross-validation:

| Metric | Model (95% CI) | Baseline |
|---|---|---|
| PR-AUC | 0.179 (0.143 to 0.224) | 0.091 |
| ROC-AUC | 0.681 (0.644 to 0.715) | 0.5 |
| Recall when flagging the top 20% | 41.7% (35.0 to 47.8) | 20% |
| Precision at the same cut | 19.4% (15.4 to 22.7) | 9.3% |
| Lift | 2.09x (1.75 to 2.39) | 1x |

Predictive power is modest (McFadden R² = 0.07). This ranks priorities; it does not predict individuals.

## Recommendation
Run a randomized proactive-outreach test on the at-risk segment (12 or fewer sessions, or 5 or more tickets): 923 subscribers (38.5%), churn 15.6% (13.4 to 18.1) vs 5.3% for the rest, holding 64.6% of all churners. Randomize inside the segment: roughly 337 per arm detects a drop from 15.6% to 8.6% (80% power); about 711 per arm for a 5-point drop. Generalize only if the confidence interval of the difference excludes zero and the revenue saved exceeds the cost of outreach (cost is not in the data).

## Limits
- Associations, not causes. Low usage and many tickets may be symptoms of a decision already made. The data has no measurement dates or cancellation date.
- The 12-session and 5-ticket cutoffs were found by exploring this data. They are re-selected inside every cross-validation fold (same cuts in 48 of 50 folds) and performance is unchanged, but they remain hypotheses for new data.
- Tenure is inconsistent with the dates for 716 records (signup date plus tenure falls after 2026-10-26), so it is not used as a driver. Survival analysis would need a real cancellation date.
- The outcome is binary and ignores exposure time.

## Reproduce
```bash
pip install -r requirements.txt
# put subscriptions-churn.csv in data/ (not committed; it holds customer records)
python3 churn_analysis.py --data data/subscriptions-churn.csv --out results/
```
Runs in about a minute. The seed is fixed (42), so the numbers repeat exactly. `results/results.json` holds every figure quoted above (aggregates only, no row-level data).

## Repository layout
| Path | Contents |
|---|---|
| `churn_analysis.py` | Descriptive statistics, logistic regression, bootstrap, cross-validation |
| `results/results.json` | All computed results |
| `docs/powerbi/` | Power Query script, DAX columns and measures, build guide |
| `docs/images/` | Mockups of dashboard pages 1 to 3 |
| `data/` | Place the CSV here locally (ignored by git) |
