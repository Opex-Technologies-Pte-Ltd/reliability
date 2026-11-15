# @opexcg/reliability

Reliability and condition-monitoring math for TypeScript and JavaScript:
Weibull life data analysis (with suspended units), B-lives, remaining useful
life, Weibull proportional hazards, MTBF / MTTR / availability, and PCA
anomaly detection with Hotelling T² and SPE control limits.

- No runtime dependencies. Runs in Node 18+, Bun, Deno, browsers and edge
  runtimes such as Cloudflare Workers.
- ESM with TypeScript declarations.
- Deterministic: no random numbers, the same input always gives the same output.
- All times are in **hours** unless a name says otherwise.

```sh
npm install @opexcg/reliability
```

## Examples

Fit a Weibull distribution to failure times and read off the B10 life:

```ts
import { fitWeibull, bLife, weibullMean } from '@opexcg/reliability';

const fit = fitWeibull([16, 34, 53, 75, 93, 120]); // hours to failure
// fit.shape ≈ 1.43 (β), fit.scale ≈ 76.3 h (η), fit.rSquared ≈ 0.991
bLife(fit!, 10);   // ≈ 15.8 h: 10 % of units have failed by this age
weibullMean(fit!); // ≈ 69.4 h mean life
```

Units that were removed or are still running without having failed
(suspensions, right-censored data) are passed as events:

```ts
import { fitWeibull, remainingUsefulLife } from '@opexcg/reliability';

const fit = fitWeibull(
  [
    { hours: 150, failed: true },
    { hours: 560, failed: false }, // replaced at a planned overhaul
    { hours: 800, failed: true },
    // ...
  ],
  { method: 'mle' },
);
if (fit) {
  const rul = remainingUsefulLife(fit, 400); // a unit that has run 400 h
  rul.medianHours;                           // median remaining life
  rul.interval;                              // 90 % interval { lowerHours, upperHours }
}
```

`fitWeibull` returns `null` when the data cannot support a fit (for example
fewer than two failures), and throws a `ReliabilityError` for invalid input
such as negative or non-finite times.

Maintenance KPIs from down events:

```ts
import { summarizeDowntime } from '@opexcg/reliability';

const kpi = summarizeDowntime(
  [{ start: new Date('2026-03-02T08:00Z'), end: new Date('2026-03-02T11:00Z') }],
  { start: new Date('2026-03-01T00:00Z'), end: new Date('2026-04-01T00:00Z') },
);
kpi.mtbfHours; kpi.mttrHours; kpi.availability;
```

Multivariate condition monitoring with PCA:

```ts
import { trainPca, scorePca, isPcaAnomalous } from '@opexcg/reliability';

const model = trainPca(normalReadings); // rows = observations, columns = sensors
const score = scorePca(model!, latestReading);
isPcaAnomalous(model!, score!); // SPE or Hotelling T² over its 99 % limit
score!.contributors[0];         // the sensor that contributes most to SPE
```

## API

### Weibull distribution (`WeibullParams = { shape, scale }`, scale in hours)

| Function | Returns |
|---|---|
| `weibullPdf(p, hours)` | probability density f(t), per hour |
| `weibullCdf(p, hours)` | probability of failure by t, F(t) |
| `weibullReliability(p, hours)` | probability of surviving past t, R(t) |
| `weibullHazard(p, hours)` | failure rate h(t), failures per hour |
| `weibullMean(p)` | mean life η·Γ(1 + 1/β) in hours (MTTF; MTBF under as-good-as-new repair) |
| `weibullQuantile(p, probability)` | age by which that fraction has failed |
| `bLife(p, percent)` | B-life, e.g. `bLife(p, 10)` = B10 |
| `conditionalReliability(p, ageHours, missionHours)` | R(age + mission) / R(age) |
| `remainingUsefulLife(model, ageHours, { confidence })` | median remaining life given survival to `ageHours`, with a delta-method interval when the model has a covariance |

### Fitting

| Function | Description |
|---|---|
| `fitWeibull(data, { method, minFailures })` | `data` is `number[]` (failure times) or `{ hours, failed }[]`. `method: 'mrr'` (default): median rank regression on Y with Benard's median ranks and Johnson's adjusted ranks for suspensions. `method: 'mle'`: maximum likelihood with right censoring. Returns shape, scale, failures, suspensions, `rSquared` (MRR), `logLikelihood`, `covariance` of `[ln η, ln β]`, and the plotted rank points (MRR). |
| `medianRanks(data)` | Johnson adjusted ranks and Benard median ranks, for Weibull plots |
| `rankRegressionCovariance(shape, failures)` | the approximate covariance used for MRR intervals |

### Proportional hazards (covariates)

| Function | Description |
|---|---|
| `fitWeibullPh(samples, covariateNames, { warmStart, minFailures })` | Weibull proportional-hazards MLE: hazard scaled by exp(γ'x). Supports right censoring and time-varying covariates (counting-process pieces with `startHours`). Reports hazard ratios per standard deviation and their standard errors. |
| `weibullPhScale(fit, x)` | Weibull scale for a unit with covariates `x` |
| `weibullPhLinearPredictor(fit, x)` | γ'x on standardized covariates |
| `weibullPhRemainingLife(fit, x, ageHours, { confidence })` | median remaining life with an interval from the full covariance |
| `compareSurvivalModels(cases, models)` | Harrell's concordance index and median absolute percentage error of remaining-life predictors on labelled cases |
| `weibullPhPredictor(fit)` | adapter from a PH fit to `compareSurvivalModels` |

### Maintenance metrics

| Function | Description |
|---|---|
| `meanTimeBetweenFailures(operatingHours, failures)` | MTBF, `null` without failures |
| `meanTimeToRepair(repairHours[])` | MTTR, `null` for an empty list |
| `failureRate(failures, operatingHours)` | failures per hour |
| `inherentAvailability(mtbfHours, mttrHours)` | MTBF / (MTBF + MTTR) |
| `timeBetweenFailures(timestamps)` | hours between consecutive failures |
| `summarizeDowntime(events, window)` | observed, down and up hours, failures, MTBF, MTTR and availability over a window (overlaps merged, events clipped) |

### PCA condition monitoring

| Function | Description |
|---|---|
| `trainPca(rows, { minSamples, varianceToKeep, speQuantile, t2Confidence })` | standardizes features, keeps components up to 90 % of variance (at most d − 1), sets the SPE limit (99 % training quantile) and the Hotelling T² limit (F approximation) |
| `scorePca(model, reading)` | SPE, T², reconstruction and per-feature contributions |
| `isPcaAnomalous(model, score)` | SPE or T² over its limit |
| `hotellingT2Limit(k, n, confidence)` | T² control limit for k components and n training rows |
| `symmetricEigen(matrix)` | Jacobi eigen-decomposition |

### Numerical helpers

`gamma`, `lnGamma`, `incompleteBeta`, `fCdf`, `fQuantile`, `normalQuantile`, `empiricalQuantile`.

### Errors

Invalid input throws `ReliabilityError` with a `code` of `'INVALID_INPUT'`,
`'INVALID_PARAMETER'` or `'DIMENSION_MISMATCH'`. Not having enough data to
fit is not an error: fitting functions return `null`.

## Accuracy

The tests compare results with independent implementations: scipy
(`weibull_min`, `norm.ppf`, `f.ppf`, `betainc`, `gammaln`), lifelines
(`WeibullFitter` estimates, log-likelihood and covariance with censored data)
and numpy (eigenvalues, quantiles). The script that produces the reference
values is [`scripts/oracle.py`](https://github.com/Opex-Technologies-Pte-Ltd/reliability/blob/main/scripts/oracle.py). Adjusted ranks are
checked against the worked examples in ReliaSoft's Life Data Analysis Reference.

## Limitations

- Two-parameter Weibull only: no location (γ, failure-free time)
  parameter, no mixed Weibull, lognormal or other life distributions.
- Right censoring only (plus left truncation in the PH fit); no interval censoring.
- Intervals on MRR fits use a rough approximation (`rankRegressionCovariance`).
  For intervals you rely on, fit with `method: 'mle'`. MLE intervals use the
  Fisher-matrix / delta method, which is optimistic for small samples;
  likelihood-ratio bounds are not implemented.
- `fitWeibullPh` uses finite-difference derivatives and is meant for a few
  covariates and a few thousand observations.
- No reliability-growth models (Crow-AMSAA / Duane) yet.
- `trainPca` uses Jacobi rotations, fine for tens of features, not for thousands.

## Used in OpexMX

This package holds the reliability math of [OpexMX](https://mx.opexcg.com), a
maintenance management system, where it powers the Weibull failure predictions,
remaining-useful-life estimates and multivariate sensor anomaly detection.

## License

Apache-2.0. Issues and pull requests are welcome at
https://github.com/Opex-Technologies-Pte-Ltd/reliability.
