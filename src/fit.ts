/**
 * Fitting a two-parameter Weibull distribution to life data, with or without
 * right-censored units (suspensions: units removed or still running without
 * having failed).
 */
import { ReliabilityError } from './errors.js';
import type { LogParamCovariance, WeibullParams } from './weibull.js';

/** One unit's life: hours run, and whether it ended in a failure (else a suspension). */
export interface LifeEvent {
  hours: number;
  failed: boolean;
}

/** Plain numbers are failure times in hours (complete data); use {@link LifeEvent}s for suspensions. */
export type LifeData = readonly number[] | readonly LifeEvent[];

export type WeibullFitMethod = 'mrr' | 'mle';

export interface WeibullFitOptions {
  /**
   * `'mrr'` (default): median rank regression, regressing ln(−ln(1−F)) on ln t
   * ("rank regression on Y"), Benard's median ranks, Johnson's adjusted ranks
   * for suspensions. Usual choice for small samples and Weibull plots.
   *
   * `'mle'`: maximum likelihood with right censoring. Usual choice for heavily
   * censored data and the basis of likelihood-based confidence bounds.
   */
  method?: WeibullFitMethod;
  /** Return `null` when there are fewer failures than this. Default 2, never below 2. */
  minFailures?: number;
}

/** One plotted point of a rank regression. */
export interface RankPoint {
  /** Failure time in hours. */
  hours: number;
  /** Johnson's adjusted rank (equals the plain order number for complete data). */
  adjustedRank: number;
  /** Benard's median rank, an estimate of F(hours). */
  medianRank: number;
}

export interface WeibullFit extends WeibullParams {
  method: WeibullFitMethod;
  failures: number;
  suspensions: number;
  /** Coefficient of determination of the rank regression (MRR only, else null). */
  rSquared: number | null;
  /** Log-likelihood of the data at the fitted parameters (null if undefined, e.g. a failure at t = 0). */
  logLikelihood: number | null;
  /**
   * Covariance of `[ln scale, ln shape]`. MLE: inverse observed Fisher information.
   * MRR: the approximation from {@link rankRegressionCovariance}.
   */
  covariance: LogParamCovariance | null;
  /** Points used by the rank regression (MRR only, else empty). */
  points: RankPoint[];
}

function normalise(data: LifeData): LifeEvent[] {
  if (!Array.isArray(data)) {
    throw new ReliabilityError('INVALID_INPUT', 'life data must be an array');
  }
  return (data as readonly (number | LifeEvent)[]).map((d, i) => {
    const ev: LifeEvent = typeof d === 'number' ? { hours: d, failed: true } : d;
    if (!ev || typeof ev.hours !== 'number' || !Number.isFinite(ev.hours) || ev.hours < 0) {
      throw new ReliabilityError(
        'INVALID_INPUT',
        `life data[${i}].hours must be a finite number >= 0, got ${ev?.hours}`,
      );
    }
    return { hours: ev.hours, failed: ev.failed === true };
  });
}

/**
 * Fit a two-parameter Weibull distribution (hours) to failure and suspension times.
 *
 * Returns `null`, rather than throwing, when the data cannot support a fit:
 * fewer than `minFailures` failures, all failures at the same time, or a
 * non-positive slope. Throws {@link ReliabilityError} for invalid input
 * (negative or non-finite times; for MLE also a failure at t = 0).
 *
 * MRR note: a failure at t = 0 keeps its rank but cannot be placed on the
 * log-time axis, so it is left out of the regression.
 */
export function fitWeibull(data: LifeData, options: WeibullFitOptions = {}): WeibullFit | null {
  const events = normalise(data);
  const minFailures = Math.max(2, options.minFailures ?? 2);
  const failures = events.filter((e) => e.failed).length;
  if (failures < minFailures) return null;
  const method = options.method ?? 'mrr';
  if (method === 'mrr') return fitMrr(events, failures);
  if (method === 'mle') return fitMle(events, failures);
  throw new ReliabilityError('INVALID_INPUT', `unknown fit method '${String(method)}'`);
}

/**
 * Johnson's adjusted ranks and Benard's median ranks for the failures in
 * `data`. Units are sorted by time with failures before suspensions on ties.
 *
 *   reverse rank r_i = number of units at or after position i (including i)
 *   adjusted rank  MON_i = MON_prev + (N + 1 − MON_prev) / (1 + r_i)
 *   median rank    F_i = (MON_i − 0.3) / (N + 0.4)
 *
 * For complete data the adjusted ranks are 1, 2, …, n.
 */
export function medianRanks(data: LifeData): RankPoint[] {
  return rankPoints(normalise(data));
}

function rankPoints(events: LifeEvent[]): RankPoint[] {
  const sorted = [...events].sort(
    (a, b) => a.hours - b.hours || (a.failed === b.failed ? 0 : a.failed ? -1 : 1),
  );
  const n = sorted.length;
  const points: RankPoint[] = [];
  let mon = 0;
  for (let i = 0; i < n; i++) {
    const ev = sorted[i]!;
    if (!ev.failed) continue; // suspensions only shift the following ranks
    const reverseRank = n - i;
    mon += (n + 1 - mon) / (1 + reverseRank);
    points.push({ hours: ev.hours, adjustedRank: mon, medianRank: (mon - 0.3) / (n + 0.4) });
  }
  return points;
}

/**
 * Approximate covariance of `[ln η, ln β]` for a rank-regression fit with
 * `failures` failures: Var(ln η) ≈ Σ_{k=1..n} 1/k² / β², Var(ln β) ≈ 1/n,
 * no covariance. A rough, conservative heuristic (rank regression has no
 * likelihood to derive bounds from); prefer an MLE fit when the interval matters.
 */
export function rankRegressionCovariance(shape: number, failures: number): LogParamCovariance {
  if (!(shape > 0) || !(failures >= 1)) {
    throw new ReliabilityError('INVALID_INPUT', 'rankRegressionCovariance needs shape > 0 and failures >= 1');
  }
  let varLnEta = 0;
  for (let k = 1; k <= failures; k++) varLnEta += 1 / (k * k);
  varLnEta /= shape * shape;
  return [
    [varLnEta, 0],
    [0, 1 / failures],
  ];
}

function fitMrr(events: LifeEvent[], failures: number): WeibullFit | null {
  const all = rankPoints(events);
  const points: RankPoint[] = [];
  const xs: number[] = [];
  const ys: number[] = [];
  for (const p of all) {
    if (!(p.medianRank > 0 && p.medianRank < 1) || !(p.hours > 0)) continue;
    const x = Math.log(p.hours);
    const y = Math.log(-Math.log(1 - p.medianRank));
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    xs.push(x);
    ys.push(y);
    points.push(p);
  }
  if (xs.length < 2) return null;
  const fit = ordinaryLeastSquares(xs, ys);
  if (!fit || !Number.isFinite(fit.slope) || fit.slope <= 0) return null;
  const shape = fit.slope;
  const scale = Math.exp(-fit.intercept / fit.slope);
  if (!Number.isFinite(scale) || scale <= 0) return null;
  return {
    method: 'mrr',
    shape,
    scale,
    failures,
    suspensions: events.length - failures,
    rSquared: fit.rSquared,
    logLikelihood: logLikelihood(events, shape, scale),
    covariance: rankRegressionCovariance(shape, failures),
    points,
  };
}

/** @internal Simple least squares y = slope·x + intercept; null for a degenerate x. */
export function ordinaryLeastSquares(
  x: readonly number[],
  y: readonly number[],
): { slope: number; intercept: number; rSquared: number } | null {
  const n = x.length;
  if (n < 2) return null;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i]!;
    sumY += y[i]!;
    sumXY += x[i]! * y[i]!;
    sumX2 += x[i]! * x[i]!;
  }
  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return null;
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  const meanY = sumY / n;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    ssTot += (y[i]! - meanY) ** 2;
    ssRes += (y[i]! - (slope * x[i]! + intercept)) ** 2;
  }
  return { slope, intercept, rSquared: ssTot > 0 ? 1 - ssRes / ssTot : 0 };
}

/** Weibull log-likelihood with right censoring; null when not finite. */
function logLikelihood(events: LifeEvent[], shape: number, scale: number): number | null {
  let ll = 0;
  for (const e of events) {
    const z = e.hours / scale;
    if (e.failed) ll += Math.log(shape / scale) + (shape - 1) * Math.log(z);
    ll -= z ** shape;
  }
  return Number.isFinite(ll) ? ll : null;
}

/**
 * MLE. With a = ln η, the shape solves the profile equation
 *   g(β) = Σ t^β ln t / Σ t^β − 1/β − (1/r)·Σ_failures ln t = 0
 * (sums over all units unless noted), which is increasing in β; then
 *   η = (Σ t^β / r)^(1/β).
 * Times are divided by the largest time first so t^β cannot overflow.
 */
function fitMle(events: LifeEvent[], failures: number): WeibullFit | null {
  if (events.some((e) => e.failed && e.hours === 0)) {
    throw new ReliabilityError('INVALID_INPUT', 'MLE needs failure times > 0');
  }
  const used = events.filter((e) => e.hours > 0); // a suspension at 0 carries no information
  const tMax = Math.max(...used.map((e) => e.hours));
  const lnT = used.map((e) => Math.log(e.hours / tMax));
  const failed = used.map((e) => e.failed);
  const r = failures;
  let meanLnFail = 0;
  for (let i = 0; i < lnT.length; i++) if (failed[i]) meanLnFail += lnT[i]!;
  meanLnFail /= r;

  const g = (beta: number): number => {
    let s0 = 0;
    let s1 = 0;
    for (const l of lnT) {
      const w = Math.exp(beta * l);
      s0 += w;
      s1 += w * l;
    }
    return s1 / s0 - 1 / beta - meanLnFail;
  };

  // Bracket the root on a log grid, then bisect in ln β.
  let lo = 1e-3;
  let hi = 1;
  while (g(lo) > 0) {
    lo /= 10;
    if (lo < 1e-12) return null;
  }
  while (g(hi) < 0) {
    hi *= 2;
    if (hi > 1e4) return null; // no finite root: all failures at one time, nothing later
  }
  let lnLo = Math.log(lo);
  let lnHi = Math.log(hi);
  for (let i = 0; i < 200 && lnHi - lnLo > 1e-14; i++) {
    const mid = (lnLo + lnHi) / 2;
    if (g(Math.exp(mid)) < 0) lnLo = mid;
    else lnHi = mid;
  }
  const shape = Math.exp((lnLo + lnHi) / 2);
  let s0 = 0;
  for (const l of lnT) s0 += Math.exp(shape * l);
  const scale = tMax * (s0 / r) ** (1 / shape);
  if (!Number.isFinite(shape) || !Number.isFinite(scale) || scale <= 0) return null;

  return {
    method: 'mle',
    shape,
    scale,
    failures,
    suspensions: events.length - failures,
    rSquared: null,
    logLikelihood: logLikelihood(events, shape, scale),
    covariance: mleCovariance(used, shape, scale),
    points: [],
  };
}

/**
 * Inverse observed information for θ = [ln η, ln β]. With z = ln t − ln η and
 * u = e^{βz}, the log-likelihood is ℓ = r·ln β − r·ln η + (β−1)·Σ_f z − Σ u, and
 *   −∂²ℓ/∂a²   = β²·Σu
 *   −∂²ℓ/∂a∂b = −β(Σu − r) − β²·Σ z·u
 *   −∂²ℓ/∂b²   = −β·Σ_f z + β·Σ z·u + β²·Σ z²·u
 */
function mleCovariance(events: LifeEvent[], shape: number, scale: number): LogParamCovariance | null {
  const lnEta = Math.log(scale);
  let r = 0;
  let sumU = 0;
  let sumZU = 0;
  let sumZ2U = 0;
  let sumZf = 0;
  for (const e of events) {
    const z = Math.log(e.hours) - lnEta;
    const u = Math.exp(shape * z);
    sumU += u;
    sumZU += z * u;
    sumZ2U += z * z * u;
    if (e.failed) {
      r++;
      sumZf += z;
    }
  }
  const b = shape;
  const iaa = b * b * sumU;
  const iab = -b * (sumU - r) - b * b * sumZU;
  const ibb = -b * sumZf + b * sumZU + b * b * sumZ2U;
  const det = iaa * ibb - iab * iab;
  if (!(det > 0) || !Number.isFinite(det)) return null;
  return [
    [ibb / det, -iab / det],
    [-iab / det, iaa / det],
  ];
}
