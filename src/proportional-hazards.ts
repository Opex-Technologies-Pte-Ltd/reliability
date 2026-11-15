/**
 * Weibull proportional-hazards regression: a Weibull life model whose hazard
 * is scaled by covariates (for example a condition or health score),
 *
 *   h_i(t) = (p/η₀)·(t/η₀)^(p−1) · exp(γ'x_i)   ⇔   S_i(t) = exp(−(t/η₀)^p · e^{γ'x_i})
 *
 * equivalently a per-unit scale η_i = η₀·exp(−γ'x_i / p). Covariates are
 * standardized inside the fit, so exp(γ_j) is the hazard ratio per standard
 * deviation of covariate j.
 *
 * Fitted by maximum likelihood with right censoring and left truncation
 * (counting-process pieces), using a damped Newton solver with
 * finite-difference derivatives. Deterministic: fixed starting point, no RNG.
 * Intended for a handful of covariates; cost is O(n·(2+k)²) per iteration.
 */
import { ReliabilityError, assertNonNegative } from './errors.js';
import type { RemainingLife, RemainingLifeOptions } from './weibull.js';
import { zForConfidence } from './weibull.js';

/**
 * One observation piece with its covariate vector (raw, unstandardized).
 * The unit is at risk over (startHours, hours] with the covariates held
 * constant, and `failed` marks a failure at `hours`. A whole life is the case
 * startHours = 0. Time-varying covariates split one life into consecutive
 * pieces that share a `group` id; only the last piece may carry the failure.
 */
export interface PhSample {
  hours: number;
  failed: boolean;
  x: number[];
  /** Entry time on the same clock as `hours` (default 0). */
  startHours?: number;
  /** Unit id: pieces of one life share it (used to count suspensions). */
  group?: string;
}

export interface WeibullPhFit {
  /** Weibull shape p, shared by all units. */
  shape: number;
  /** Baseline scale η₀ in hours: the scale of a unit at the mean covariate vector. */
  baseScale: number;
  /** Coefficients on standardized covariates. */
  coef: number[];
  /** Standard errors of `coef` from the inverse observed information (null if singular). */
  coefSe: Array<number | null>;
  covariateNames: string[];
  covariateMeans: number[];
  covariateStdDevs: number[];
  /** exp(coef): hazard ratio per standard deviation of each covariate. */
  hazardRatios: number[];
  logLik: number;
  iterations: number;
  converged: boolean;
  failures: number;
  suspensions: number;
  /** Inverse-Hessian covariance of [ln η₀, ln p, γ…] at the optimum (null if singular). */
  cov: number[][] | null;
}

export interface WeibullPhOptions {
  /** Starting values, e.g. from a plain Weibull fit; improves stability. */
  warmStart?: { shape: number; baseScale: number };
  /** Return null with fewer failures than this. Default 5. */
  minFailures?: number;
}

const MAX_NEWTON_ITERS = 100;
const STEP_TOL = 1e-8;
const NLL_TOL = 1e-10;
// Keep the solver inside a sane region; outside it the objective is a wall.
const BOUND_LN_ETA = 25;
const BOUND_LN_P = 5;
const BOUND_GAMMA = 40;

/**
 * Fit a Weibull proportional-hazards model by maximum likelihood.
 *
 * Returns null when there are too few failures or the solver fails to produce
 * a finite optimum (check `converged` on a returned fit). Throws
 * {@link ReliabilityError} for malformed samples: non-positive or non-finite
 * `hours`, `startHours` outside [0, hours), non-finite covariates, or a
 * covariate vector of the wrong length.
 */
export function fitWeibullPh(
  samples: readonly PhSample[],
  covariateNames: readonly string[],
  options: WeibullPhOptions = {},
): WeibullPhFit | null {
  const k = covariateNames.length;
  if (k === 0) throw new ReliabilityError('INVALID_INPUT', 'at least one covariate name is required');
  samples.forEach((s, i) => {
    if (!(s.hours > 0) || !Number.isFinite(s.hours)) {
      throw new ReliabilityError('INVALID_INPUT', `samples[${i}].hours must be a finite number > 0`);
    }
    const start = s.startHours ?? 0;
    if (!(start >= 0) || start >= s.hours) {
      throw new ReliabilityError('INVALID_INPUT', `samples[${i}].startHours must be in [0, hours)`);
    }
    if (!Array.isArray(s.x) || s.x.length !== k) {
      throw new ReliabilityError('DIMENSION_MISMATCH', `samples[${i}].x must have ${k} values`);
    }
    if (s.x.some((v) => !Number.isFinite(v))) {
      throw new ReliabilityError('INVALID_INPUT', `samples[${i}].x must be finite numbers`);
    }
  });
  if (samples.length < 2) return null;
  const failures = samples.filter((s) => s.failed).length;
  if (failures < Math.max(1, options.minFailures ?? 5)) return null;

  // Standardize covariates; a zero-variance column gets a floor.
  const means = new Array<number>(k).fill(0);
  const stds = new Array<number>(k).fill(1);
  for (let j = 0; j < k; j++) {
    const col = samples.map((s) => s.x[j]!);
    means[j] = col.reduce((a, b) => a + b, 0) / col.length;
    const varj = col.reduce((a, b) => a + (b - means[j]!) ** 2, 0) / (col.length - 1);
    stds[j] = Math.max(Math.sqrt(Math.max(varj, 0)), 1e-9);
  }
  const xs = samples.map((s) => s.x.map((v, j) => (v - means[j]!) / stds[j]!));

  // Negative log-likelihood over θ = [ln η₀, ln p, γ…].
  const nll = (theta: number[]): number => {
    const a = theta[0]!;
    const b = theta[1]!;
    if (!Number.isFinite(a) || !Number.isFinite(b)) return Number.POSITIVE_INFINITY;
    if (Math.abs(a) > BOUND_LN_ETA || Math.abs(b) > BOUND_LN_P) return 1e15;
    const gamma = theta.slice(2);
    if (gamma.some((g) => !Number.isFinite(g) || Math.abs(g) > BOUND_GAMMA)) return 1e15;
    const p = Math.exp(b);
    let total = 0;
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i]!;
      const lnd = Math.log(s.hours) - a;
      let z = 0;
      for (let j = 0; j < k; j++) z += gamma[j]! * xs[i]![j]!;
      const hEnd = Math.exp(p * lnd + z);
      if (!Number.isFinite(hEnd)) return 1e15;
      // Cumulative hazard over the piece: ((end/η)^p − (start/η)^p)·e^z.
      const start = s.startHours ?? 0;
      const hStart = start > 0 ? Math.exp(p * (Math.log(start) - a) + z) : 0;
      total += hEnd - hStart;
      if (s.failed) total += -b + a - (p - 1) * lnd - z;
    }
    return Number.isFinite(total) ? total : 1e15;
  };

  const sortedHours = samples.map((s) => s.hours).sort((m, n) => m - n);
  const mid = sortedHours[Math.floor(sortedHours.length / 2)]!;
  const warm = options.warmStart;
  const theta0 = [
    Math.log(Math.max(warm?.baseScale ?? mid, 1e-9)),
    Math.log(Math.max(warm?.shape ?? 1, 1e-6)),
    ...new Array<number>(k).fill(0),
  ];

  const solved = newtonSolve(nll, theta0);
  if (!solved) return null;
  const { theta, iterations, converged } = solved;
  const gamma = theta.slice(2);

  const hess = fdHessian(nll, theta);
  const cov = hess ? invertMatrix(hess) : null;
  const coefSe = cov
    ? gamma.map((_, j) => {
        const v = cov[j + 2]?.[j + 2];
        return v != null && v >= 0 ? Math.sqrt(v) : null;
      })
    : gamma.map(() => null);

  return {
    shape: Math.exp(theta[1]!),
    baseScale: Math.exp(theta[0]!),
    coef: gamma,
    coefSe,
    covariateNames: [...covariateNames],
    covariateMeans: means,
    covariateStdDevs: stds,
    hazardRatios: gamma.map((g) => Math.exp(g)),
    logLik: -solved.nll,
    iterations,
    converged,
    failures,
    // Lives, not pieces: a split life is still one failed or censored unit.
    suspensions: new Set(samples.map((s, i) => s.group ?? `__${i}`)).size - failures,
    cov,
  };
}

function checkX(fit: WeibullPhFit, rawX: readonly number[]): void {
  if (rawX.length !== fit.coef.length) {
    throw new ReliabilityError(
      'DIMENSION_MISMATCH',
      `expected ${fit.coef.length} covariate values, got ${rawX.length}`,
    );
  }
}

/** Linear predictor γ'x* for raw covariate values (x* standardized with the fit's means/SDs). */
export function weibullPhLinearPredictor(fit: WeibullPhFit, rawX: readonly number[]): number {
  checkX(fit, rawX);
  let z = 0;
  for (let j = 0; j < rawX.length; j++) {
    z += (fit.coef[j]! * (rawX[j]! - fit.covariateMeans[j]!)) / fit.covariateStdDevs[j]!;
  }
  return z;
}

/** Weibull scale in hours for a unit with these raw covariates: η₀·exp(−z / p), z = the linear predictor. */
export function weibullPhScale(fit: WeibullPhFit, rawX: readonly number[]): number {
  return fit.baseScale * Math.exp(-weibullPhLinearPredictor(fit, rawX) / fit.shape);
}

/**
 * Median remaining life (hours) of a unit with these covariates that has
 * survived to `ageHours`, with a delta-method interval from the fit's full
 * covariance. Same closed form as {@link remainingUsefulLife}, with the
 * covariate-adjusted scale:
 *
 *   ln t = ln η₀ − z/p + (1/p)·ln w,   w = ln 2 + u,   u = (age/η_i)^p,   L = ln(age/η_i)
 *   ∂ln t/∂ln η₀ = 1 − u/w
 *   ∂ln t/∂ln p  = (z − ln w)/p + u·(L − z/p)/w
 *   ∂ln t/∂γ_j   = −(x*_j/p)·(1 − u/w)
 */
export function weibullPhRemainingLife(
  fit: WeibullPhFit,
  rawX: readonly number[],
  ageHours: number,
  options: RemainingLifeOptions = {},
): RemainingLife {
  assertNonNegative(ageHours, 'ageHours');
  const zScore = zForConfidence(options.confidence ?? 0.9);
  const z = weibullPhLinearPredictor(fit, rawX);
  const p = fit.shape;
  const eta = fit.baseScale * Math.exp(-z / p);
  const L = Math.log(Math.max(ageHours, Number.MIN_VALUE)) - Math.log(eta);
  const u = Math.exp(p * L);
  const w = Math.LN2 + u;
  const t = eta * Math.pow(w, 1 / p);
  const medianHours = Math.max(0, t - ageHours);
  if (!fit.cov) return { medianHours, interval: null };

  const xs = rawX.map((v, j) => (v - fit.covariateMeans[j]!) / fit.covariateStdDevs[j]!);
  const g = new Array<number>(2 + fit.coef.length).fill(0);
  g[0] = 1 - u / w;
  g[1] = (z - Math.log(w)) / p + (u > 0 ? (u * (L - z / p)) / w : 0);
  for (let j = 0; j < xs.length; j++) g[j + 2] = -(xs[j]! / p) * (1 - u / w);
  let variance = 0;
  for (let i = 0; i < g.length; i++) {
    for (let j = 0; j < g.length; j++) variance += g[i]! * g[j]! * (fit.cov[i]?.[j] ?? 0);
  }
  if (!Number.isFinite(variance) || variance < 0) return { medianHours, interval: null };
  const se = Math.sqrt(variance);
  const lnT = Math.log(Math.max(t, Number.MIN_VALUE));
  return {
    medianHours,
    interval: {
      lowerHours: Math.max(0, Math.exp(lnT - zScore * se) - ageHours),
      upperHours: Math.max(0, Math.exp(lnT + zScore * se) - ageHours),
    },
  };
}

// ─── Solver (damped Newton with finite-difference derivatives) ───────────────

interface NewtonResult {
  theta: number[];
  nll: number;
  iterations: number;
  converged: boolean;
}

function newtonSolve(f: (theta: number[]) => number, theta0: number[]): NewtonResult | null {
  let theta = theta0.slice();
  let fTheta = f(theta);
  if (!Number.isFinite(fTheta)) return null;
  let converged = false;
  let iterations = 0;
  for (let it = 0; it < MAX_NEWTON_ITERS; it++) {
    iterations = it + 1;
    const grad = fdGradient(f, theta);
    const hess = fdHessian(f, theta);
    if (!grad || !hess) break;
    // Add a ridge to the Hessian until the step is a descent direction.
    let step: number[] | null = null;
    const trace = hess.reduce((s, r, i) => s + (r[i] ?? 0), 0);
    const baseRidge = Math.max(trace * 1e-8, 1e-10);
    for (let tries = 0; tries < 12; tries++) {
      const add = tries === 0 ? 0 : baseRidge * 10 ** (tries - 1);
      const ridged = hess.map((row, i) => row.map((v, j) => (i === j ? v + add : v)));
      const d = solveLinear(
        ridged,
        grad.map((v) => -v),
      );
      if (d && d.every((v) => Number.isFinite(v))) {
        const slope = grad.reduce((s, v, i) => s + v * d[i]!, 0);
        if (slope < 0) {
          step = d;
          break;
        }
      }
    }
    if (!step) break;
    const dir = step;

    // Backtracking line search (Armijo).
    const slope = grad.reduce((s, v, i) => s + v * dir[i]!, 0);
    let t = 1;
    let next = f(theta.map((v, i) => v + t * dir[i]!));
    while (t > 1 / 2 ** 30 && !(Number.isFinite(next) && next <= fTheta + 1e-4 * t * slope)) {
      t /= 2;
      next = f(theta.map((v, i) => v + t * dir[i]!));
    }
    const maxStep = Math.max(...dir.map((v) => Math.abs(t * v)));
    theta = theta.map((v, i) => v + t * dir[i]!);
    const fNext = next;
    if (maxStep < STEP_TOL || Math.abs(fTheta - fNext) < NLL_TOL) {
      fTheta = fNext;
      converged = true;
      break;
    }
    fTheta = fNext;
  }
  if (!Number.isFinite(fTheta)) return null;
  return { theta, nll: fTheta, iterations, converged };
}

function fdGradient(f: (theta: number[]) => number, theta: number[]): number[] | null {
  const g: number[] = [];
  for (let j = 0; j < theta.length; j++) {
    const h = 1e-6 * Math.max(1, Math.abs(theta[j]!));
    const tp = theta.slice();
    const tm = theta.slice();
    tp[j] = tp[j]! + h;
    tm[j] = tm[j]! - h;
    const fp = f(tp);
    const fm = f(tm);
    if (!Number.isFinite(fp) || !Number.isFinite(fm)) return null;
    g.push((fp - fm) / (2 * h));
  }
  return g;
}

function fdHessian(f: (theta: number[]) => number, theta: number[]): number[][] | null {
  const n = theta.length;
  const hess: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n).fill(0);
    const hi = 1e-4 * Math.max(1, Math.abs(theta[i]!));
    for (let j = i; j < n; j++) {
      const hj = 1e-4 * Math.max(1, Math.abs(theta[j]!));
      const tpp = theta.slice();
      const tpm = theta.slice();
      const tmp = theta.slice();
      const tmm = theta.slice();
      tpp[i] = tpp[i]! + hi;
      tpp[j] = tpp[j]! + hj;
      tpm[i] = tpm[i]! + hi;
      tpm[j] = tpm[j]! - hj;
      tmp[i] = tmp[i]! - hi;
      tmp[j] = tmp[j]! + hj;
      tmm[i] = tmm[i]! - hi;
      tmm[j] = tmm[j]! - hj;
      const v = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) / (4 * hi * hj);
      if (!Number.isFinite(v)) return null;
      row[j] = v;
    }
    hess.push(row);
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) hess[i]![j] = hess[j]![i]!;
  }
  return hess;
}

/** Gaussian elimination with partial pivoting; null when singular. */
function solveLinear(a: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r]![col]!) > Math.abs(m[piv]![col]!)) piv = r;
    }
    if (Math.abs(m[piv]![col]!) < 1e-14) return null;
    [m[col], m[piv]] = [m[piv]!, m[col]!];
    for (let r = col + 1; r < n; r++) {
      const factor = m[r]![col]! / m[col]![col]!;
      for (let c = col; c <= n; c++) m[r]![c] = m[r]![c]! - factor * m[col]![c]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r]![n]!;
    for (let c = r + 1; c < n; c++) s -= m[r]![c]! * x[c]!;
    x[r] = s / m[r]![r]!;
  }
  return x;
}

/** Gauss-Jordan inverse; null when singular. */
function invertMatrix(a: number[][]): number[][] | null {
  const n = a.length;
  const m = a.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r]![col]!) > Math.abs(m[piv]![col]!)) piv = r;
    }
    if (Math.abs(m[piv]![col]!) < 1e-14) return null;
    [m[col], m[piv]] = [m[piv]!, m[col]!];
    const pivot = m[col]![col]!;
    for (let c = 0; c < 2 * n; c++) m[col]![c] = m[col]![c]! / pivot;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = m[r]![col]!;
      if (factor === 0) continue;
      for (let c = 0; c < 2 * n; c++) m[r]![c] = m[r]![c]! - factor * m[col]![c]!;
    }
  }
  return m.map((row) => row.slice(n));
}
