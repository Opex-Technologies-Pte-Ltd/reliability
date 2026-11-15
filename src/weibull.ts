/**
 * Two-parameter Weibull distribution in hours.
 *
 *   R(t) = exp(−(t/η)^β)        reliability (survival)
 *   F(t) = 1 − R(t)             unreliability (CDF)
 *   h(t) = (β/η)·(t/η)^(β−1)    hazard rate (failures per hour)
 *
 * β (shape) < 1 means a falling failure rate (early-life failures), β = 1 a
 * constant rate (random failures, exponential), β > 1 a rising rate (wear-out).
 * η (scale, "characteristic life") is the age in hours by which 63.2% of units fail.
 */
import { ReliabilityError, assertFinite, assertNonNegative } from './errors.js';
import { gamma, normalQuantile } from './special.js';

export interface WeibullParams {
  /** β, dimensionless, > 0. */
  shape: number;
  /** η in hours, > 0. */
  scale: number;
}

/**
 * Covariance of the parameter estimates on the log scale, ordered
 * `[ln scale, ln shape]`: `[[Var(ln η), Cov], [Cov, Var(ln β)]]`.
 */
export type LogParamCovariance = readonly [readonly [number, number], readonly [number, number]];

/** Weibull parameters plus, optionally, the uncertainty of their estimate. */
export interface WeibullModel extends WeibullParams {
  covariance?: LogParamCovariance | null;
}

/** @internal */
export function assertWeibull(params: WeibullParams): void {
  if (!params || !(params.shape > 0) || !Number.isFinite(params.shape)) {
    throw new ReliabilityError('INVALID_PARAMETER', `Weibull shape must be a finite number > 0, got ${params?.shape}`);
  }
  if (!(params.scale > 0) || !Number.isFinite(params.scale)) {
    throw new ReliabilityError('INVALID_PARAMETER', `Weibull scale must be a finite number > 0, got ${params.scale}`);
  }
}

/** Probability density f(t) at `hours` (per hour). */
export function weibullPdf(params: WeibullParams, hours: number): number {
  assertWeibull(params);
  assertNonNegative(hours, 'hours');
  const { shape: b, scale: e } = params;
  return (b / e) * (hours / e) ** (b - 1) * Math.exp(-((hours / e) ** b));
}

/** Probability of failure by `hours`, F(t). */
export function weibullCdf(params: WeibullParams, hours: number): number {
  assertWeibull(params);
  assertNonNegative(hours, 'hours');
  return -Math.expm1(-((hours / params.scale) ** params.shape));
}

/** Probability of surviving past `hours`, R(t) = 1 − F(t). */
export function weibullReliability(params: WeibullParams, hours: number): number {
  assertWeibull(params);
  assertNonNegative(hours, 'hours');
  return Math.exp(-((hours / params.scale) ** params.shape));
}

/** Instantaneous failure rate h(t) at age `hours`, in failures per hour. */
export function weibullHazard(params: WeibullParams, hours: number): number {
  assertWeibull(params);
  assertNonNegative(hours, 'hours');
  const { shape: b, scale: e } = params;
  return (b / e) * (hours / e) ** (b - 1);
}

/**
 * Mean life in hours, η·Γ(1 + 1/β). For a repairable item that is restored
 * "as good as new" at each repair this is the MTBF; for a non-repairable item, the MTTF.
 */
export function weibullMean(params: WeibullParams): number {
  assertWeibull(params);
  return params.scale * gamma(1 + 1 / params.shape);
}

/** Age in hours by which a fraction `probability` (0 < p < 1) of units has failed. */
export function weibullQuantile(params: WeibullParams, probability: number): number {
  assertWeibull(params);
  assertFinite(probability, 'probability');
  if (!(probability > 0 && probability < 1)) {
    throw new ReliabilityError('INVALID_INPUT', `probability must be in (0, 1), got ${probability}`);
  }
  return params.scale * (-Math.log1p(-probability)) ** (1 / params.shape);
}

/**
 * B-life in hours: the age by which `percent` % of units has failed.
 * `bLife(p, 10)` is the B10 life; `bLife(p, 50)` the median life.
 */
export function bLife(params: WeibullParams, percent: number): number {
  assertFinite(percent, 'percent');
  if (!(percent > 0 && percent < 100)) {
    throw new ReliabilityError('INVALID_INPUT', `percent must be in (0, 100), got ${percent}`);
  }
  return weibullQuantile(params, percent / 100);
}

/**
 * Probability that a unit which has survived to `ageHours` also survives the
 * next `missionHours`: R(age + mission) / R(age).
 */
export function conditionalReliability(
  params: WeibullParams,
  ageHours: number,
  missionHours: number,
): number {
  assertWeibull(params);
  assertNonNegative(ageHours, 'ageHours');
  assertNonNegative(missionHours, 'missionHours');
  const { shape: b, scale: e } = params;
  return Math.exp((ageHours / e) ** b - ((ageHours + missionHours) / e) ** b);
}

export interface RemainingLifeOptions {
  /** Two-sided confidence level of the interval, in (0, 1). Default 0.9. */
  confidence?: number;
}

export interface RemainingLife {
  /** Median remaining life in hours for a unit that has survived to the given age. */
  medianHours: number;
  /**
   * Two-sided confidence interval on the median remaining life (hours, floored at 0).
   * `null` when the model carries no covariance.
   */
  interval: { lowerHours: number; upperHours: number } | null;
}

/**
 * Remaining useful life of a unit that has survived to `ageHours`: the median
 * of (T − age) given T > age,
 *
 *   m(age) = η·(ln 2 + (age/η)^β)^(1/β) − age.
 *
 * At age 0 this is the B50 life. The interval uses the delta method on
 * ln t, t = m(age) + age, with the model's `[ln η, ln β]` covariance
 * (from {@link fitWeibull}). It reflects parameter uncertainty only, not the
 * spread of individual lives.
 */
export function remainingUsefulLife(
  model: WeibullModel,
  ageHours: number,
  options: RemainingLifeOptions = {},
): RemainingLife {
  assertWeibull(model);
  assertNonNegative(ageHours, 'ageHours');
  const z = zForConfidence(options.confidence ?? 0.9);
  const { shape: b, scale: e } = model;

  const ratio = ageHours / e;
  const u = ratio > 0 ? ratio ** b : 0;
  const w = Math.LN2 + u;
  const t = e * w ** (1 / b);
  const medianHours = Math.max(0, t - ageHours);

  const cov = model.covariance;
  if (!cov) return { medianHours, interval: null };

  // ∂ln t/∂ln η = 1 − u/w;  ∂ln t/∂ln β = −ln(w)/β + u·ln(age/η)/w
  const gEta = 1 - u / w;
  const gBeta = -Math.log(w) / b + (u > 0 ? (u * Math.log(ratio)) / w : 0);
  const variance =
    gEta * gEta * cov[0][0] + 2 * gEta * gBeta * cov[0][1] + gBeta * gBeta * cov[1][1];
  if (!Number.isFinite(variance) || variance < 0) return { medianHours, interval: null };
  const se = Math.sqrt(variance);
  const lnT = Math.log(Math.max(t, Number.MIN_VALUE));
  return {
    medianHours,
    interval: {
      lowerHours: Math.max(0, Math.exp(lnT - z * se) - ageHours),
      upperHours: Math.max(0, Math.exp(lnT + z * se) - ageHours),
    },
  };
}

/** @internal Two-sided z for a confidence level in (0, 1). */
export function zForConfidence(confidence: number): number {
  assertFinite(confidence, 'confidence');
  if (!(confidence > 0 && confidence < 1)) {
    throw new ReliabilityError('INVALID_INPUT', `confidence must be in (0, 1), got ${confidence}`);
  }
  return normalQuantile(0.5 + confidence / 2);
}
