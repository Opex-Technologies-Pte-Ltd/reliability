/**
 * Special functions and distribution quantiles used by the rest of the package.
 * Plain double-precision implementations, no dependencies.
 */
import { ReliabilityError, assertFinite } from './errors.js';

const LANCZOS_G = 7;
const LANCZOS_COEF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];
const HALF_LN_2PI = 0.5 * Math.log(2 * Math.PI);

/**
 * Gamma function Γ(z) (Lanczos approximation, g = 7, about 15 significant digits).
 * Uses the reflection formula for z < 0.5. Poles (0, −1, −2, …) return ±Infinity or NaN.
 */
export function gamma(z: number): number {
  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  }
  const zm = z - 1;
  let x = LANCZOS_COEF[0]!;
  for (let i = 1; i < LANCZOS_G + 2; i++) x += LANCZOS_COEF[i]! / (zm + i);
  const t = zm + LANCZOS_G + 0.5;
  return Math.sqrt(2 * Math.PI) * t ** (zm + 0.5) * Math.exp(-t) * x;
}

/** Natural log of Γ(z) for z > 0, computed in log space (no overflow for large z). */
export function lnGamma(z: number): number {
  if (!(z > 0)) {
    throw new ReliabilityError('INVALID_INPUT', `lnGamma is defined here for z > 0, got ${z}`);
  }
  if (z < 0.5) {
    // Reflection: Γ(z)Γ(1−z) = π / sin(πz), sin(πz) > 0 on (0, 0.5).
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  }
  const zm = z - 1;
  let x = LANCZOS_COEF[0]!;
  for (let i = 1; i < LANCZOS_G + 2; i++) x += LANCZOS_COEF[i]! / (zm + i);
  const t = zm + LANCZOS_G + 0.5;
  return HALF_LN_2PI + (zm + 0.5) * Math.log(t) - t + Math.log(x);
}

/** Continued fraction for the regularized incomplete beta (Numerical Recipes §6.4). */
function betaContinuedFraction(a: number, b: number, x: number): number {
  const MAXIT = 200;
  const EPS = 3e-12;
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** Regularized incomplete beta function I_x(a, b) for a, b > 0. x is clamped to [0, 1]. */
export function incompleteBeta(a: number, b: number, x: number): number {
  if (!(a > 0) || !(b > 0)) {
    throw new ReliabilityError('INVALID_PARAMETER', `incompleteBeta needs a, b > 0 (got ${a}, ${b})`);
  }
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(
    lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x),
  );
  if (x < (a + 1) / (a + b + 2)) return (bt * betaContinuedFraction(a, b, x)) / a;
  return 1 - (bt * betaContinuedFraction(b, a, 1 - x)) / b;
}

/** CDF of the F distribution with (d1, d2) degrees of freedom. */
export function fCdf(x: number, d1: number, d2: number): number {
  if (x <= 0) return 0;
  return incompleteBeta(d1 / 2, d2 / 2, (d1 * x) / (d1 * x + d2));
}

/**
 * Quantile of the F distribution: the x with P(F ≤ x) = p, by bisection on {@link fCdf}.
 * Returns NaN for d1 ≤ 0, d2 ≤ 0 or p outside (0, 1).
 */
export function fQuantile(d1: number, d2: number, p: number): number {
  if (!(d1 > 0) || !(d2 > 0) || !(p > 0) || !(p < 1)) return NaN;
  const cdf = (x: number) => fCdf(x, d1, d2);
  let hi = 1;
  while (cdf(hi) < p && hi < 1e12) hi *= 2;
  let lo = 0;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/**
 * Standard normal quantile Φ⁻¹(p) (Wichura, AS 241 "PPND16", relative error about 1e-16).
 * Throws for p outside (0, 1).
 */
export function normalQuantile(p: number): number {
  assertFinite(p, 'p');
  if (!(p > 0 && p < 1)) {
    throw new ReliabilityError('INVALID_INPUT', `normalQuantile needs 0 < p < 1, got ${p}`);
  }
  const q = p - 0.5;
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q;
    return (
      (q *
        (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r +
          45921.953931549871457) *
          r +
          13731.693765509461125) *
          r +
          1971.5909503065514427) *
          r +
          133.14166789178437745) *
          r +
          3.387132872796366608)) /
      (((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r +
        21213.794301586595867) *
        r +
        5394.1960214247511077) *
        r +
        687.1870074920579083) *
        r +
        42.313330701600911252) *
        r +
        1)
    );
  }
  let r = q < 0 ? p : 1 - p;
  r = Math.sqrt(-Math.log(r));
  let val: number;
  if (r <= 5) {
    r -= 1.6;
    val =
      (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) *
        r +
        1.27045825245236838258) *
        r +
        3.64784832476320460504) *
        r +
        5.7694972214606914055) *
        r +
        4.6303378461565452959) *
        r +
        1.42343711074968357734) /
      (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) *
        r +
        0.14810397642748007459) *
        r +
        0.68976733498510000455) *
        r +
        1.6763848301838038494) *
        r +
        2.05319162663775882187) *
        r +
        1);
  } else {
    r -= 5;
    val =
      (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) *
        r +
        0.026532189526576123093) *
        r +
        0.29656057182850489123) *
        r +
        1.7848265399172913358) *
        r +
        5.4637849111641143699) *
        r +
        6.6579046435011037772) /
      (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) *
        r +
        7.868691311456132591e-4) *
        r +
        0.0148753612908506148525) *
        r +
        0.13692988092273580531) *
        r +
        0.59983220655588793769) *
        r +
        1);
  }
  return q < 0 ? -val : val;
}

/**
 * Empirical quantile with linear interpolation between order statistics
 * (the "type 7" definition, same as numpy's default). q is clamped to [0, 1];
 * an empty sample returns 0.
 */
export function empiricalQuantile(values: readonly number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * Math.min(Math.max(q, 0), 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}
