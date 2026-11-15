/**
 * PCA-based multivariate condition monitoring: train on readings from normal
 * operation, then score new readings with two statistics and their control limits.
 *
 * - SPE (squared prediction error, Q statistic): squared distance from the
 *   retained principal subspace, in standardized units. Limit: an empirical
 *   quantile of the training SPE values.
 * - Hotelling T²: distance inside the retained subspace, weighted by the
 *   inverse eigenvalues. Limit: the F-distribution approximation
 *   T² = k(n−1)(n+1) / (n(n−k)) · F(k, n−k; α).
 *
 * Deterministic (Jacobi eigen-decomposition, no RNG). Features are standardized;
 * a constant feature gets a tiny standard-deviation floor instead of being dropped,
 * so the model's dimensions always match the input columns.
 */
import { ReliabilityError } from './errors.js';
import { empiricalQuantile, fQuantile } from './special.js';

export interface PcaOptions {
  /** Fewer training rows than this returns null. Default 50. */
  minSamples?: number;
  /** Keep the smallest number of components that explains this share of variance. Default 0.9. */
  varianceToKeep?: number;
  /** SPE limit = this quantile of the training SPE values. Default 0.99. */
  speQuantile?: number;
  /** Confidence level of the T² limit. Default 0.99. */
  t2Confidence?: number;
}

export interface PcaModel {
  /** Number of features. */
  dim: number;
  means: number[];
  stdDevs: number[];
  /** Retained unit eigenvectors, one row per component. */
  components: number[][];
  /** All eigenvalues of the correlation matrix, descending. */
  eigenvalues: number[];
  retained: number;
  sampleCount: number;
  /** Mean and standard deviation of the training SPE values. */
  errorMean: number;
  errorStd: number;
  /** SPE control limit. */
  threshold: number;
  /** Hotelling T² control limit, null when the sample is too small (n ≤ k + 1). */
  t2Threshold: number | null;
}

export interface PcaContribution {
  /** Feature (column) index. */
  index: number;
  contribution: number;
}

export interface PcaScore {
  /** SPE: squared reconstruction error in standardized space. */
  error: number;
  /** Reconstruction of the input from the retained components, in original units. */
  reconstruction: number[];
  /** Per-feature share of SPE, largest first. */
  contributors: PcaContribution[];
  /** Hotelling T² over the retained components. */
  t2: number;
  /** Per-feature T² contributions z_j²·Σ_i v_ji²/λ_i, largest first (cross terms left out). */
  t2Contributors: PcaContribution[];
}

const STD_FLOOR = 1e-9;
const EIGEN_FLOOR = 1e-12;
const JACOBI_SWEEPS = 50;

/**
 * Upper control limit of Hotelling T² for k retained components and n training
 * samples (F approximation). Null when n ≤ k + 1 or k < 1.
 */
export function hotellingT2Limit(k: number, n: number, confidence = 0.99): number | null {
  if (!Number.isFinite(k) || !Number.isFinite(n) || k < 1 || n <= k + 1) return null;
  const f = fQuantile(k, n - k, confidence);
  if (!Number.isFinite(f) || f <= 0) return null;
  return ((k * (n - 1) * (n + 1)) / (n * (n - k))) * f;
}

/** True when SPE is over its limit, or T² is over its limit (when the model has one). */
export function isPcaAnomalous(model: PcaModel, score: PcaScore): boolean {
  if (score.error > model.threshold) return true;
  return model.t2Threshold != null && score.t2 > model.t2Threshold;
}

/**
 * Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
 * Returns eigenvalues (unsorted) and eigenvectors as columns: `vectors[row][i]`
 * pairs with `values[i]`.
 */
export function symmetricEigen(matrix: readonly (readonly number[])[]): {
  values: number[];
  vectors: number[][];
} {
  const d = matrix.length;
  for (const row of matrix) {
    if (row.length !== d) throw new ReliabilityError('DIMENSION_MISMATCH', 'matrix must be square');
  }
  const a = matrix.map((row) => [...row]);
  const v: number[][] = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
    let off = 0;
    for (let i = 0; i < d; i++) {
      for (let j = i + 1; j < d; j++) off += a[i]![j]! ** 2;
    }
    if (off < 1e-12) break;
    for (let p = 0; p < d - 1; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-12) continue;
        const app = a[p]![p]!;
        const aqq = a[q]![q]!;
        const theta = (aqq - app) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        // A ← JᵀAJ, J = identity with J[p][p] = J[q][q] = c, J[p][q] = s, J[q][p] = −s.
        for (let k = 0; k < d; k++) {
          if (k === p || k === q) continue;
          const akp = a[k]![p]!;
          const akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[p]![k] = a[k]![p]!;
          a[k]![q] = s * akp + c * akq;
          a[q]![k] = a[k]![q]!;
        }
        a[p]![p] = c * c * app - 2 * c * s * apq + s * s * aqq;
        a[q]![q] = s * s * app + 2 * c * s * apq + c * c * aqq;
        a[p]![q] = 0;
        a[q]![p] = 0;
        for (let k = 0; k < d; k++) {
          const vkp = v[k]![p]!;
          const vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: a.map((row, i) => row[i]!), vectors: v };
}

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/**
 * Train a PCA model on rows of readings from normal operation (one row per
 * observation, one column per feature). Returns null with fewer than
 * `minSamples` rows or fewer than 2 features. At most d − 1 components are kept,
 * so SPE always has a residual space.
 */
export function trainPca(matrix: readonly (readonly number[])[], options: PcaOptions = {}): PcaModel | null {
  const minSamples = options.minSamples ?? 50;
  const varianceToKeep = options.varianceToKeep ?? 0.9;
  const n = matrix.length;
  if (n < Math.max(minSamples, 3)) return null;
  const d = matrix[0]!.length;
  if (d < 2) return null;
  matrix.forEach((row, i) => {
    if (row.length !== d) {
      throw new ReliabilityError('DIMENSION_MISMATCH', `row ${i} has ${row.length} values, expected ${d}`);
    }
    if (row.some((v) => !Number.isFinite(v))) {
      throw new ReliabilityError('INVALID_INPUT', `row ${i} contains a non-finite value`);
    }
  });

  const means = Array.from({ length: d }, (_, j) => matrix.reduce((s, r) => s + r[j]!, 0) / n);
  const stdDevs = Array.from({ length: d }, (_, j) => {
    const variance = matrix.reduce((s, r) => s + (r[j]! - means[j]!) ** 2, 0) / (n - 1);
    return Math.max(Math.sqrt(variance), STD_FLOOR);
  });
  const z = matrix.map((row) => row.map((v, j) => (v - means[j]!) / stdDevs[j]!));

  // Covariance of standardized data = correlation matrix.
  const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let s = 0;
      for (const row of z) s += row[i]! * row[j]!;
      cov[i]![j] = s / (n - 1);
      cov[j]![i] = cov[i]![j]!;
    }
  }

  const { values, vectors } = symmetricEigen(cov);
  const order = values.map((value, idx) => ({ value, idx })).sort((a, b) => b.value - a.value);
  const eigenvalues = order.map((e) => e.value);
  const totalVariance = eigenvalues.reduce((s, v) => s + Math.max(v, 0), 0);
  let retained = 1;
  let cumulative = 0;
  const maxK = Math.max(1, d - 1);
  for (let k = 1; k <= maxK && retained < maxK; k++) {
    cumulative += Math.max(eigenvalues[k - 1]!, 0);
    retained = k;
    if (totalVariance > 0 && cumulative / totalVariance >= varianceToKeep) break;
  }
  const components = order.slice(0, retained).map((e) => vectors.map((row) => row[e.idx]!));

  const errors = z.map((row) => {
    const hat = reconstructZ(components, row);
    let err = 0;
    for (let i = 0; i < row.length; i++) err += (row[i]! - hat[i]!) ** 2;
    return err;
  });
  const errorMean = errors.reduce((s, v) => s + v, 0) / errors.length;
  const errorVar = errors.reduce((s, v) => s + (v - errorMean) ** 2, 0) / Math.max(1, errors.length - 1);

  return {
    dim: d,
    means,
    stdDevs,
    components,
    eigenvalues,
    retained,
    sampleCount: n,
    errorMean,
    errorStd: Math.sqrt(errorVar),
    threshold: empiricalQuantile(errors, options.speQuantile ?? 0.99),
    t2Threshold: hotellingT2Limit(retained, n, options.t2Confidence ?? 0.99),
  };
}

function reconstructZ(components: readonly (readonly number[])[], zRow: readonly number[]): number[] {
  const hat = new Array<number>(zRow.length).fill(0);
  for (const component of components) {
    const w = dot(zRow, component);
    for (let i = 0; i < hat.length; i++) hat[i] = hat[i]! + w * component[i]!;
  }
  return hat;
}

/** Score one reading (original units) against a model. Null when its length differs from the model's. */
export function scorePca(model: PcaModel, vector: readonly number[]): PcaScore | null {
  if (vector.length !== model.dim) return null;
  const zRow = vector.map((v, j) => (v - model.means[j]!) / model.stdDevs[j]!);
  const hat = reconstructZ(model.components, zRow);
  const reconstruction = hat.map((z, j) => model.means[j]! + z * model.stdDevs[j]!);
  let error = 0;
  const contributors = zRow.map((z, i) => {
    const residual = (z - hat[i]!) ** 2;
    error += residual;
    return { index: i, contribution: residual };
  });
  contributors.sort((a, b) => b.contribution - a.contribution);

  const inverseEigen = model.components.map((_, i) => 1 / Math.max(model.eigenvalues[i] ?? 0, EIGEN_FLOOR));
  let t2 = 0;
  for (let i = 0; i < model.components.length; i++) {
    const w = dot(zRow, model.components[i]!);
    t2 += w * w * inverseEigen[i]!;
  }
  const t2Contributors = zRow.map((z, j) => {
    let weight = 0;
    for (let i = 0; i < model.components.length; i++) {
      weight += model.components[i]![j]! ** 2 * inverseEigen[i]!;
    }
    return { index: j, contribution: z * z * weight };
  });
  t2Contributors.sort((a, b) => b.contribution - a.contribution);

  return { error, reconstruction, contributors, t2, t2Contributors };
}
