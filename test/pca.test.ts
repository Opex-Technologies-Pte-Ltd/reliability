// Expected values: numpy.linalg.eigh, numpy.quantile and scipy.stats.f (scripts/oracle.py, section 5).
import { describe, expect, it } from 'bun:test';
import { ReliabilityError, hotellingT2Limit, isPcaAnomalous, scorePca, symmetricEigen, trainPca } from '../src';

function near(actual: number, expected: number, tol: number): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(Math.abs(expected), 1e-12));
}

/** Same deterministic data as the oracle: 80 rows, 3 correlated features. */
const rows = Array.from({ length: 80 }, (_, i) => {
  const a = Math.sin(i * 0.37) * 3;
  const b = Math.cos(i * 0.11) * 2;
  return [a + 0.1 * Math.sin(i * 1.7), 2 * a + b + 0.05 * Math.cos(i * 2.3), b - a + 0.2 * Math.sin(i * 0.9 + 1)];
});

/** Points tightly along y = x. */
const line = Array.from({ length: 60 }, (_, i) => {
  const base = i / 5 - 6;
  return [base, base + (((i * 7) % 5) - 2) / 50];
});

describe('symmetricEigen', () => {
  it('diagonalizes [[2,1],[1,2]] → {3, 1}', () => {
    const { values, vectors } = symmetricEigen([
      [2, 1],
      [1, 2],
    ]);
    const i3 = values[0]! > values[1]! ? 0 : 1;
    near(values[i3]!, 3, 1e-12);
    near(values[1 - i3]!, 1, 1e-12);
    near(Math.abs(vectors[0]![i3]!), Math.SQRT1_2, 1e-12);
  });

  it('A·v = λ·v for a 4×4 matrix', () => {
    const m = [
      [4, 1, 0.5, 0],
      [1, 3, 0.2, 0.1],
      [0.5, 0.2, 2, 0.3],
      [0, 0.1, 0.3, 1],
    ];
    const { values, vectors } = symmetricEigen(m);
    for (let k = 0; k < 4; k++) {
      for (let i = 0; i < 4; i++) {
        const av = m[i]!.reduce((s, mij, j) => s + mij * vectors[j]![k]!, 0);
        expect(Math.abs(av - values[k]! * vectors[i]![k]!)).toBeLessThan(1e-10);
      }
    }
  });
});

describe('trainPca / scorePca', () => {
  it('matches numpy: eigenvalues, retained components, SPE limit, SPE and T² of a reading', () => {
    const model = trainPca(rows)!;
    near(model.eigenvalues[0]!, 2.60468759, 1e-8);
    near(model.eigenvalues[1]!, 0.394295368, 1e-8);
    near(model.eigenvalues[2]!, 0.00101704126, 1e-7);
    expect(model.retained).toBe(2);
    near(model.threshold, 0.003816620747025311, 1e-9);
    near(model.t2Threshold!, 10.025279913278691, 1e-9);
    const s = scorePca(model, [1, -2, 4])!;
    near(s.error, 1.210735218125429, 1e-9);
    near(s.t2, 2.857000998776332, 1e-9);
  });

  it('first component follows the y = x direction; one component kept', () => {
    const model = trainPca(line)!;
    near(Math.abs(model.components[0]![0]!), Math.SQRT1_2, 1e-3);
    expect(model.retained).toBe(1);
  });

  it('flags an off-line reading by SPE and a far on-line reading by T²', () => {
    const model = trainPca(line)!;
    expect(isPcaAnomalous(model, scorePca(model, [1, 1.02])!)).toBe(false);
    const off = scorePca(model, [5, -5])!;
    expect(off.error).toBeGreaterThan(model.threshold * 10);
    const far = scorePca(model, [30, 30.02])!;
    expect(far.error).toBeLessThan(model.threshold);
    expect(far.t2).toBeGreaterThan(model.t2Threshold!);
    expect(isPcaAnomalous(model, far)).toBe(true);
  });

  it('ranks the broken feature first in both contribution lists', () => {
    const model = trainPca(line)!;
    const s = scorePca(model, [0, 8])!;
    expect(s.contributors[0]!.index).toBe(1);
    expect(s.t2Contributors[0]!.index).toBe(1);
  });

  it('keeps a constant feature instead of dropping it', () => {
    const model = trainPca(line.map(([a, b]) => [a!, b!, 42]))!;
    expect(model.dim).toBe(3);
    expect(scorePca(model, [0, 0, 42])!.error).toBeLessThan(model.threshold);
  });

  it('null for too few rows or one feature; null score on length mismatch; throws on bad rows', () => {
    expect(trainPca(line.slice(0, 49))).toBeNull();
    expect(trainPca(line.slice(0, 20), { minSamples: 10 })).not.toBeNull();
    expect(trainPca(line.map(([a]) => [a!]))).toBeNull();
    expect(scorePca(trainPca(line)!, [1, 2, 3])).toBeNull();
    const ragged = line.map((r) => [...r]);
    ragged[5] = [1];
    expect(() => trainPca(ragged)).toThrow(ReliabilityError);
    const nan = line.map((r) => [...r]);
    nan[5] = [1, Number.NaN];
    expect(() => trainPca(nan)).toThrow(ReliabilityError);
  });
});

describe('hotellingT2Limit', () => {
  it('matches the F approximation computed with scipy', () => {
    near(hotellingT2Limit(2, 60)!, 10.32326948038325, 1e-9);
  });

  it('is null when n ≤ k + 1 or k < 1', () => {
    expect(hotellingT2Limit(2, 3)).toBeNull();
    expect(hotellingT2Limit(0, 60)).toBeNull();
  });
});
