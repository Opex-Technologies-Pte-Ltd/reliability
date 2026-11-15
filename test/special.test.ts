// Expected values: scipy.special / scipy.stats (scripts/oracle.py, section 4).
import { describe, expect, it } from 'bun:test';
import {
  ReliabilityError,
  empiricalQuantile,
  fQuantile,
  gamma,
  incompleteBeta,
  lnGamma,
  normalQuantile,
} from '../src';

function near(actual: number, expected: number, tol: number): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(Math.abs(expected), 1));
}

describe('gamma and lnGamma', () => {
  const ref = [
    [0.3, 2.991568987687591, 1.0957979948180756],
    [1, 1, 0],
    [2.5, 1.329340388179137, 0.2846828704729192],
    [4.5, 11.63172839656745, 2.4537365708424423],
    [10, 362880, 12.801827480081469],
  ] as const;
  it('match scipy to about 1e-13', () => {
    for (const [z, g, lg] of ref) {
      near(gamma(z), g, 1e-13);
      near(lnGamma(z), lg, 1e-13);
    }
  });

  it('lnGamma does not overflow where gamma does', () => {
    expect(gamma(171.5)).toBe(Number.POSITIVE_INFINITY);
    near(lnGamma(171.5), 709.1431630309283, 1e-14);
    expect(() => lnGamma(0)).toThrow(ReliabilityError);
  });
});

describe('normalQuantile', () => {
  it('matches scipy.stats.norm.ppf', () => {
    const ref = [
      [1e-10, -6.361340902404056],
      [0.001, -3.090232306167813],
      [0.05, -1.6448536269514729],
      [0.5, 0],
      [0.95, 1.6448536269514722],
      [0.975, 1.959963984540054],
      [0.999, 3.090232306167813],
    ] as const;
    for (const [p, z] of ref) near(normalQuantile(p), z, 1e-14);
  });

  it('throws outside (0, 1)', () => {
    expect(() => normalQuantile(0)).toThrow(ReliabilityError);
    expect(() => normalQuantile(1)).toThrow(ReliabilityError);
  });
});

describe('incompleteBeta and fQuantile', () => {
  it('match scipy', () => {
    near(incompleteBeta(2, 3, 0.4), 0.5248, 1e-12);
    near(incompleteBeta(0.5, 7, 0.9), 0.9999999780702157, 1e-12);
    near(fQuantile(1, 30, 0.95), 4.170876785766691, 1e-9);
    near(fQuantile(3, 60, 0.99), 4.125891930795664, 1e-9);
  });

  it('fQuantile returns NaN for invalid arguments', () => {
    expect(fQuantile(2, 10, 0)).toBeNaN();
    expect(fQuantile(0, 10, 0.5)).toBeNaN();
  });
});

describe('empiricalQuantile', () => {
  it('interpolates linearly like numpy.quantile', () => {
    expect(empiricalQuantile([], 0.99)).toBe(0);
    expect(empiricalQuantile([5], 0.5)).toBe(5);
    expect(empiricalQuantile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(empiricalQuantile([3, 1, 2], 0.99)).toBeCloseTo(2.98, 12);
  });
});
