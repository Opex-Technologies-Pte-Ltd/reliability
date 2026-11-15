import { describe, expect, it } from 'bun:test';
import { ReliabilityError, fitWeibull, medianRanks, rankRegressionCovariance, type LifeEvent } from '../src';

function near(actual: number, expected: number, tol: number): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.abs(expected));
}

// Six failure times, the standard complete-data example of ReliaSoft's
// "Life Data Analysis Reference". Expected values: scripts/oracle.py, section 1
// (own rank regression with scipy.stats.linregress; scipy.stats.weibull_min.fit
// and lifelines.WeibullFitter for MLE).
const six = [16, 34, 53, 75, 93, 120];

// Ten units with three suspensions (oracle section 2, lifelines.WeibullFitter).
const censored: LifeEvent[] = [150, 340, 560, 800, 1130, 1720, 2470, 4210, 5230, 6890].map((hours, i) => ({
  hours,
  failed: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1][i] === 1,
}));

describe('medianRanks (Johnson adjusted ranks, Benard median ranks)', () => {
  it('reproduces the ReliaSoft example F,S,F,S,F → adjusted ranks 1, 2.25, 4.125', () => {
    const ranks = medianRanks([
      { hours: 1, failed: true },
      { hours: 2, failed: false },
      { hours: 3, failed: true },
      { hours: 4, failed: false },
      { hours: 5, failed: true },
    ]);
    expect(ranks.map((r) => r.adjustedRank)).toEqual([1, 2.25, 4.125]);
    expect(ranks[0]!.medianRank).toBeCloseTo(0.7 / 5.4, 15);
  });

  it('reproduces the ReliaSoft example F,S,F,F → 1, 2.3333, 3.6667', () => {
    const ranks = medianRanks([
      { hours: 1, failed: true },
      { hours: 2, failed: false },
      { hours: 3, failed: true },
      { hours: 4, failed: true },
    ]);
    expect(ranks[1]!.adjustedRank).toBeCloseTo(7 / 3, 12);
    expect(ranks[2]!.adjustedRank).toBeCloseTo(11 / 3, 12);
  });

  it('complete data gives plain order numbers, including tied times', () => {
    expect(medianRanks([10, 10, 20, 30, 30, 30]).map((r) => r.adjustedRank)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe('fitWeibull — median rank regression', () => {
  it('six-unit example matches the oracle', () => {
    const fit = fitWeibull(six)!;
    expect(fit.method).toBe('mrr');
    near(fit.shape, 1.426967110604193, 1e-12);
    near(fit.scale, 76.34541545544332, 1e-12);
    near(fit.rSquared!, 0.9911811557643286, 1e-12);
    expect(fit.failures).toBe(6);
    expect(fit.points).toHaveLength(6);
  });

  it('with suspensions matches the oracle (Johnson ranks)', () => {
    const fit = fitWeibull(censored)!;
    near(fit.shape, 0.797425965941538, 1e-12);
    near(fit.scale, 3479.941504692575, 1e-12);
    near(fit.rSquared!, 0.987447405350552, 1e-12);
    expect(fit.failures).toBe(7);
    expect(fit.suspensions).toBe(3);
  });

  it('number[] and all-failed LifeEvent[] give the same fit', () => {
    const a = fitWeibull(six)!;
    const b = fitWeibull(six.map((hours) => ({ hours, failed: true })))!;
    expect(b.shape).toBe(a.shape);
    expect(b.scale).toBe(a.scale);
  });

  it('carries the rank-regression covariance approximation', () => {
    const fit = fitWeibull(six)!;
    expect(fit.covariance).toEqual(rankRegressionCovariance(fit.shape, 6));
    const sumInvSq = 1 + 1 / 4 + 1 / 9 + 1 / 16 + 1 / 25 + 1 / 36;
    near(fit.covariance![0][0], sumInvSq / fit.shape ** 2, 1e-14);
    near(fit.covariance![1][1], 1 / 6, 1e-14);
  });

  it('a failure at t = 0 keeps its rank but is left out of the regression', () => {
    const fit = fitWeibull([0, 10, 20, 30])!;
    expect(fit.points.map((p) => p.adjustedRank)).toEqual([2, 3, 4]);
  });
});

describe('fitWeibull — maximum likelihood', () => {
  it('six-unit example matches scipy and lifelines', () => {
    const fit = fitWeibull(six, { method: 'mle' })!;
    near(fit.shape, 1.9326775387803645, 1e-6);
    near(fit.scale, 73.52604303546299, 1e-6);
    near(fit.logLikelihood!, -29.58492161490257, 1e-9);
    expect(fit.rSquared).toBeNull();
    expect(fit.points).toEqual([]);
  });

  it('right-censored sample matches lifelines, including the covariance of [ln η, ln β]', () => {
    const fit = fitWeibull(censored, { method: 'mle' })!;
    near(fit.shape, 0.9330383455178215, 1e-6);
    near(fit.scale, 3344.6509325046486, 1e-6);
    near(fit.logLikelihood!, -63.80506949184229, 1e-9);
    const c = fit.covariance!;
    near(c[0][0], 0.16421155, 1e-4);
    near(c[1][1], 0.09223552, 1e-4);
    near(c[0][1], 0.00323787, 1e-3);
    expect(c[1][0]).toBe(c[0][1]);
  });

  it('is scale-invariant: multiplying times by 1000 multiplies η by 1000', () => {
    const a = fitWeibull(censored, { method: 'mle' })!;
    const b = fitWeibull(
      censored.map((e) => ({ ...e, hours: e.hours * 1000 })),
      { method: 'mle' },
    )!;
    near(b.shape, a.shape, 1e-9);
    near(b.scale, a.scale * 1000, 1e-9);
  });

  it('a failure at t = 0 is invalid for MLE', () => {
    expect(() => fitWeibull([0, 10, 20], { method: 'mle' })).toThrow(ReliabilityError);
  });

  it('returns null when all failures happen at the same time with nothing later', () => {
    expect(fitWeibull([50, 50, 50], { method: 'mle' })).toBeNull();
  });
});

describe('fitWeibull — data checks', () => {
  it('returns null with fewer failures than required', () => {
    expect(fitWeibull([100])).toBeNull();
    expect(fitWeibull([{ hours: 5, failed: true }, { hours: 9, failed: false }])).toBeNull();
    expect(fitWeibull([1, 2, 3, 4], { minFailures: 5 })).toBeNull();
    expect(fitWeibull([], { method: 'mle' })).toBeNull();
  });

  it('returns null for two identical failure times (no slope)', () => {
    expect(fitWeibull([10, 10])).toBeNull();
  });

  it('throws for negative or non-finite times', () => {
    expect(() => fitWeibull([1, -2, 3])).toThrow(ReliabilityError);
    expect(() => fitWeibull([1, Number.NaN, 3])).toThrow(ReliabilityError);
    expect(() => fitWeibull([{ hours: Number.POSITIVE_INFINITY, failed: false }, 1, 2] as never)).toThrow(
      ReliabilityError,
    );
  });

  it('suspensions push later ranks up and lengthen the characteristic life', () => {
    const without = fitWeibull([10, 30, 40, 50, 60])!;
    const withSusp = fitWeibull([
      { hours: 10, failed: true },
      { hours: 20, failed: false },
      { hours: 30, failed: true },
      { hours: 40, failed: true },
      { hours: 50, failed: true },
      { hours: 60, failed: true },
    ])!;
    expect(withSusp.scale).toBeGreaterThan(without.scale);
  });
});
