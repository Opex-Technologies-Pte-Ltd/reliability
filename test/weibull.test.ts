// Expected values come from scipy.stats.weibull_min (see scripts/oracle.py, section 3).
import { describe, expect, it } from 'bun:test';
import {
  ReliabilityError,
  bLife,
  conditionalReliability,
  remainingUsefulLife,
  weibullCdf,
  weibullHazard,
  weibullMean,
  weibullPdf,
  weibullQuantile,
  weibullReliability,
} from '../src';

const p = { shape: 1.8, scale: 1200 };

/** Relative closeness: |a − b| ≤ tol·|b|. */
function near(actual: number, expected: number, tol = 1e-12): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(Math.abs(expected), 1e-300));
}

describe('Weibull distribution functions (β = 1.8, η = 1200 h)', () => {
  it('pdf, cdf, reliability and hazard match scipy', () => {
    const ref = [
      [100, 0.00020313689756560534, 0.011350039986183575, 0.9886499600138164, 0.0002054689786896532],
      [1200, 0.0005518191617571636, 0.6321205588285577, 0.36787944117144233, 0.0015],
      [3000, 1.716367833589959e-5, 0.9945024764196675, 0.005497523580332466, 0.003122074527757024],
    ] as const;
    for (const [t, pdf, cdf, r, h] of ref) {
      near(weibullPdf(p, t), pdf);
      near(weibullCdf(p, t), cdf);
      near(weibullReliability(p, t), r);
      near(weibullHazard(p, t), h);
    }
  });

  it('at t = η, F = 1 − 1/e regardless of shape', () => {
    near(weibullCdf({ shape: 0.5, scale: 10 }, 10), 1 - Math.exp(-1));
    near(weibullCdf({ shape: 4, scale: 10 }, 10), 1 - Math.exp(-1));
  });

  it('shape 1 is the exponential distribution (constant hazard 1/η)', () => {
    const exp = { shape: 1, scale: 500 };
    near(weibullHazard(exp, 0), 1 / 500);
    near(weibullHazard(exp, 2000), 1 / 500);
    near(weibullMean(exp), 500);
  });

  it('mean life and B-lives match scipy', () => {
    near(weibullMean(p), 1067.1440789426554, 1e-13);
    near(bLife(p, 10), 343.7356026992573);
    near(bLife(p, 50), 978.9284416641858);
    near(bLife(p, 1), 93.17117348292841);
    near(weibullQuantile(p, 0.1), bLife(p, 10));
  });

  it('conditional reliability R(700)/R(500)', () => {
    near(conditionalReliability(p, 500, 200), 0.8418304524530847);
    near(conditionalReliability(p, 0, 700), weibullReliability(p, 700));
  });

  it('rejects invalid parameters and inputs with typed errors', () => {
    expect(() => weibullCdf({ shape: 0, scale: 1 }, 1)).toThrow(ReliabilityError);
    expect(() => weibullCdf({ shape: 1, scale: -1 }, 1)).toThrow(ReliabilityError);
    expect(() => weibullCdf(p, -1)).toThrow(ReliabilityError);
    expect(() => weibullCdf(p, Number.NaN)).toThrow(ReliabilityError);
    expect(() => bLife(p, 0)).toThrow(ReliabilityError);
    expect(() => bLife(p, 100)).toThrow(ReliabilityError);
    try {
      weibullMean({ shape: Number.POSITIVE_INFINITY, scale: 1 });
      throw new Error('unreachable');
    } catch (e) {
      expect(e).toBeInstanceOf(ReliabilityError);
      expect((e as ReliabilityError).code).toBe('INVALID_PARAMETER');
    }
  });
});

describe('remainingUsefulLife', () => {
  it('median residual life matches a root-find on R(age + m)/R(age) = 0.5 (scipy brentq)', () => {
    near(remainingUsefulLife(p, 0).medianHours, 978.9284416641858);
    near(remainingUsefulLife(p, 500).medianHours, 631.7619692319275, 1e-10);
    near(remainingUsefulLife(p, 2000).medianHours, 290.375542239175, 1e-10);
  });

  it('has no interval without a covariance', () => {
    expect(remainingUsefulLife(p, 100).interval).toBeNull();
  });

  it('interval brackets the median and widens with confidence', () => {
    const model = {
      ...p,
      covariance: [
        [0.02, 0.001],
        [0.001, 0.03],
      ] as const,
    };
    for (const age of [0, 300, 1500, 5000]) {
      const r90 = remainingUsefulLife(model, age);
      const r99 = remainingUsefulLife(model, age, { confidence: 0.99 });
      expect(r90.interval!.lowerHours).toBeLessThanOrEqual(r90.medianHours);
      expect(r90.interval!.upperHours).toBeGreaterThanOrEqual(r90.medianHours);
      expect(r99.interval!.lowerHours).toBeLessThanOrEqual(r90.interval!.lowerHours);
      expect(r99.interval!.upperHours).toBeGreaterThanOrEqual(r90.interval!.upperHours);
    }
  });

  it('delta-method gradient matches a finite-difference gradient of ln t', () => {
    // Var(ln t) with a unit covariance on one parameter is the squared partial derivative.
    const lnT = (lnEta: number, lnBeta: number, age: number) =>
      Math.log(remainingUsefulLife({ shape: Math.exp(lnBeta), scale: Math.exp(lnEta) }, age).medianHours + age);
    const z = 1.6448536269514722;
    for (const age of [0, 400, 1500]) {
      const h = 1e-6;
      const a = Math.log(p.scale);
      const b = Math.log(p.shape);
      const dEta = (lnT(a + h, b, age) - lnT(a - h, b, age)) / (2 * h);
      const dBeta = (lnT(a, b + h, age) - lnT(a, b - h, age)) / (2 * h);
      const t = remainingUsefulLife(p, age).medianHours + age;
      const etaOnly = remainingUsefulLife({ ...p, covariance: [[1, 0], [0, 0]] }, age).interval!;
      const betaOnly = remainingUsefulLife({ ...p, covariance: [[0, 0], [0, 1]] }, age).interval!;
      near(Math.log(etaOnly.upperHours + age) - Math.log(t), z * Math.abs(dEta), 1e-6);
      near(Math.log(betaOnly.upperHours + age) - Math.log(t), z * Math.abs(dBeta), 1e-6);
    }
  });

  it('rejects a negative age', () => {
    expect(() => remainingUsefulLife(p, -1)).toThrow(ReliabilityError);
    expect(() => remainingUsefulLife(p, 1, { confidence: 1 })).toThrow(ReliabilityError);
  });
});
