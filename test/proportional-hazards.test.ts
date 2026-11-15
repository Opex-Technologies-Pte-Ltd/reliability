// Weibull proportional hazards: known-answer tests on simulated data with a
// fixed-seed generator, plus checks of the delta-method gradient.
import { describe, expect, it } from 'bun:test';
import {
  ReliabilityError,
  compareSurvivalModels,
  fitWeibull,
  fitWeibullPh,
  remainingUsefulLife,
  weibullPhPredictor,
  weibullPhRemainingLife,
  weibullPhScale,
  type PhSample,
  type SurvivalEvalCase,
  type WeibullPhFit,
} from '../src';

let seed = 12345;
const rand = (): number => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};
const gauss = (): number => {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
};

/** t = η₀·(E·e^{−γx})^{1/p}, E ~ Exp(1), right-censored at `censorAt`. */
function samplePh(n: number, p: number, eta0: number, gamma: number, censorAt: number | null): PhSample[] {
  const out: PhSample[] = [];
  for (let i = 0; i < n; i++) {
    const x = gauss();
    const e = -Math.log(Math.max(rand(), 1e-12));
    const t = eta0 * Math.pow(e * Math.exp(-gamma * x), 1 / p);
    const failed = censorAt == null || t <= censorAt;
    out.push({ hours: failed ? t : censorAt!, failed, x: [x] });
  }
  return out;
}

describe('fitWeibullPh', () => {
  it('recovers shape and scale when the covariate has no effect', () => {
    seed = 777;
    const fit = fitWeibullPh(samplePh(300, 2, 1000, 0, 4000), ['health'], {
      warmStart: { shape: 1.5, baseScale: 900 },
    })!;
    expect(fit.converged).toBe(true);
    expect(fit.shape).toBeGreaterThan(1.7);
    expect(fit.shape).toBeLessThan(2.3);
    expect(fit.baseScale).toBeGreaterThan(850);
    expect(fit.baseScale).toBeLessThan(1150);
    expect(Math.abs(fit.coef[0]!)).toBeLessThan(0.25);
  });

  it('recovers the covariate coefficient under censoring', () => {
    seed = 4242;
    const fit = fitWeibullPh(samplePh(250, 1.5, 500, 0.8, 1200), ['health'])!;
    expect(fit.converged).toBe(true);
    // The fit reports γ per standard deviation of x; x ~ N(0, 1) so it estimates 0.8.
    expect(fit.coef[0]!).toBeGreaterThan(0.5);
    expect(fit.coef[0]!).toBeLessThan(1.1);
    expect(fit.hazardRatios[0]).toBeCloseTo(Math.exp(fit.coef[0]!), 12);
    const se = fit.coefSe[0]!;
    expect(fit.coef[0]! - 1.96 * se).toBeLessThan(0.8);
    expect(fit.coef[0]! + 1.96 * se).toBeGreaterThan(0.8);
    expect(fit.suspensions).toBeGreaterThan(0);
  });

  it('with no covariate effect it agrees with the plain Weibull MLE', () => {
    // Constant covariate: γ carries no information, so the PH likelihood
    // reduces to the plain Weibull likelihood.
    seed = 99;
    const raw = samplePh(200, 2, 800, 0, 1000);
    const ph = fitWeibullPh(
      raw.map((s) => ({ ...s, x: [5] })),
      ['health'],
      { warmStart: { shape: 2, baseScale: 800 } },
    )!;
    const mle = fitWeibull(raw, { method: 'mle' })!;
    expect(Math.abs(ph.coef[0]!)).toBeLessThan(1e-6);
    expect(ph.shape).toBeCloseTo(mle.shape, 4);
    expect(ph.baseScale / mle.scale).toBeCloseTo(1, 4);
    expect(ph.logLik).toBeCloseTo(mle.logLikelihood!, 4);
  });

  it('handles time-varying covariates as counting-process pieces', () => {
    seed = 2026;
    const tau = 250;
    const uTau = (tau / 500) ** 1.5;
    const samples: PhSample[] = [];
    for (let i = 0; i < 300; i++) {
      const x1 = gauss();
      const x2 = gauss();
      const e = -Math.log(Math.max(rand(), 1e-12));
      if (e <= uTau * Math.exp(0.8 * x1)) {
        samples.push({ hours: 500 * Math.pow(e * Math.exp(-0.8 * x1), 1 / 1.5), failed: true, x: [x1], group: `u${i}` });
      } else {
        const rest = (e - uTau * Math.exp(0.8 * x1)) * Math.exp(-0.8 * x2);
        samples.push({ hours: tau, failed: false, x: [x1], group: `u${i}` });
        samples.push({ hours: 500 * Math.pow(uTau + rest, 1 / 1.5), startHours: tau, failed: true, x: [x2], group: `u${i}` });
      }
    }
    const fit = fitWeibullPh(samples, ['health'], { warmStart: { shape: 1.5, baseScale: 500 } })!;
    expect(fit.converged).toBe(true);
    expect(fit.coef[0]!).toBeGreaterThan(0.55);
    expect(fit.coef[0]!).toBeLessThan(1.05);
    expect(fit.failures).toBe(300);
    expect(fit.suspensions).toBe(0);
  });

  it('returns null with too few failures; throws on malformed samples', () => {
    seed = 1;
    expect(fitWeibullPh(samplePh(4, 2, 100, 0, null), ['h'])).toBeNull();
    expect(fitWeibullPh([], ['h'])).toBeNull();
    expect(() => fitWeibullPh(samplePh(50, 2, 100, 0, null), [])).toThrow(ReliabilityError);
    const bad = samplePh(50, 2, 100, 0, null);
    bad[10]!.x = [Number.NaN];
    expect(() => fitWeibullPh(bad, ['h'])).toThrow(ReliabilityError);
    const wrongDim = samplePh(50, 2, 100, 0, null);
    wrongDim[3]!.x = [1, 2];
    expect(() => fitWeibullPh(wrongDim, ['h'])).toThrow(ReliabilityError);
    const badStart = samplePh(50, 2, 100, 0, null);
    badStart[0] = { ...badStart[0]!, startHours: badStart[0]!.hours };
    expect(() => fitWeibullPh(badStart, ['h'])).toThrow(ReliabilityError);
  });
});

describe('weibullPhRemainingLife', () => {
  const fit: WeibullPhFit = {
    shape: 1.7,
    baseScale: 900,
    coef: [0.6, -0.3],
    coefSe: [null, null],
    covariateNames: ['a', 'b'],
    covariateMeans: [10, 2],
    covariateStdDevs: [4, 0.5],
    hazardRatios: [Math.exp(0.6), Math.exp(-0.3)],
    logLik: 0,
    iterations: 0,
    converged: true,
    failures: 50,
    suspensions: 0,
    cov: null,
  };

  it('equals the plain Weibull result at the covariate-adjusted scale', () => {
    const x = [13, 1.5];
    const eta = weibullPhScale(fit, x);
    for (const age of [0, 200, 1500]) {
      expect(weibullPhRemainingLife(fit, x, age).medianHours).toBeCloseTo(
        remainingUsefulLife({ shape: fit.shape, scale: eta }, age).medianHours,
        9,
      );
    }
    expect(weibullPhScale(fit, [10, 2])).toBe(900);
  });

  it('delta-method gradient matches finite differences of ln t for every parameter', () => {
    const x = [13, 1.5];
    const theta0 = [Math.log(fit.baseScale), Math.log(fit.shape), ...fit.coef];
    const lnT = (theta: number[], age: number) => {
      const f = { ...fit, baseScale: Math.exp(theta[0]!), shape: Math.exp(theta[1]!), coef: theta.slice(2) };
      return Math.log(weibullPhRemainingLife(f, x, age).medianHours + age);
    };
    const z = 1.6448536269514722;
    for (const age of [0, 300, 2000]) {
      const t = weibullPhRemainingLife(fit, x, age).medianHours + age;
      for (let k = 0; k < theta0.length; k++) {
        const h = 1e-6;
        const up = theta0.slice();
        const dn = theta0.slice();
        up[k]! += h;
        dn[k]! -= h;
        const deriv = (lnT(up, age) - lnT(dn, age)) / (2 * h);
        const cov = theta0.map((_, i) => theta0.map((__, j) => (i === k && j === k ? 1 : 0)));
        const band = weibullPhRemainingLife({ ...fit, cov }, x, age).interval!;
        const halfWidth = Math.log(band.upperHours + age) - Math.log(t);
        expect(Math.abs(halfWidth - z * Math.abs(deriv))).toBeLessThan(1e-6 * Math.max(1, Math.abs(deriv)));
      }
    }
  });

  it('throws on a covariate-count mismatch', () => {
    expect(() => weibullPhRemainingLife(fit, [1], 0)).toThrow(ReliabilityError);
  });
});

describe('compareSurvivalModels', () => {
  it('ranks a covariate model above a covariate-blind one', () => {
    seed = 31337;
    const fit = fitWeibullPh(samplePh(400, 1.5, 500, 0.8, 1200), ['health'])!;
    const cases: SurvivalEvalCase[] = [];
    for (let i = 0; i < 300; i++) {
      const x = gauss();
      const age = 50 + 100 * rand();
      const e = -Math.log(Math.max(rand(), 1e-12));
      const life = 500 * Math.pow(e * Math.exp(-0.8 * x), 1 / 1.5);
      const failed = life <= age + 1500;
      cases.push({ ageHours: age, x: [x], actualRemainingHours: failed ? Math.max(0, life - age) : 1500 - age, failedWithinWindow: failed });
    }
    const pooled = { name: 'pooled', predict: (age: number) => remainingUsefulLife({ shape: 1.5, scale: 500 }, age).medianHours };
    const [base, ph] = compareSurvivalModels(cases, [pooled, { name: 'ph', predict: weibullPhPredictor(fit) }]);
    expect(ph!.concordance!).toBeGreaterThan(0.7);
    expect(ph!.concordance!).toBeGreaterThan(base!.concordance! + 0.1);
    expect(ph!.medianAbsPctError!).toBeLessThan(base!.medianAbsPctError!);
  });

  it('counts pairs by hand on a tiny example', () => {
    const cases: SurvivalEvalCase[] = [
      { ageHours: 0, x: [], actualRemainingHours: 10, failedWithinWindow: true },
      { ageHours: 0, x: [], actualRemainingHours: 20, failedWithinWindow: true },
      { ageHours: 0, x: [], actualRemainingHours: 30, failedWithinWindow: false },
      { ageHours: 0, x: [], actualRemainingHours: 5, failedWithinWindow: false },
    ];
    // Comparable: (0,1), (0,2), (1,2). Case 3 is censored before every failure.
    const preds = [12, 25, 18, 1];
    const [score] = compareSurvivalModels(cases, [{ name: 'm', predict: (_a, _x) => preds.shift() ?? null }]);
    expect(score!.comparablePairs).toBe(3);
    // (0,1): 12 < 25 concordant; (0,2): 12 < 18 concordant; (1,2): 25 > 18 discordant.
    expect(score!.concordance).toBeCloseTo(2 / 3, 12);
    // |12−10|/10 = 20 %, |25−20|/20 = 25 % → median 22.5 %.
    expect(score!.medianAbsPctError).toBeCloseTo(22.5, 12);
  });

  it('returns null metrics when a model never predicts', () => {
    const [s] = compareSurvivalModels(
      [{ ageHours: 1, x: [1], actualRemainingHours: 100, failedWithinWindow: true }],
      [{ name: 'none', predict: () => null }],
    );
    expect(s!.concordance).toBeNull();
    expect(s!.medianAbsPctError).toBeNull();
  });
});
