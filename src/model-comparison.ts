/**
 * Scoring remaining-life predictors against what actually happened.
 */
import { weibullPhRemainingLife, type WeibullPhFit } from './proportional-hazards.js';

/** One evaluation snapshot: a unit's age and covariates, and what happened afterwards. */
export interface SurvivalEvalCase {
  ageHours: number;
  x: number[];
  /** Observed remaining life in hours: to failure, or to the end of the observation window. */
  actualRemainingHours: number;
  /** True when the unit failed within the window (so `actualRemainingHours` is exact). */
  failedWithinWindow: boolean;
}

export interface SurvivalModelCandidate {
  name: string;
  /** Predicted remaining life in hours, or null when the model has no prediction. */
  predict: (ageHours: number, x: number[]) => number | null;
}

export interface SurvivalModelScore {
  name: string;
  /** Harrell's C over comparable pairs (null when there are none). */
  concordance: number | null;
  comparablePairs: number;
  /** Median of |predicted − actual| / actual, in percent, over units that failed in the window. */
  medianAbsPctError: number | null;
  evaluatedFailures: number;
}

/**
 * Score candidate remaining-life predictors on the same snapshots with
 * Harrell's concordance index and the median absolute percentage error.
 *
 * A pair is comparable when the unit with the shorter observed remaining life
 * failed (its true value is known to be shorter); pairs where the censored
 * unit did not outlive the failure, and pairs of two censored units, are
 * skipped. A prediction tie counts as half concordant. Percentage error uses
 * only units that failed within the window.
 */
export function compareSurvivalModels(
  cases: readonly SurvivalEvalCase[],
  models: readonly SurvivalModelCandidate[],
): SurvivalModelScore[] {
  return models.map((model) => {
    const preds = cases.map((c) => model.predict(c.ageHours, c.x));
    let concordant = 0;
    let comparable = 0;
    for (let i = 0; i < cases.length; i++) {
      if (preds[i] == null || !Number.isFinite(preds[i]!)) continue;
      for (let j = i + 1; j < cases.length; j++) {
        if (preds[j] == null || !Number.isFinite(preds[j]!)) continue;
        // Mixed pairs: one failure, one censored unit.
        let early: number;
        let late: number;
        if (cases[i]!.failedWithinWindow && !cases[j]!.failedWithinWindow) {
          early = i;
          late = j;
        } else if (!cases[i]!.failedWithinWindow && cases[j]!.failedWithinWindow) {
          early = j;
          late = i;
        } else {
          continue; // both failed (handled below) or both censored (not comparable)
        }
        if (cases[late]!.actualRemainingHours <= cases[early]!.actualRemainingHours) continue;
        comparable++;
        const dp = preds[early]! - preds[late]!;
        if (Math.abs(dp) < 1e-12) concordant += 0.5;
        else if (dp < 0) concordant += 1;
      }
    }
    // Pairs of two failures: both remaining lives are known.
    for (let i = 0; i < cases.length; i++) {
      if (!cases[i]!.failedWithinWindow || preds[i] == null) continue;
      for (let j = i + 1; j < cases.length; j++) {
        if (!cases[j]!.failedWithinWindow || preds[j] == null) continue;
        if (cases[i]!.actualRemainingHours === cases[j]!.actualRemainingHours) continue;
        const early = cases[i]!.actualRemainingHours < cases[j]!.actualRemainingHours ? i : j;
        const late = early === i ? j : i;
        comparable++;
        const dp = preds[early]! - preds[late]!;
        if (Math.abs(dp) < 1e-12) concordant += 0.5;
        else if (dp < 0) concordant += 1;
      }
    }

    const errs: number[] = [];
    for (let i = 0; i < cases.length; i++) {
      if (!cases[i]!.failedWithinWindow || preds[i] == null) continue;
      const actual = Math.max(cases[i]!.actualRemainingHours, 1e-9);
      errs.push((Math.abs(preds[i]! - cases[i]!.actualRemainingHours) / actual) * 100);
    }
    errs.sort((m, n) => m - n);
    const median =
      errs.length === 0
        ? null
        : errs.length % 2 === 1
          ? errs[(errs.length - 1) / 2]!
          : (errs[errs.length / 2 - 1]! + errs[errs.length / 2]!) / 2;
    return {
      name: model.name,
      concordance: comparable > 0 ? concordant / comparable : null,
      comparablePairs: comparable,
      medianAbsPctError: median,
      evaluatedFailures: errs.length,
    };
  });
}

/** Predictor for {@link compareSurvivalModels} from a proportional-hazards fit (null on a covariate-count mismatch or a negative age). */
export function weibullPhPredictor(fit: WeibullPhFit): (ageHours: number, x: number[]) => number | null {
  return (age, x) => {
    if (x.length !== fit.coef.length || !(age >= 0) || !Number.isFinite(age)) return null;
    return weibullPhRemainingLife(fit, x, age).medianHours;
  };
}
