/** Machine-readable reason attached to every {@link ReliabilityError}. */
export type ReliabilityErrorCode =
  /** A number is NaN, infinite, or outside its allowed range (e.g. a negative time). */
  | 'INVALID_INPUT'
  /** A distribution parameter is not usable (e.g. Weibull shape ≤ 0). */
  | 'INVALID_PARAMETER'
  /** Arrays or vectors that must line up do not (e.g. covariate count). */
  | 'DIMENSION_MISMATCH';

/**
 * Thrown for invalid input. Functions never throw for "not enough data to fit":
 * they return `null` instead, so callers can treat that as an expected outcome.
 */
export class ReliabilityError extends Error {
  readonly code: ReliabilityErrorCode;

  constructor(code: ReliabilityErrorCode, message: string) {
    super(message);
    this.name = 'ReliabilityError';
    this.code = code;
  }
}

/** @internal */
export function assertFinite(value: number, name: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ReliabilityError('INVALID_INPUT', `${name} must be a finite number, got ${value}`);
  }
}

/** @internal */
export function assertNonNegative(value: number, name: string): void {
  assertFinite(value, name);
  if (value < 0) {
    throw new ReliabilityError('INVALID_INPUT', `${name} must be >= 0, got ${value}`);
  }
}

/** @internal */
export function assertPositive(value: number, name: string): void {
  assertFinite(value, name);
  if (value <= 0) {
    throw new ReliabilityError('INVALID_INPUT', `${name} must be > 0, got ${value}`);
  }
}
