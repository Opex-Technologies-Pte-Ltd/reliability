/**
 * MTBF, MTTR, failure rate and availability from operating time and failure/repair events.
 * All durations are in hours; timestamps are epoch milliseconds or Date objects.
 */
import { ReliabilityError, assertNonNegative } from './errors.js';

const MS_PER_HOUR = 3_600_000;

/** Mean time between failures in hours: operating hours / failures. Null when there were no failures. */
export function meanTimeBetweenFailures(operatingHours: number, failures: number): number | null {
  assertNonNegative(operatingHours, 'operatingHours');
  assertNonNegative(failures, 'failures');
  return failures > 0 ? operatingHours / failures : null;
}

/** Mean time to repair in hours: the mean of the repair durations. Null for an empty list. */
export function meanTimeToRepair(repairHours: readonly number[]): number | null {
  if (repairHours.length === 0) return null;
  let sum = 0;
  repairHours.forEach((h, i) => {
    assertNonNegative(h, `repairHours[${i}]`);
    sum += h;
  });
  return sum / repairHours.length;
}

/** Failures per operating hour. Null when there was no operating time. */
export function failureRate(failures: number, operatingHours: number): number | null {
  assertNonNegative(failures, 'failures');
  assertNonNegative(operatingHours, 'operatingHours');
  return operatingHours > 0 ? failures / operatingHours : null;
}

/** Inherent availability MTBF / (MTBF + MTTR), a fraction in [0, 1]. */
export function inherentAvailability(mtbfHours: number, mttrHours: number): number {
  assertNonNegative(mtbfHours, 'mtbfHours');
  assertNonNegative(mttrHours, 'mttrHours');
  const total = mtbfHours + mttrHours;
  if (total === 0) throw new ReliabilityError('INVALID_INPUT', 'MTBF and MTTR are both 0');
  return mtbfHours / total;
}

function toMs(t: number | Date, name: string): number {
  const ms = t instanceof Date ? t.getTime() : t;
  if (typeof ms !== 'number' || !Number.isFinite(ms)) {
    throw new ReliabilityError('INVALID_INPUT', `${name} must be a finite timestamp`);
  }
  return ms;
}

/**
 * Hours between consecutive failures, from failure timestamps (any order).
 * Gaps of zero (duplicate timestamps) are skipped.
 */
export function timeBetweenFailures(timestamps: readonly (number | Date)[]): number[] {
  const ms = timestamps.map((t, i) => toMs(t, `timestamps[${i}]`)).sort((a, b) => a - b);
  const out: number[] = [];
  for (let i = 1; i < ms.length; i++) {
    const hours = (ms[i]! - ms[i - 1]!) / MS_PER_HOUR;
    if (hours > 0) out.push(hours);
  }
  return out;
}

/** One failure: when the unit went down, and when it was back (null = still down). */
export interface DowntimeEvent {
  start: number | Date;
  end: number | Date | null;
}

export interface DowntimeSummary {
  observedHours: number;
  /** Hours down inside the window; overlapping events are counted once. */
  downtimeHours: number;
  uptimeHours: number;
  /** Events that start inside the window. */
  failures: number;
  /** uptimeHours / failures; null without failures. */
  mtbfHours: number | null;
  /** Mean duration (clipped to the window) of the events that start inside it; null without failures. */
  mttrHours: number | null;
  /** uptimeHours / observedHours, a fraction in [0, 1]. */
  availability: number;
}

/**
 * Summarise a list of down events over an observation window `[start, end)`.
 * Events are clipped to the window; an open event (end = null) runs to the
 * window end. Downtime from an event that began before the window counts as
 * downtime but not as a failure.
 */
export function summarizeDowntime(
  events: readonly DowntimeEvent[],
  window: { start: number | Date; end: number | Date },
): DowntimeSummary {
  const ws = toMs(window.start, 'window.start');
  const we = toMs(window.end, 'window.end');
  if (!(we > ws)) throw new ReliabilityError('INVALID_INPUT', 'window.end must be after window.start');

  const clipped: Array<[number, number]> = [];
  let failures = 0;
  let failureDowntimeMs = 0;
  events.forEach((ev, i) => {
    const s = toMs(ev.start, `events[${i}].start`);
    const e = ev.end == null ? we : toMs(ev.end, `events[${i}].end`);
    if (e < s) throw new ReliabilityError('INVALID_INPUT', `events[${i}] ends before it starts`);
    const cs = Math.max(s, ws);
    const ce = Math.min(e, we);
    if (s >= ws && s < we) {
      failures++;
      failureDowntimeMs += Math.max(0, ce - cs);
    }
    if (ce > cs) clipped.push([cs, ce]);
  });

  clipped.sort((a, b) => a[0] - b[0]);
  let downMs = 0;
  let curS = -Infinity;
  let curE = -Infinity;
  for (const [s, e] of clipped) {
    if (s > curE) {
      if (curE > curS) downMs += curE - curS;
      curS = s;
      curE = e;
    } else if (e > curE) {
      curE = e;
    }
  }
  if (curE > curS) downMs += curE - curS;

  const observedHours = (we - ws) / MS_PER_HOUR;
  const downtimeHours = downMs / MS_PER_HOUR;
  const uptimeHours = observedHours - downtimeHours;
  return {
    observedHours,
    downtimeHours,
    uptimeHours,
    failures,
    mtbfHours: failures > 0 ? uptimeHours / failures : null,
    mttrHours: failures > 0 ? failureDowntimeMs / MS_PER_HOUR / failures : null,
    availability: uptimeHours / observedHours,
  };
}
