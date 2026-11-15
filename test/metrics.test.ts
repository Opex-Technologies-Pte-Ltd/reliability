import { describe, expect, it } from 'bun:test';
import {
  ReliabilityError,
  failureRate,
  inherentAvailability,
  meanTimeBetweenFailures,
  meanTimeToRepair,
  summarizeDowntime,
  timeBetweenFailures,
} from '../src';

const H = 3_600_000;

describe('basic metrics', () => {
  it('MTBF, MTTR, failure rate, inherent availability', () => {
    expect(meanTimeBetweenFailures(1000, 4)).toBe(250);
    expect(meanTimeBetweenFailures(1000, 0)).toBeNull();
    expect(meanTimeToRepair([2, 4, 6])).toBe(4);
    expect(meanTimeToRepair([])).toBeNull();
    expect(failureRate(4, 1000)).toBe(0.004);
    expect(failureRate(4, 0)).toBeNull();
    // A = MTBF / (MTBF + MTTR) = 250 / 254
    expect(inherentAvailability(250, 4)).toBeCloseTo(250 / 254, 15);
  });

  it('rejects negative inputs', () => {
    expect(() => meanTimeBetweenFailures(-1, 2)).toThrow(ReliabilityError);
    expect(() => meanTimeToRepair([1, -1])).toThrow(ReliabilityError);
    expect(() => inherentAvailability(0, 0)).toThrow(ReliabilityError);
  });
});

describe('timeBetweenFailures', () => {
  it('sorts, converts to hours and skips duplicate timestamps', () => {
    const t0 = Date.UTC(2026, 0, 1);
    expect(timeBetweenFailures([t0 + 30 * H, new Date(t0), t0 + 10 * H, t0 + 10 * H])).toEqual([10, 20]);
    expect(timeBetweenFailures([t0])).toEqual([]);
  });
});

describe('summarizeDowntime', () => {
  const ws = Date.UTC(2026, 0, 1);
  const window = { start: ws, end: ws + 100 * H };

  it('computes uptime, MTBF, MTTR and availability over a window', () => {
    const s = summarizeDowntime(
      [
        { start: ws + 10 * H, end: ws + 12 * H },
        { start: ws + 50 * H, end: ws + 54 * H },
      ],
      window,
    );
    expect(s).toEqual({
      observedHours: 100,
      downtimeHours: 6,
      uptimeHours: 94,
      failures: 2,
      mtbfHours: 47,
      mttrHours: 3,
      availability: 0.94,
    });
  });

  it('clips to the window, merges overlaps and runs open events to the end', () => {
    const s = summarizeDowntime(
      [
        { start: ws - 5 * H, end: ws + 2 * H }, // carried in: downtime, not a failure
        { start: ws + 20 * H, end: ws + 30 * H },
        { start: ws + 25 * H, end: ws + 35 * H }, // overlaps the previous one
        { start: ws + 90 * H, end: null }, // still down
      ],
      window,
    );
    expect(s.downtimeHours).toBe(2 + 15 + 10);
    expect(s.failures).toBe(3);
    expect(s.mttrHours).toBe((10 + 10 + 10) / 3);
    expect(s.availability).toBeCloseTo(0.73, 15);
  });

  it('no failures: null MTBF and MTTR, full availability', () => {
    const s = summarizeDowntime([], window);
    expect(s.mtbfHours).toBeNull();
    expect(s.mttrHours).toBeNull();
    expect(s.availability).toBe(1);
  });

  it('rejects an empty window and events that end before they start', () => {
    expect(() => summarizeDowntime([], { start: ws, end: ws })).toThrow(ReliabilityError);
    expect(() => summarizeDowntime([{ start: ws + H, end: ws }], window)).toThrow(ReliabilityError);
  });
});
