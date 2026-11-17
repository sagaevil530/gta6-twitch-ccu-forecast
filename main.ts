// gta6-twitch-ccu-forecast-01 — forecast GTA VI's launch-day Twitch viewers,
// hours 2..24, as five quantiles an hour (Node 22, standard library only).
//
// Idea. Hour 1 of a launch day is a bad anchor on its own: for some past
// launches it already held the day's peak (early access / a simultaneous PC
// unlock the evening before), for others it was a near-empty category whose
// unlock was still hours away, so the day's peak ran 100x-2000x hour 1.
// What separates the two groups is visible at the checkpoint: how many live
// channels the category already has in hour 1. A category with hundreds of
// channels live at 00:00 UTC is already unlocked in the UK / central Europe
// (a rolling midnight-local launch, like GTA VI); one with a handful of
// channels is still dormant and its peak is yet to come.
//
// So the forecaster works in two steps, both fitted at run time from the
// case's own `comparables` (past launches' hourly curves):
//
//   1. Peak. Split the comparables into a "live" pool (hour-1 channels >= 150)
//      and a "dormant" pool, and pick the pool matching the target's hour-1
//      channels. Inside the pool, predict the day's peak (log scale) as the
//      average of two estimators: the pool median of log(peak / hour 1), and
//      a least-squares fit of log(peak) on log(hour 1). The first is robust,
//      the second captures that bigger hour-1 prints grow less (its slope is
//      well below 1); blending them beats either alone in leave-one-out
//      tests over the comparables. The predicted peak is clamped to be at
//      least hour 1 itself, since the peak is taken over the whole day.
//
//   2. Shape. For every comparable in the pool, predict its peak the same
//      way but leaving that comparable out, and record the log residuals
//      log(viewers_h) - log(predicted peak) for each hour h. The forecast
//      quantiles for hour h are the target's predicted peak times the
//      empirical quantiles of those residuals. This carries the timezone
//      wave (the rise into the US unlock hours 5-9, the evening peak around
//      hours 19-21, the overnight fall) and an honest, hour-varying spread:
//      early hours are genuinely uncertain, late hours are tight because
//      every launch's curve is pinned to its peak by then.
//
// Everything is deterministic (no randomness, fixed iteration order) and
// runs in well under a second. If the case carries no usable comparables,
// it falls back to a flat persistence forecast with a widening spread.
//
// Usage (outside the grader): mkdir -p out &&
//   MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// It reads every *.in in the input directory and writes <id>.out per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Hour-1 channels at or above which a launch counts as already live. */
const LIVE_CHANNELS = 150;
/** Minimum pool size before falling back to all comparables. */
const MIN_POOL = 6;
/** Weight of the least-squares peak estimator (rest: median ratio). */
const OLS_WEIGHT = 0.5;

interface KnownRow {
  hour: number;
  viewers: number;
  channels?: number | null;
}

interface HourlyRow {
  from: string;
  viewers: number;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyRow[];
}

interface Case {
  id?: string;
  game?: { name?: string };
  levels: number[];
  known: KnownRow[];
  forecast: Array<{ hour: number }>;
  history_hourly?: Array<{ from: string; viewers: number; channels?: number | null; reads?: number }>;
  comparables?: Comparable[];
}

/** One comparable's release-day curve, reduced to what the model needs. */
interface Curve {
  /** Hour-1 viewers (must be > 0). */
  p1: number;
  /** Hour-1 live channels, or null when unknown. */
  ch1: number | null;
  /** log viewers for hours 1..24; null where the source has no point. */
  logHours: Array<number | null>;
  /** log of the day's peak viewers. */
  logPeak: number;
}

const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

/** Empirical quantile of a sorted array by linear interpolation. */
const quantileOf = (sorted: number[], tau: number): number => {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = tau * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] * (1 - (pos - lo)) + sorted[hi] * (pos - lo);
};

/** Solve a small linear system by Gaussian elimination with partial pivoting. */
const solveLinear = (a: number[][], b: number[]): number[] | null => {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    }
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    const pv = m[col][col];
    for (let j = col; j <= n; j++) m[col][j] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === col || m[r][col] === 0) continue;
      const f = m[r][col];
      for (let j = col; j <= n; j++) m[r][j] -= f * m[col][j];
    }
  }
  return m.map((row) => row[n]);
};

/**
 * A peak predictor fitted on a pool of curves. Returns log(peak) for a
 * curve given its log hour-1 viewers: the blend of the pool median of
 * log(peak / hour 1) and the least-squares fit of log(peak) on log(hour 1),
 * clamped so the peak is never below hour 1.
 */
const fitPeakPredictor = (pool: Curve[]): ((logP1: number) => number) => {
  const medianLogRatio = median(pool.map((c) => c.logPeak - Math.log(c.p1)));
  const xs = pool.map((c) => Math.log(c.p1));
  const ys = pool.map((c) => c.logPeak);
  const n = pool.length;
  const sx = xs.reduce((a, x) => a + x, 0);
  const sy = ys.reduce((a, y) => a + y, 0);
  const sxx = xs.reduce((a, x) => a + x * x, 0);
  const sxy = xs.reduce((a, x, i) => a + x * ys[i], 0);
  const beta = solveLinear(
    [
      [n, sx],
      [sx, sxx],
    ],
    [sy, sxy],
  );
  return (logP1: number): number => {
    const byMedian = logP1 + medianLogRatio;
    const byOls = beta === null ? byMedian : beta[0] + beta[1] * logP1;
    const blended = (1 - OLS_WEIGHT) * byMedian + OLS_WEIGHT * byOls;
    return Math.max(blended, logP1);
  };
};

/** Whether a curve's launch was already live in hour 1. */
const isLive = (ch1: number | null, p1: number): boolean =>
  ch1 !== null ? ch1 >= LIVE_CHANNELS : p1 >= 10000;

/** Extract a comparable's release-day curve, or null when unusable. */
const toCurve = (c: Comparable, dayStartMs: number): Curve | null => {
  const logHours: Array<number | null> = new Array(24).fill(null);
  let ch1: number | null = null;
  for (const row of c.hourly ?? []) {
    const t = Date.parse(row.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.round((t - dayStartMs) / 3_600_000);
    if (h < 0 || h > 23) continue;
    if (typeof row.viewers === 'number' && row.viewers > 0) {
      logHours[h] = Math.log(row.viewers);
    }
    if (h === 0 && typeof row.channels === 'number') ch1 = row.channels;
  }
  if (logHours[0] === null) return null;
  const present = logHours.filter((v): v is number => v !== null);
  if (present.length < 20) return null;
  return {
    p1: Math.exp(logHours[0] as number),
    ch1,
    logHours,
    logPeak: Math.max(...present),
  };
};

/** Normalise a name for the "same game" check (the target's own history). */
const normName = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]/g, '');

/** Fallback forecast when no comparable curve is usable: flat, widening. */
const fallbackQuantiles = (p1: number, h: number): number[] => {
  const z = [-1.645, -0.674, 0, 0.674, 1.645];
  const s = 0.15 * Math.sqrt(h - 1);
  return z.map((zi) => Math.round(p1 * Math.exp(zi * s) * 100) / 100);
};

/** Forecast one case: 23 rows of five quantiles for hours 2..24. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  // Hour-1 live channels: from the known row, else the last hourly history
  // point (its hour is the one just ended), else unknown.
  let targetCh1: number | null =
    typeof first.channels === 'number' ? first.channels : null;
  if (targetCh1 === null && Array.isArray(c.history_hourly) && c.history_hourly.length > 0) {
    const last = c.history_hourly[c.history_hourly.length - 1];
    if (typeof last.channels === 'number') targetCh1 = last.channels;
  }
  const hours = (c.forecast.length > 0 ? c.forecast : Array.from({ length: 23 }, (_, i) => ({ hour: i + 2 }))).map(
    (f) => f.hour,
  );

  const targetName = normName(c.game?.name ?? '');
  const curves: Curve[] = [];
  for (const comp of c.comparables ?? []) {
    // The target game's own pre-release history is not a launch curve.
    const compName = normName(comp.name ?? '');
    if (targetName && compName && (compName.includes(targetName) || targetName.includes(compName))) continue;
    const dayStartMs = Date.parse(comp.day_start);
    if (!Number.isFinite(dayStartMs)) continue;
    const curve = toCurve(comp, dayStartMs);
    if (curve !== null) curves.push(curve);
  }

  if (curves.length < MIN_POOL) {
    return hours.map((h) => ({ hour: h, q: fallbackQuantiles(p1, h) }));
  }

  const liveTarget = isLive(targetCh1, p1);
  let pool = curves.filter((cv) => isLive(cv.ch1, cv.p1) === liveTarget);
  if (pool.length < MIN_POOL) pool = curves;

  // Leave-one-out log residuals per hour inside the pool: for each curve,
  // predict its peak from the rest of the pool and record how each hour's
  // actual log viewers differ from that prediction.
  const residuals: number[][] = Array.from({ length: 24 }, () => []);
  for (let i = 0; i < pool.length; i++) {
    const rest = pool.filter((_, j) => j !== i);
    if (rest.length === 0) continue;
    const predict = fitPeakPredictor(rest);
    const logPeakPred = predict(Math.log(pool[i].p1));
    pool[i].logHours.forEach((lv, h) => {
      if (lv !== null) residuals[h].push(lv - logPeakPred);
    });
  }
  const sortedResiduals = residuals.map((r) => [...r].sort((a, b) => a - b));

  const predictTarget = fitPeakPredictor(pool);
  const logPeakTarget = predictTarget(Math.log(p1));

  return hours.map((h) => {
    const res = sortedResiduals[h - 1];
    let q: number[];
    if (res.length === 0) {
      q = fallbackQuantiles(p1, h);
    } else {
      q = LEVELS.map((tau) => {
        const v = Math.exp(logPeakTarget + quantileOf(res, tau));
        return Math.min(1e9, Math.max(1, Math.round(v * 100) / 100));
      });
    }
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { hour: h, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
