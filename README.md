# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sagaevil530/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sagaevil530/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), from everything recorded up to the end of the first hour. `main.ts` reads one case file, fits a forecaster on that case's own past-launch comparables at run time, and writes five quantiles (5%, 25%, 50%, 75%, 95%) for hours 2–24.

The model: hour 1 alone is a bad anchor — some launches already held their day's peak in hour 1, others were still dormant with their peak 100×–2000× hour 1. What separates them is visible at the checkpoint: the category's hour-1 live-channel count. So the program splits the comparables into a "live" pool (hour-1 channels ≥ 150) and a "dormant" pool, picks the pool matching the target, predicts the day's peak as a 50/50 blend of the pool median of log(peak / hour 1) and a least-squares fit of log(peak) on log(hour 1), then sets each hour's quantiles from the leave-one-out log residuals of that pool — which carries the timezone wave (rise into the US unlock hours, evening peak, overnight fall) and an honest, hour-varying spread. Leave-one-out over the 37 usable comparables in the kit (each launch forecast from the other 36, scored against its own hours 2–24 with the contest's pinball loss): mean loss 0.396 vs 0.930 for the house persistence baseline; on the already-live launches most like GTA VI (hour-1 channels ≥ 150) 0.263 vs 0.331, and on console-only launches 0.191 vs 0.323.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads every `*.in` in the input directory and writes `<id>.out` per case to the output directory. It needs Node.js 22 and nothing else — standard library only, no network, deterministic, well under the 8-second limit per case.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
