# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bamphd/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bamphd/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An hour-by-hour quantile forecast of Grand Theft Auto VI's launch-day Twitch viewers. The program anchors on the first hour's print, learns the release-day shape — the early-hours dip, the unlock-timezone ramp as each region wakes up, the evening peak — from the comparable past game launches shipped with each case, weighting most the launches that opened at a similar scale, and sets honest quantile bands from how much those launches disagree. Python 3.13, standard library only, no network, fully deterministic: two runs produce byte-identical forecasts.

## Run it

```sh
MAKEFUN_INPUT_DIR=<case-dir> MAKEFUN_OUTPUT_DIR=<out-dir> python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=3r8va27c&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
