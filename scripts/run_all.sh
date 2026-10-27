#!/usr/bin/env bash
set -euo pipefail
python -m mlds_rl.cli --output artifacts shuttle
python -m mlds_rl.cli --output artifacts cartpole --episodes 1500 --evaluation-episodes 500 --seed 42
python -m mlds_rl.cli --output artifacts cartpole --baseline --episodes 1500 --evaluation-episodes 500 --seed 43
python -m mlds_rl.cli --output artifacts pong --episodes 10000 --evaluation-episodes 500 --seed 42
python -m mlds_rl.cli --output artifacts pong --baseline --episodes 10000 --evaluation-episodes 500 --seed 43

