#!/bin/bash

sessionName="cka"
device1="cuda:0"
device2="cuda:2"
ckaDevice="cuda:3"

tmux kill-session -t $sessionName
tmux new -s $sessionName -d

tmux send-keys -t $sessionName "conda deactivate" Enter
tmux send-keys -t $sessionName "uv run scripts/cka_ablations.py --descendant \"nvidia/AceReason-Nemotron-1.1-7B\" --cka-device $ckaDevice --device1 $device1 --device2 $device2" Enter
tmux send-keys -t $sessionName "uv run scripts/cka_ablations.py --descendant \"sail/Qwen2.5-Math-7B-Oat-Zero\" --cka-device $ckaDevice --device1 $device1 --device2 $device2" Enter
tmux send-keys -t $sessionName "uv run scripts/cka_ablations.py --descendant \"open-r1/OpenR1-Distill-7B\" --cka-device $ckaDevice --device1 $device1 --device2 $device2" Enter
tmux send-keys -t $sessionName "uv run scripts/cka_ablations.py --descendant \"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B\" --cka-device $ckaDevice --device1 $device1 --device2 $device2" Enter
tmux send-keys -t $sessionName "uv run scripts/cka_ablations.py --descendant \"Nickyang/ConciseR-Zero-7B\" --cka-device $ckaDevice --device1 $device1 --device2 $device2" Enter
