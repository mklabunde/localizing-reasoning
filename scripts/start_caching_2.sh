#!/bin/bash

# Define session name
SESSION="cache-reps2"

# Define models and corresponding devices
models=("acereason1-1" "openr1" "qwen2.5-math-7b" "qwen2.5-math-7b-rope300k" "qwen-r1-distill")
devices=("cuda:1" "cuda:4" "cuda:5" "cuda:6" "cuda:0")

cache_dir="cache_representations_2"

# Start the tmux session in detached mode
tmux new-session -d -s $SESSION

# Loop through models and devices
for i in "${!models[@]}"; do
    model=${models[$i]}
    device=${devices[$i]}

    # Create a new window and run the command
    if [ "$i" -eq 0 ]; then
        # First window, use the session start window
        tmux rename-window -t $SESSION:0 "$model"
        tmux send-keys -t $SESSION:0 "conda deactivate" C-m
        tmux send-keys -t $SESSION:0 "uv run scripts/cache_representations.py model=$model device=\"$device\" cache_dir=$cache_dir" C-m
    else
        # Subsequent windows
        tmux new-window -t $SESSION -n "$model"
        tmux send-keys -t $SESSION:"$i" "conda deactivate" C-m
        tmux send-keys -t $SESSION:"$i" "uv run scripts/cache_representations.py model=$model device=\"$device\" cache_dir=$cache_dir" C-m
    fi

    # Wait for 1 second before starting the next one
    sleep 1
done

# Attach to the session
tmux attach -t $SESSION
