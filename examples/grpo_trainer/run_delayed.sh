#!/bin/bash

# This script will execute run_internvl_mix_conf.sh after 3 hours

echo "Script scheduled to run in 9 hours..."
echo "Start time: $(date)"
echo "Expected execution time: $(date -d '+9 hours')"

# Wait for 8 hours
sleep 9h

echo "Starting execution at: $(date)"

# Change to the script directory
cd /data/tct/ActivePerception

# Execute the target script
bash ./examples/grpo_trainer/run_internvl_mix.sh

echo "Execution completed at: $(date)"

