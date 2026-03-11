sleep 5s

echo "Starting execution at: $(date)"

# Change to the script directory
cd /data1/tct_data/verl-agent

# Execute the target script
python -m agent_system.environments.env_package.habitat_sim.scripts.test

echo "Execution completed at: $(date)"