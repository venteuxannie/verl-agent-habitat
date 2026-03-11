from agent_system.environments.env_package.habitat_sim.utils.third_party import call_generate_task_description

if __name__ == "__main__":
    response = call_generate_task_description("a red cube", "segment")
    print(response)
