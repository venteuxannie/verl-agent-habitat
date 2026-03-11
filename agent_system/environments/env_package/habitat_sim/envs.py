import gymnasium as gym
import ray
import numpy as np
from .utils.habitat_envs import CreateHabitatEnv
from .utils.constants import TEMP_DIR

# @ray.remote(num_cpus=0.2, num_gpus=0.05)
# @ray.remote(num_cpus=0.2, num_gpus=0.0138)
@ray.remote(num_cpus=0.2, num_gpus=0.0125)    # 80 replica
class HabitatWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds an independent instance of the specified gym environment.
    """
    
    def __init__(self, seed, dataset_name, alpha_conf=0.5):
        """Initialize the gym environment in this worker"""
        self.env = CreateHabitatEnv(seed, dataset_name, alpha_conf=alpha_conf)

    def step(self, pred_task_type, pred_task_prompt, action, is_valid):
        """
        Execute a step in the environment.
        
        Args:
            pred_task_type: The predicted task type from the model
            pred_task_prompt: The predicted task prompt (target object description) from the model
            action: The action index to execute
            is_valid: Whether the model output was valid
        
        Returns:
            obs: The observation (PIL image)
            reward: The reward
            done: Whether the episode is done
            info: Additional information
        """
        obs, reward, done, info = self.env.step(pred_task_type, pred_task_prompt, action, is_valid)
        return obs, reward, done, info
    
    def reset(self, seed_for_reset=0, is_unique=True, sync_info=None):
        """Reset the environment with optional seed"""
        obs, info = self.env.reset(seed=seed_for_reset, is_unique=is_unique, sync_info=sync_info)
        return obs, info

    def get_task_type(self):
        return self.env.get_task_type()


class HabitatEnvs(gym.Env):
    """
    Ray-based parallel environment wrapper for gym cards environments.
    - env_id: Gym environment ID
    - env_num: Number of distinct environments
    - group_n: Number of replicas within each group (commonly used for multiple copies with the same seed)
    - env_kwargs: Parameters needed to create a single gym.make(env_id)
    """

    def __init__(self,
                 dataset_name,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 is_train=True,
                 alpha_conf=0.5):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(_temp_dir=TEMP_DIR,)

        self.dataset_name = dataset_name
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.alpha_conf = alpha_conf

        np.random.seed(seed)

        # Create Ray remote actors instead of processes
        self.workers = []
        for _ in range(self.num_processes):
            worker = HabitatWorker.remote(seed, self.dataset_name, self.alpha_conf)
            self.workers.append(worker)

    def step(self, pred_task_types, pred_task_prompts, actions, valids):
        """
        Perform step in parallel.
        
        Args:
            pred_task_types: list of predicted task types, length must equal self.num_processes
            pred_task_prompts: list of predicted task prompts, length must equal self.num_processes
            actions: list or numpy array of action indices, length must equal self.num_processes
            valids: list of validity flags, length must equal self.num_processes
        
        Returns:
            obs_list: list of observations
            reward_list: list of rewards
            done_list: list of done flags
            info_list: list of info dicts
        """
        assert len(actions) == self.num_processes
        assert len(pred_task_types) == self.num_processes
        assert len(pred_task_prompts) == self.num_processes
        assert len(valids) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, pred_task_type, pred_task_prompt, action, is_valid in zip(
            self.workers, pred_task_types, pred_task_prompts, actions, valids
        ):
            future = worker.step.remote(pred_task_type, pred_task_prompt, action, is_valid)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        Different seeds will be assigned to each environment (or the same seed within a group).
        :return: (obs_list, info_list)
        """
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # Repeat seed for environments in the same group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        init_sync_info = {}
        for i, worker in enumerate(self.workers):
            if i % self.group_n == 0:
                future = worker.reset.remote(seeds[i], is_unique=True, sync_info=None)
                _, init_sync_info = ray.get(future)
                futures.append(future)
            else:
                future = worker.reset.remote(seeds[i], is_unique=False, sync_info=init_sync_info)
                futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, info_list

    # @property
    # def get_admissible_commands(self):
    #     """
    #     Simply return the prev_admissible_commands stored by the main process.
    #     You could also design it to fetch after each step or another method.
    #     """
    #     return self.prev_admissible_commands
    
    def gather_task_types(self):
        futures = []
        for worker in self.workers:
            future = worker.get_task_type.remote()
            futures.append(future)
        return ray.get(futures)

    def close(self):
        """
        Close all Ray actors.
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    # def __del__(self):
    #     self.close()


def build_habitat_envs(dataset_name,
                        seed,
                        env_num,
                        group_n,
                        is_train=True,
                        alpha_conf=0.5):
    """
    Externally exposed constructor function to create parallel Gym environments.
    - env_name: [gym_cards/Blackjack-v0, gym_cards/NumberLine-v0, gym_cards/EZPoints-v0, gym_cards/Points24-v0]
    - seed: For reproducible randomness
    - env_num: Number of distinct environments
    - group_n: Number of environment replicas under the same seed
    - is_train: Determines the seed range used (train/test)
    """
    return HabitatEnvs(
        dataset_name=dataset_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        alpha_conf=alpha_conf,
    )
