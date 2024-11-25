import random
import sys
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box, Text
from llamagym import evaluate_state
from datasets import load_dataset

def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)

class QAEnv(gym.Env):
    def __init__(self):
        super(QAEnv, self).__init__()
        self.observation_space = Dict({
            'question': Text(max_length=500),
            'context': Text(max_length=500)
        })
        self.dataset = load_dataset('danikhan632/OpenMystery')
        self.current_episode_rewards = []
        self.current_episode_messages = []

    def reset(self):
        choice = random.randint(0, len(self.dataset['train']) - 1)
        question, answer = self.dataset['train'][choice]['start_state'],  self.dataset['train'][choice]['solution']
        self.goal_state = answer
        return question, {}

    def step(self, action):

        task = self.goal_state
        
        agent_state = f"Reasoning:\n {action[1]['content']}\nAssistant:\n {action[2]['content']}"
        
        data = evaluate_state(task, agent_state, self.goal_state)
        printc(data,'yellow')
        
        reward = data['reward']
        info = data['feedback']
        
        print(f"Reward: {reward}")
        printc(info, "green")


        return self.observation_space.sample(), reward, True,"", info

