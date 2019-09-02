import gym
import torch
import time
import os
import ray
import numpy as np
import copy
from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt
# %matplotlib inline
from memory import ReplayBuffer
from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv
ray.shutdown()
# ray.init()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=100000000, object_store_memory=1000000000)

FloatTensor = torch.FloatTensor
def plot_result(total_rewards ,learning_num, legend,cw,ew):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
#     plt.show()
    name=str(cw)+'_'+str(ew)+'results.png'
    plt.savefig(name)
#Full
hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
class DQN_agent(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT)):
        
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
#         memory: Store and sample experience replay.
#         self.memory = ReplayBuffer(hyper_params['memory_size'])
        
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        
    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)
        
    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    # save model
    def save_model(self):
        self.eval_model.save(result_floder + '/best_model.pt')
        
    # load model
    def load_model(self):
        self.eval_model.load(result_floder + '/best_model.pt')
        
@ray.remote    
class RLAgent_model_server(DQN_agent):
    def __init__(self, env,hyper_params,training_episodes, test_interval):
        super().__init__(env, hyper_params)
        self.steps=0
#         self.memory_server=memory_server
        self.training_episodes=training_episodes
        self.test_interval=test_interval
        self.episodes=0
        self.collector_done = False
        self.evaluator_done = False
        self.privous_eval_model = []
        self.reuslt_count=0
        self.upti=0
        self.results=[0]*(training_episodes//test_interval)
        self.memory_server=ReplayBuffer_remote.remote(hyper_params['memory_size'])
    def update_batch(self):
        if self.steps % self.update_steps != 0:
            return
        #Randomly sample a mini-batch of 𝐵 transition {𝑠,𝑎,𝑟,𝑠′ }from𝐷
        batch = ray.get(self.memory_server.sample.remote(self.batch_size))
        if not batch:
            return
        (states, actions, reward, next_states,
         is_terminal) = batch
        
        states = states
        next_states = next_states
        terminal = FloatTensor([0 if t else 1 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
            
        #INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
        
        q_target = reward + self.beta * torch.max(q_next,1)[0]*terminal
        
        # update model
        self.eval_model.fit(q_values, q_target)
    def learn(self,done,update):
        if update:
            self.steps+=10
            self.update_batch()
        if self.steps%self.model_replace_freq==0:
            self.target_model.replace(self.eval_model)
        if self.collector_done:
            return self.collector_done
        if done:
            self.episodes+=1
#             print(self.episodes)
        if (self.episodes//self.test_interval)>len(self.privous_eval_model):
            id = ray.put(self.eval_model)
            self.privous_eval_model.append(id)
        if self.episodes>=self.training_episodes:
            self.collector_done=True
        
        return self.collector_done
    #evaluate
    def add_result(self, result, num):
        self.results[num] = result
        self.privous_eval_model[num]=None
    def get_reuslts(self):
        return self.results, self.eval_model
    def ask_evaluation(self):
        if len(self.privous_eval_model) > self.reuslt_count:
            num = self.reuslt_count
            eval_model_id = self.privous_eval_model[num]
            self.reuslt_count += 1
            return eval_model_id, False, num
        else:
            if self.episodes >= self.training_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None
    def getmem(self):
        return self.memory_server, self.max_episode_steps
@ray.remote    
def collecting_worker(env,model_server,memory_server,test_interval,max_episode_steps,update_steps):
    
    learn_done=False
    while True:        
        if learn_done:
            print(learn_done)
            break
        state = env.reset()
        done = False
        steps=0
        while steps < max_episode_steps and not done:
            action=ray.get(model_server.explore_or_exploit_policy.remote(state))
            state_, reward, done, _ = env.step(action)
            memory_server.add.remote(state, action, reward, state_, done)
            state=state_
            steps+=1
            if steps==max_episode_steps:
                done=True
            if steps%update_steps==0 and not done:
                learn_done=ray.get(model_server.learn.remote(done,True))
        learn_done=ray.get(model_server.learn.remote(done,steps%update_steps==0))

@ray.remote
def evaluation_worker(env, model_server,max_episode_steps, trials = 10):
    def greedy_policy(state):
        return eval_model.predict(state)
    
    while True:
        eval_model_id, evaluator_done, num = ray.get(model_server.ask_evaluation.remote())
        if num!=None or evaluator_done==True:
            print(evaluator_done,num)
        if evaluator_done:
            break
        if num==None:
            time.sleep(int(uniform(5,20)))
            continue 
        eval_model=ray.get(eval_model_id)
        total_reward = 0
        for _ in range(trials):
            state = env.reset()
            evaluator_done = False
            steps = 0
            done=False
            while steps < max_episode_steps and not done:
                steps += 1
                action=greedy_policy(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        print('~~~~~~~~~~~',avg_reward,'~~~~~~~~~~~',num+1)

        ray.get(model_server.add_result.remote(avg_reward, num))

class distributed_RL_agent():
    def __init__(self, env, hyper_params,cw_num,ew_num,training_episodes, test_interval):
        
        
        self.model_server = RLAgent_model_server.remote(env, hyper_params,training_episodes, test_interval)
        self.memory_server,self.max_episode_steps = ray.get(self.model_server.getmem.remote())
        self.env = env
        self.workers_id = []
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.agent_name = "CartPole_distributed"
        self.training_episodes=training_episodes
        self.test_interval=test_interval
        self.hyper_params=hyper_params
        self.update_steps=hyper_params['update_steps']
    def learn_and_evaluate(self):
        workers_id = []
        
        #learn
        for _ in range(self.cw_num):
            w_id = collecting_worker.remote(self.env,self.model_server,self.memory_server,test_interval,self.max_episode_steps,self.update_steps)
            workers_id.append(w_id)

        #evaluate
        for _ in range(self.ew_num):
            w_id = evaluation_worker.remote(self.env,self.model_server,self.max_episode_steps)
            workers_id.append(w_id)

        ray.wait(workers_id, len(workers_id))
        return ray.get(self.model_server.get_reuslts.remote())
    
#######main######
# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'
env=CartPoleEnv()
env.reset()
cw_num = 4
ew_num = 4
training_episodes, test_interval = 7000, 50
start = time.time()
agent = distributed_RL_agent(env, hyperparams_CartPole,cw_num,ew_num,training_episodes, test_interval)
result = copy.deepcopy(agent.learn_and_evaluate())
print('time: ', time.time() - start)
plot_result(result[0], test_interval, ["batch_update with target_model"],cw_num,ew_num)
