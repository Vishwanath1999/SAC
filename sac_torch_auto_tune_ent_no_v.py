import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from sac_networks_2 import ActorNetwork, CriticNetwork, ValueNetwork

class SAC_Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            env_id, gamma=0.99, 
            n_actions=2, max_size=1000000, layer1_size=256,
            layer2_size=256, batch_size=100, reward_scale=2,omega=1e-2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name=env_id+'_actor', 
                                  max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_2')
       
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_target_critic_1')

        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_target_critic_2')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.target_entropy_coef = -np.prod(env.action_space.shape)
        self.log_ent_coef = T.log(T.ones(1,device=self.actor.device)).requires_grad_(True)
        self.ent_coef_optim = T.optim.Adam([self.log_ent_coef],lr=omega)


    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        #actions, _ = self.actor.sample_mvnormal(state)
        # actions is an array of arrays due to the added dimension in state
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
       
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        actions_, log_probs_ = self.actor.sample_normal(state_, reparameterize=False)
        log_probs_ = log_probs_.view(-1)

        ent_coef = T.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef*(log_probs+self.target_ent_coef).detach()).mean()

        q1_new_policy = self.target_critic_1.forward(state_, actions_)
        q2_new_policy = self.target_critic_2.forward(state_, actions_)
        q1_new_policy[done] = 0.0
        q1_new_policy[done] = 0.0
        critic_value_ = T.min(q1_new_policy, q2_new_policy)
        critic_value_ = critic_value_.view(-1)

        # Critic Optimization
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*(critic_value_ - ent_coef*log_probs_)
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        cl = critic_loss.item()
        critic_loss.backward()
        # T.nn.utils.clip_grad_value_(self.critic_1.parameters(), 1)
        # T.nn.utils.clip_grad_value_(self.critic_2.parameters(), 1)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        #Actor Optimization
        actor_loss = ent_coef*log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        # T.nn.utils.clip_grad_value_(self.actor.parameters(), 1)
        self.actor.optimizer.step()        
        al = actor_loss.item()    
             
        # Entropy Optimization
        self.ent_coef_optim.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optim.step()

        self.update_network_parameters()
        return al,cl,ent_coef_loss.item(),ent_coef.item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_critic_1_params = self.target_critic_1.named_parameters()
        critic_1_params = self.critic_1.named_parameters()

        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_1_state_dict = dict(target_critic_1_params)
        critic_1_state_dict = dict(critic_1_params)

        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_2_state_dict = dict(critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()