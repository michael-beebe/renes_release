import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SAC:
    def __init__(
        self,
        actor_critic,
        gamma,
        tau,
        alpha,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        automatic_entropy_tuning=True,
    ):
        self.actor_critic = actor_critic

        self.gamma = gamma
        self.tau = tau

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True).to(actor_critic.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_critic.actor_parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic_parameters(), lr=critic_lr)

        # Target networks
        self.actor_critic_target = copy.deepcopy(actor_critic)

    def update(self, replay_buffer, batch_size):
        # Sample a batch from replay buffer
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = replay_buffer.sample(batch_size)

        # Move tensors to the correct device
        state_batch = state_batch.to(self.actor_critic.device)
        action_batch = action_batch.to(self.actor_critic.device)
        reward_batch = reward_batch.to(self.actor_critic.device)
        next_state_batch = next_state_batch.to(self.actor_critic.device)
        done_batch = done_batch.to(self.actor_critic.device)

        # Compute target Q-values
        with torch.no_grad():
            next_action_dist = self.actor_critic.get_action_distribution(next_state_batch)
            next_action = next_action_dist.rsample()
            next_log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)

            q1_next_target, q2_next_target = self.actor_critic_target.evaluate_critics(
                next_state_batch, next_action
            )
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob

            target_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target

        # Compute current Q-values
        q1_current, q2_current = self.actor_critic.evaluate_critics(state_batch, action_batch)
        critic_loss = F.mse_loss(q1_current, target_q_value) + F.mse_loss(q2_current, target_q_value)

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        action_dist = self.actor_critic.get_action_distribution(state_batch)
        action = action_dist.rsample()
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        q1_pi, q2_pi = self.actor_critic.evaluate_critics(state_batch, action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature parameter for entropy regularization)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.).to(self.actor_critic.device)

        # Soft update target networks
        self.soft_update(self.actor_critic_target, self.actor_critic)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

