import torch
from torch.autograd import grad

class TRPO:
    def __init__(
        self,
        actor_critic,
        max_kl,
        damping_coeff,
        value_loss_coef,
        entropy_coef,
        cg_iters=10,
        backtrack_iters=10,
        backtrack_coeff=0.8,
    ):
        self.actor_critic = actor_critic
        self.max_kl = max_kl
        self.damping_coeff = damping_coeff
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff

        self.value_optimizer = torch.optim.Adam(
            self.actor_critic.get_critic_parameters(), lr=3e-4
        )

    def conjugate_gradient(self, fisher_vector_product, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for _ in range(nsteps):
            Ap = fisher_vector_product(p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def fisher_vector_product(self, kl, vector):
        actor_params = self.actor_critic.get_actor_parameters()
        grads = grad(kl, actor_params, create_graph=True, retain_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        gvp = grad(torch.dot(flat_grads, vector), actor_params, retain_graph=True)
        gvp_flat = torch.cat([g.contiguous().view(-1) for g in gvp if g is not None])
        return gvp_flat + self.damping_coeff * vector

    def _apply_update(self, params, step):
        """Helper function to update parameters."""
        offset = 0
        for param in params:
            numel = param.numel()
            param.data += step[offset : offset + numel].view_as(param.data)
            offset += numel

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        obs = rollouts.obs[:-1].view(-1, *obs_shape)
        actions = rollouts.actions.view(-1, action_shape)
        returns = rollouts.returns[:-1].view(-1, 1)
        masks = rollouts.masks[:-1].view(-1, 1)

        # Compute value loss
        values = self.actor_critic.get_value(obs, None, masks)
        advantages = returns - values
        value_loss = advantages.pow(2).mean()

        # Optimize value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Cache old actor features
        with torch.no_grad():
            old_actor_features = self.actor_critic.base(obs, None, masks)[1]

        # Compute policy loss and KL divergence
        values, action_log_probs, dist_entropy, kl = self.actor_critic.evaluate_actions_with_kl(
            obs,
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            masks,
            actions,
            old_actor_features=old_actor_features,
        )

        # Compute policy loss
        policy_loss = -(advantages.detach() * action_log_probs).mean()

        # Compute gradients of the policy loss with respect to actor parameters
        actor_params = self.actor_critic.get_actor_parameters()
        policy_grads = grad(policy_loss, actor_params, retain_graph=True)
        policy_grads_flat = torch.cat([g.view(-1) for g in policy_grads if g is not None])

        # Compute the natural gradient using conjugate gradient
        def fisher_vector_product_fn(v):
            return self.fisher_vector_product(kl, v)

        step_dir = self.conjugate_gradient(fisher_vector_product_fn, policy_grads_flat, nsteps=self.cg_iters)

        # Compute step size
        shs = 0.5 * torch.dot(step_dir, fisher_vector_product_fn(step_dir))
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = step_dir / lm

        # Apply the update to the actor parameters
        self._apply_update(actor_params, fullstep)

        return value_loss.item(), policy_loss.item(), dist_entropy.item()

