"""SAC networks.

Changes vs. the original:
  * Actor.mean_action(): differentiable tanh(mean) action. The CMC actor anchor
    must use this. Actor.deterministic() is still @torch.no_grad() and is only
    meant for acting / data collection.               (FIX 1)
  * SACAgent.reset_alpha_and_optimizers(): optional per-task reset of the
    entropy temperature and Adam states (plasticity ablation; off by default).
  * Optional LayerNorm in the critics (layer_norm=True); the flag is saved in
    checkpoints so loading always rebuilds the same architecture.
"""

import copy
import math

import torch
import torch.nn as nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256), out_dim=256, layer_norm=False):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)]
            if layer_norm:
                layers += [nn.LayerNorm(h)]
            layers += [nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high):
        super().__init__()
        self.backbone = MLP(obs_dim, (256, 256), 256)
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.register_buffer("action_scale",
                             torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def distribution(self, obs):
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return torch.distributions.Normal(mean, log_std.exp())

    def forward(self, obs):
        return self.distribution(obs)

    def sample(self, obs):
        dist = self.distribution(obs)
        z = dist.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias
        logp = dist.log_prob(z) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        return action, logp.sum(-1, keepdim=True)

    def mean_action(self, obs):
        """Differentiable deterministic action. Use this inside any loss."""
        h = self.backbone(obs)
        return torch.tanh(self.mean(h)) * self.action_scale + self.action_bias

    @torch.no_grad()
    def deterministic(self, obs):
        """Gradient-free deterministic action. For acting/evaluation ONLY."""
        return self.mean_action(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, layer_norm=False):
        super().__init__()
        # LayerNorm keeps critic activations bounded; it is a standard fix for
        # Q-value blow-up in off-policy RL (only the critics use it).
        self.net = MLP(obs_dim + act_dim, (256, 256), 1, layer_norm=layer_norm)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class SACAgent:
    def __init__(self, obs_dim, act_dim, low, high, device,
                 lr=3e-4, init_alpha=0.2, layer_norm=False):
        self.device = device
        self.lr = lr
        self.init_alpha = init_alpha
        self.layer_norm = bool(layer_norm)
        self.actor = Actor(obs_dim, act_dim, low, high).to(device)
        self.q1 = Critic(obs_dim, act_dim, self.layer_norm).to(device)
        self.q2 = Critic(obs_dim, act_dim, self.layer_norm).to(device)
        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device=device)
        self.target_entropy = -float(act_dim)
        self._build_optimizers()

    def _build_optimizers(self):
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def reset_alpha_and_optimizers(self):
        """Reset entropy temperature and all Adam moments (task-boundary ablation)."""
        with torch.no_grad():
            self.log_alpha.fill_(math.log(self.init_alpha))
        self._build_optimizers()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, deterministic=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor.deterministic(x) if deterministic else self.actor.sample(x)[0]
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def soft_update(self, tau=0.005):
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "layer_norm": self.layer_norm,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if bool(ckpt.get("layer_norm", False)) != self.layer_norm:
            raise ValueError(f"{path} was saved with layer_norm={ckpt.get('layer_norm', False)}, "
                             f"but this agent has layer_norm={self.layer_norm}")
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_()
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)
