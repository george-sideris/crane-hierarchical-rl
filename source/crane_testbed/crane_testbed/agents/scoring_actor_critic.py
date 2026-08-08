"""Categorical-over-points Actor-Critic for the scoring grasp head (P3 = BC->RL (scoring)).

RSL-RL-compatible AC whose ACTION DISTRIBUTION is
    pi(a | cloud) = Categorical(which point; logits = per-point scores)
                    x Normal(dz around the chosen point's dz-head prediction)
with yaw sampled in the (cos 2psi, sin 2psi) EMBEDDING around the yaw head's prediction -
the same trick the 5D Gaussian line uses: the Gaussian lives on the unconstrained pair, the
circle only appears in the atan2 decode, so there is no angular-wrap problem.

Action tensor layout (B, 4): [point_index (as float), dz, cos2yaw, sin2yaw]. The env decodes
the grasp as points[idx] + (0, 0, dz), yaw = 0.5*atan2(s, c).

Why categorical instead of Gaussian-on-coordinates: every explored action is an OBSERVED point,
so exploration cannot propose empty space (Gaussian BCRL measurably worsened gap-aiming, 48.5%
vs BC's 33.9%); a zero-log outcome pushes down exactly the chosen point's score - the online
form of outcome-weighted hard negatives; and the reachability (candidate) mask is exact logit
masking rather than distribution-distorting clipping.

Module attribute names (encoder/head/score_out/dz_out/yaw_out) deliberately MATCH the
ScoringGraspPolicy checkpoint so `--bc_checkpoint <P2c scoring_policy.pt>` loads with
strict=False, leaving only the critic and sigma freshly initialized.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# training-coords action box: candidates must lie inside (margin points are context only)
_BOX_MIN = (-5.364, -1.684, -1.30)
_BOX_MAX = (-3.364, 5.316, 0.10)


class ScoringActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs=None, num_critic_obs=None, num_actions: int = 4,
                 num_points: int = 2048, latent_dim: int = 256,
                 init_noise_std: float = 0.05, temperature: float = 1.0,
                 critic_hidden_dims=(128, 64), **kwargs):
        super().__init__()
        self.num_points = num_points
        self.temperature = float(temperature)

        # ---- trunk + heads: EXACT mirror of scoring_head.ScoringGraspPolicy (key-compatible)
        self.encoder = nn.Module()
        self.encoder.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ELU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ELU())
        self.encoder.fc = nn.Sequential(
            nn.Linear(256, latent_dim), nn.BatchNorm1d(latent_dim), nn.ELU())
        in_dim = 256 + latent_dim + 3
        self.head = nn.Sequential(nn.Linear(in_dim, 128), nn.ELU(),
                                  nn.Linear(128, 64), nn.ELU())
        self.score_out = nn.Linear(64, 1)
        self.dz_out = nn.Linear(64, 1)
        self.yaw_out = nn.Linear(64, 2)

        # ---- ASYMMETRIC critic: privileged state vector (height-sorted log poses), fully
        # separate from the actor. Rationale (proven on the OG BCRL): a privileged critic
        # learns value fast -> low-variance advantages from the start of a short fine-tune,
        # and no value-loss gradient ever touches the encoder features the BC actor relies on.
        self._critic_obs_dim = int(num_critic_obs) if num_critic_obs else 256
        layers, d = [], self._critic_obs_dim
        for h in critic_hidden_dims:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.critic = nn.Sequential(*layers)

        # exploration scales (learned, PPO-standard): dz in metres, yaw pair in embedding units
        self.log_sigma_dz = nn.Parameter(torch.log(torch.tensor(float(init_noise_std))))
        self.log_sigma_yaw = nn.Parameter(torch.log(torch.tensor(float(init_noise_std))))

        self._cached = None      # (logits, dz_mu_all, yaw_all, pts) from update_distribution
        print(f"[ScoringAC] categorical-over-{num_points}-points x Normal(dz), yaw deterministic")

    # ---------------- trunk ----------------
    def _trunk(self, obs: torch.Tensor):
        b = obs.shape[0]
        pts = obs.view(b, self.num_points, 3)
        f = self.encoder.mlp1(pts.reshape(b * self.num_points, 3)).view(b, self.num_points, -1)
        glob = self.encoder.fc(f.max(dim=1).values)                      # (B, latent)
        h = self.head(torch.cat([f, glob.unsqueeze(1).expand(-1, self.num_points, -1), pts], -1))
        score = self.score_out(h).squeeze(-1)
        pad = (pts.abs().sum(-1) < 1e-6)
        cmin = pts.new_tensor(_BOX_MIN)
        cmax = pts.new_tensor(_BOX_MAX)
        outbox = ~((pts >= cmin) & (pts <= cmax)).all(-1)
        score = score.masked_fill(pad | outbox, -1e9)                    # pads + context unpickable
        return score, self.dz_out(h).squeeze(-1), self.yaw_out(h), pts, glob

    # ---------------- rsl-rl interface ----------------
    def reset(self, dones=None):
        pass

    def update_normalization(self, obs=None, critic_obs=None):
        pass

    def update_distribution(self, observations):
        score, dz_mu, yaw, pts, glob = self._trunk(observations)
        self._cached = (score / self.temperature, dz_mu, yaw, pts, glob)

    def act(self, observations, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        logits, dz_mu, yaw, pts, _ = self._cached
        idx = torch.distributions.Categorical(logits=logits).sample()            # (B,)
        b = torch.arange(len(idx), device=idx.device)
        dz = Normal(dz_mu[b, idx], self.log_sigma_dz.exp()).sample()
        y2 = Normal(yaw[b, idx], self.log_sigma_yaw.exp()).sample()              # (B, 2)
        return torch.cat([idx.float().unsqueeze(-1), dz.unsqueeze(-1), y2], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        logits, dz_mu, _, _, _ = self._cached
        idx = actions[:, 0].long().clamp(0, self.num_points - 1)
        b = torch.arange(len(idx), device=idx.device)
        lp_idx = F.log_softmax(logits, dim=-1)[b, idx]
        lp_dz = Normal(dz_mu[b, idx], self.log_sigma_dz.exp()).log_prob(actions[:, 1])
        yaw_mu = self._cached[2][b, idx]
        lp_yaw = Normal(yaw_mu, self.log_sigma_yaw.exp()).log_prob(actions[:, 2:4]).sum(-1)
        return lp_idx + lp_dz + lp_yaw

    def act_inference(self, observations) -> torch.Tensor:
        score, dz_mu, yaw, pts, _ = self._trunk(observations)
        idx = score.argmax(dim=-1)
        b = torch.arange(len(idx), device=idx.device)
        return torch.cat([idx.float().unsqueeze(-1), dz_mu[b, idx].unsqueeze(-1),
                          yaw[b, idx]], dim=-1)

    def evaluate(self, critic_observations, **kwargs) -> torch.Tensor:
        # asymmetric: critic_observations is the privileged state vector, not the cloud
        return self.critic(critic_observations)

    # logging-only properties (approximate where the hybrid has no single natural value)
    @property
    def action_mean(self):
        logits, dz_mu, _, _, _ = self._cached
        idx = logits.argmax(dim=-1)
        b = torch.arange(len(idx), device=idx.device)
        return torch.cat([idx.float().unsqueeze(-1), dz_mu[b, idx].unsqueeze(-1),
                          self._cached[2][b, idx]], dim=-1)

    @property
    def action_std(self):
        logits, _, _, _, _ = self._cached
        s = self.log_sigma_dz.exp().expand(logits.shape[0])
        sy = self.log_sigma_yaw.exp().expand(logits.shape[0])
        return torch.stack([torch.zeros_like(s), s, sy, sy], -1)

    @property
    def entropy(self):
        logits, _, _, _, _ = self._cached
        cat = torch.distributions.Categorical(logits=logits).entropy()
        gauss = 0.5 * (1.0 + math.log(2 * math.pi)) + self.log_sigma_dz
        gauss_yaw = 2 * (0.5 * (1.0 + math.log(2 * math.pi)) + self.log_sigma_yaw)
        return cat + gauss + gauss_yaw

    def get_param_groups(self, base_lr: float) -> list:
        enc = [p for n, p in self.named_parameters() if n.startswith("encoder")]
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder")]
        return [{"params": enc, "lr": base_lr * 0.1}, {"params": rest, "lr": base_lr}]
