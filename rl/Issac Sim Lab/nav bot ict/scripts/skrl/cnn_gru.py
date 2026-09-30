import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                clip_actions=False, clip_log_std=True,
                min_log_std=-20.0, max_log_std=2.0, reduction="sum",
                num_envs=1, num_layers=1, hidden_size=256, sequence_length=32):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role="policy",
        )
        DeterministicMixin.__init__(
            self,
            clip_actions=clip_actions,
            role="value",
        )

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.lidar_dim = 360
        self.proprio_dim = self.num_observations - self.lidar_dim  # 369 - 360 = 9

        # --- CNN: spatial feature extractor over 360 LiDAR beams ---
        self.cnn_container = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=[1, 5], stride=[1, 2]),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=[1, 3], stride=[1, 2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(out_features=256),
            nn.ReLU(),
        )

        # --- GRU: real recurrence over CNN features ---
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # --- Proprioception MLP ---
        self.proprio_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
        )

        # --- Fusion MLP ---
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.ELU(),
            nn.LazyLinear(out_features=64),
            nn.ELU(),
        )

        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), 0.0), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

        self._shared_output = None

    # -------------------------------------------------------------
    # Declares the recurrent state shape to skrl/PPO_RNN
    # -------------------------------------------------------------
    def get_specification(self):
        # GRU has a single hidden state (unlike LSTM which has hidden+cell)
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)],
            }
        }
        
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)

    def compute(self, inputs, role=""):
        states = inputs.get("states") or inputs.get("observations")
        terminated = inputs.get("terminated", None)
        rnn_input_raw = inputs.get("rnn", [None])
        hidden_states = rnn_input_raw[0] if rnn_input_raw else None

        lidar  = states[:, :self.lidar_dim]
        proprio = states[:, self.lidar_dim:]

        cnn_out = self.cnn_container(lidar.view(-1, 1, 1, self.lidar_dim))  # [N, 256]

        if self.training:
            batch_total = cnn_out.shape[0]
            n_sequences = batch_total // self.sequence_length

            rnn_input = cnn_out.view(n_sequences, self.sequence_length, 256)

            # Extract initial hidden state per sequence from stored rollout hidden states
            if hidden_states is not None:
                hs = hidden_states.view(
                    self.num_layers, n_sequences, self.sequence_length, self.hidden_size
                )
                hs = hs[:, :, 0, :].contiguous()   # [num_layers, n_sequences, hidden]
            else:
                hs = torch.zeros(
                    self.num_layers, n_sequences, self.hidden_size,
                    device=cnn_out.device
                )

            if terminated is not None and torch.any(terminated):
                rnn_outputs   = []
                terminated_sq = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated_sq[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    s, e = indexes[i], indexes[i + 1]
                    out, hs = self.gru(rnn_input[:, s:e, :], hs)
                    rnn_outputs.append(out)
                    if e < self.sequence_length:
                        hs = hs * (~terminated_sq[:, e - 1]).view(1, -1, 1).float()
                rnn_out = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_out, hs = self.gru(rnn_input, hs)

            rnn_out = rnn_out.reshape(-1, self.hidden_size)  # [N*L, hidden]
            proprio_in = proprio

        else:
            # Rollout — single step per env
            rnn_input = cnn_out.unsqueeze(1)   # [N, 1, 256]

            if hidden_states is not None:
                hs = hidden_states
                if terminated is not None:
                    hs = hs * (~terminated).view(1, -1, 1).float()
            else:
                hs = torch.zeros(
                    self.num_layers, cnn_out.shape[0], self.hidden_size,
                    device=cnn_out.device
                )

            rnn_out, hs = self.gru(rnn_input, hs)
            rnn_out = rnn_out.squeeze(1)   # [N, hidden]
            proprio_in = proprio

        proprio_out = self.proprio_container(proprio_in)
        net_in  = torch.cat([rnn_out, proprio_out], dim=-1)
        net_out = self.net_container(net_in)

        if role == "policy":
            self._shared_output = net_out
            return self.policy_layer(net_out), {"log_std": self.log_std_parameter, "rnn": [hs]}
        elif role == "value":
            net_out_v = self._shared_output if self._shared_output is not None else net_out
            self._shared_output = None
            return self.value_layer(net_out_v), {"rnn": [hs]}