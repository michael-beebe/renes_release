import torch
import torch.nn as nn
import torch_geometric as pyg

from game2graph.pyg_graph_models import GCN, GraphAttentionPooling


class ResponseGraphEncoder(torch.nn.Module):
    def __init__(
        self,
        node_feature_dim,
        node_output_size,
        batch_norm=True,
        # one_hot_degree,
        num_layers=10,
    ):
        super(ResponseGraphEncoder, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_output_size = node_output_size
        # self.one_hot_degree = one_hot_degree
        self.batch_norm = batch_norm
        self.num_layers = num_layers

        self.gcn = GCN(
            self.node_feature_dim,
            self.node_output_size,
            num_layers=self.num_layers,
            batch_norm=self.batch_norm,
        )
        self.att = GraphAttentionPooling(self.node_output_size)

    def forward(self, data):
        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr
        graph_batch = pyg.data.Batch.from_data_list(data)
        out = self.gcn(graph_batch)
        out = self.att(out, graph_batch.batch)
        return out


class ActorNet(torch.nn.Module):
    def __init__(self, state_feature_size, action_size, batch_norm=True):
        super(ActorNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm
        self.action_size = action_size
        self.hidden_size = 64
        self.actor_mlp = nn.Sequential(
            nn.Linear(
                in_features=self.state_feature_size,
                out_features=self.hidden_size,
                bias=True,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_size, out_features=self.hidden_size, bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_size, out_features=action_size, bias=True
            ),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, state_feature):
        out = self.actor_mlp(state_feature)

        return out


class CriticNet(torch.nn.Module):
    def __init__(
        self,
        state_feature_size,
        batch_norm,
    ):
        super(CriticNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm
        self.hidden_size = 256

        self.critic_mlp = nn.Sequential(
            nn.Linear(
                in_features=self.state_feature_size,
                out_features=self.hidden_size,
                bias=True,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_size, out_features=self.hidden_size, bias=True
            ),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=1, bias=True),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, state_feat):
        return self._eval(state_feat)

    def _eval(self, state_feat):
        state_value = self.critic_mlp(state_feat)

        return state_value
