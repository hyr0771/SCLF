import math

import torch
from torch import nn
from torch.nn import functional as F


class _GraphLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, glorot=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.glorot = glorot
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.glorot:
                bound = math.sqrt(6.0 / (self.out_channels + self.in_channels))
            else:
                bound = math.sqrt(6 / ((1 + math.sqrt(5) ** 2) * self.in_channels))
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                bound = 1.0 / math.sqrt(self.in_channels)
                self.bias.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def _aggregate(x, edge_index, num_targets, reduction="sum", weights=None):
    source, target = edge_index
    messages = x.index_select(0, source)
    if weights is not None:
        messages = weights.view(-1, 1) * messages
    if reduction == "mean":
        counts = x.new_zeros(num_targets)
        counts.scatter_add_(0, target, x.new_ones(target.numel()))
        counts = counts.clamp(min=1)
    index = target.view(-1, 1).expand_as(messages)
    output = x.new_zeros((num_targets, x.size(-1)))
    output.scatter_add_(0, index, messages)
    if reduction == "mean":
        output = output / counts.view(-1, 1)
    return output


class GCNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        add_self_loops=None,
        normalize=True,
        bias=True,
    ):
        super().__init__()
        if add_self_loops is None:
            add_self_loops = normalize
        if add_self_loops and not normalize:
            raise ValueError("Self-loops require graph normalization")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self.lin = _GraphLinear(in_channels, out_channels, bias=False, glorot=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.lin.reset_parameters()
            if self.bias is not None:
                self.bias.zero_()
            self._cached_edge_index = None

    def _normalize(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        if self.add_self_loops:
            source, target = edge_index
            keep = source != target
            nodes = torch.arange(num_nodes, device=edge_index.device)
            loops = nodes.view(1, -1).repeat(2, 1)
            if edge_weight is not None:
                fill = 2.0 if self.improved else 1.0
                loop_weights = edge_weight.new_full((num_nodes,), fill)
                loop_weights[source[~keep]] = edge_weight[~keep]
                edge_weight = torch.cat((edge_weight[keep], loop_weights), dim=0)
            edge_index = torch.cat((edge_index[:, keep], loops), dim=1)
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        source, target = edge_index
        degree = edge_weight.new_zeros(num_nodes)
        degree.scatter_add_(0, target, edge_weight)
        inverse = degree.pow_(-0.5)
        inverse.masked_fill_(inverse == float("inf"), 0)
        weights = inverse[source] * edge_weight * inverse[target]
        return edge_index, weights

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(x, (tuple, list)):
            raise ValueError("GCNConv requires one node feature tensor")
        if self.normalize:
            if self._cached_edge_index is None:
                edge_index, edge_weight = self._normalize(x, edge_index, edge_weight)
                if self.cached:
                    self._cached_edge_index = edge_index, edge_weight
            else:
                edge_index, edge_weight = self._cached_edge_index
        output = _aggregate(self.lin(x), edge_index, x.size(0), weights=edge_weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class SAGEConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        aggr="mean",
        normalize=False,
        root_weight=True,
        project=False,
        bias=True,
    ):
        super().__init__()
        if aggr not in ("mean", "sum", "add"):
            raise ValueError("SAGEConv supports mean or sum aggregation")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        widths = (
            (in_channels, in_channels) if isinstance(in_channels, int) else in_channels
        )
        if self.project:
            self.lin = _GraphLinear(widths[0], widths[0])
        self.lin_l = _GraphLinear(widths[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = _GraphLinear(widths[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        source, target = (x, x) if isinstance(x, torch.Tensor) else x
        if self.project:
            source = self.lin(source).relu()
        num_targets = source.size(0) if target is None else target.size(0)
        if size is not None and size[1] is not None:
            num_targets = size[1]
        output = self.lin_l(_aggregate(source, edge_index, num_targets, self.aggr))
        if self.root_weight and target is not None:
            output = output + self.lin_r(target)
        if self.normalize:
            output = F.normalize(output, p=2.0, dim=-1)
        return output


def _reset_network(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        for child in module.children():
            _reset_network(child)


class GINConv(nn.Module):
    def __init__(self, nn, eps=0.0, train_eps=False):
        super().__init__()
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer("eps", torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        _reset_network(self.nn)
        with torch.no_grad():
            self.eps.fill_(self.initial_eps)

    def forward(self, x, edge_index, size=None):
        source, target = (x, x) if isinstance(x, torch.Tensor) else x
        num_targets = source.size(0) if target is None else target.size(0)
        if size is not None and size[1] is not None:
            num_targets = size[1]
        output = _aggregate(source, edge_index, num_targets)
        if target is not None:
            output = output + (1 + self.eps) * target
        return self.nn(output)


class CellularEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gcn = GCNConv(input_dim, 256)
        self.sage = SAGEConv(256, 128)
        self.gin = GINConv(
            nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
        )
        self.gcn_norm = nn.BatchNorm1d(256)
        self.sage_norm = nn.BatchNorm1d(128)
        self.gin_norm = nn.BatchNorm1d(128)
        self.fusion = nn.Linear(512, 256)
        self.fusion_norm = nn.BatchNorm1d(256)
        self.projector = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)
        )
        self.output_norm = nn.BatchNorm1d(256)

    def forward(self, expression, edge_index):
        a = self.gcn_norm(F.relu(self.gcn(expression, edge_index)))
        b = self.sage_norm(F.relu(self.sage(a, edge_index)))
        c = self.gin_norm(F.relu(self.gin(b, edge_index)))
        h = self.fusion_norm(F.relu(self.fusion(torch.cat([a, b, c], dim=1))))
        h = F.dropout(h, p=0.05, training=self.training)
        return self.output_norm(self.projector(h))

    def encode_frozen(self, expression, edge_index):
        with torch.no_grad():
            self.eval()
            self.requires_grad_(False)
            return self(expression, edge_index).detach()
