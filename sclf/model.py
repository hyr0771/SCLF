import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


class SpatialContext:
    cells: torch.Tensor
    neighbors: torch.Tensor
    regions: torch.Tensor
    regional: torch.Tensor
    _mean: torch.Tensor | None = None

    def neighborhood_mean(self):
        if self._mean is None:
            idx = self.neighbors.clamp_min(0)
            valid = (self.neighbors >= 0).to(self.cells.dtype)
            self._mean = (self.cells[idx] * valid[:, :, None]).sum(1) / valid.sum(
                1
            ).clamp_min(1)[:, None]
        return self._mean

    neighborhood_mean = property(neighborhood_mean)


SpatialContext = dataclass(SpatialContext)


class WeightColumns(nn.Module):
    def __init__(self, rows, columns, input_width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, columns))
        nn.init.uniform_(
            self.weight, -1 / math.sqrt(input_width), 1 / math.sqrt(input_width)
        )


class RegionCell(nn.Module):
    def __init__(self, num_regions, dim=256, output_dim=128):
        super().__init__()
        self.num_regions = num_regions
        self.temp = nn.Parameter(torch.tensor(0.2))
        self.cell_proj = nn.Linear(dim, 64)
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("proto", torch.randn(8, dim, generator=generator))
        self.register_buffer(
            "region_centroid", torch.zeros(num_regions, dim), persistent=False
        )
        self.emb = nn.Sequential(
            nn.Linear(64 + 4 + 8 + 2 * dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
        self.local_region = WeightColumns(output_dim, dim, input_width=3 * dim + 5)

    def forward(self, context):
        z, zr, regions = (context.cells, context.regional, context.regions)
        normalized = F.normalize(z, dim=-1)
        ctx = normalized @ F.normalize(self.proto, dim=-1).T
        logits = normalized @ F.normalize(zr, dim=-1).T / self.temp.exp().clamp(0.05, 1)
        p = F.softmax(logits, dim=1)
        own = p.gather(1, regions[:, None]).squeeze(1)
        ent = -(p.clamp_min(1e-09).log() * p).sum(1)
        top = p.topk(2, dim=1).values
        stats = torch.stack([own, ent, top[:, 0], top[:, 0] - top[:, 1]], dim=1)
        region = zr[regions]
        emb = self.emb(
            torch.cat(
                [self.cell_proj(z), stats, ctx, region, (z - region).abs()], dim=1
            )
        )
        count = emb.new_zeros(self.num_regions).index_add_(
            0, regions, emb.new_ones(len(regions))
        )
        sums = emb.new_zeros((self.num_regions, emb.shape[1])).index_add_(
            0, regions, emb
        )
        centroid = sums / count.clamp_min(1)[:, None]
        score = F.normalize(emb, dim=1) @ F.normalize(centroid, dim=1).T
        ownmean = (sums[regions] - emb) / (count[regions] - 1).clamp_min(1)[:, None]
        ownscore = (F.normalize(emb, dim=1) * F.normalize(ownmean, dim=1)).sum(1)
        score = score.scatter(1, regions[:, None], ownscore[:, None]) / self.temp.clamp(
            0.05, 1
        )
        valid = count[regions] > 1
        loss = (
            F.cross_entropy(score[valid], regions[valid])
            if bool(valid.any()) and self.num_regions > 1
            else emb.sum() * 0
        )
        return {"emb": emb, "loss": loss}


class CellCell(nn.Module):
    def contrast(self, emb, neighbors):
        idx, valid = (neighbors.clamp_min(0), neighbors >= 0)
        k = neighbors.shape[1]
        kp, kn = (min(self.k_pos, k), min(self.k_neg, k))
        en = F.normalize(emb, dim=1)
        selected = torch.cat([idx[:, :kp], idx[:, k - kn :]], dim=1)
        mask = torch.cat([valid[:, :kp], valid[:, k - kn :]], dim=1)
        score = (en[:, None, :] * en[selected]).sum(2) / self.temp.clamp(0.1, 1)
        lp = torch.logsumexp(
            score[:, :kp].masked_fill(~mask[:, :kp], -torch.inf), dim=1
        )
        ln = torch.logsumexp(
            score[:, kp:].masked_fill(~mask[:, kp:], -torch.inf), dim=1
        )
        keep = mask[:, :kp].any(1) & mask[:, kp:].any(1)
        return (
            torch.logaddexp(lp[keep], ln[keep]).sub(lp[keep]).mean()
            if bool(keep.any())
            else emb.sum() * 0
        )

    def __init__(self, dim=256, output_dim=128, k_pos=15, k_neg=10):
        super().__init__()
        self.cell_weight = nn.Parameter(torch.empty(output_dim, 2 * dim + 5))
        self.bias = nn.Parameter(torch.empty(output_dim))
        self.temp = nn.Parameter(torch.tensor(0.3))
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.register_buffer(
            "positive_pool", torch.full((1, 15), 1 / 15), persistent=False
        )
        bound = 1 / math.sqrt(3 * dim + 5)
        nn.init.uniform_(self.cell_weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def statistics(self, z, neighbors):
        idx = neighbors.clamp_min(0)
        valid = neighbors >= 0
        zn = F.normalize(z, dim=-1)
        sim = (zn[idx] * zn[:, None, :]).sum(-1).masked_fill(~valid, -torch.inf)
        k = neighbors.shape[1]
        kp, kn = (min(self.k_pos, k), min(self.k_neg, k))
        if kp <= 0 or kn <= 0 or kp + kn > k:
            raise ValueError(
                "Positive and negative neighborhoods must be nonempty and disjoint"
            )
        pos, neg = (sim[:, :kp], sim[:, k - kn :])
        positive_mean = F.linear(
            pos.masked_fill(~torch.isfinite(pos), 0), self.positive_pool
        ).squeeze(1)
        mx = sim.max(1).values
        mn = sim.masked_fill(~valid, torch.inf).min(1).values
        return torch.stack(
            [
                positive_mean,
                neg.masked_fill(~torch.isfinite(neg), 0).mean(1),
                torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx)),
                torch.where(torch.isfinite(mn), mn, torch.zeros_like(mn)),
                positive_mean - neg.masked_fill(~torch.isfinite(neg), 0).mean(1),
            ],
            dim=1,
        )


class CoordinateGraphLayer(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.self_projection = nn.Linear(hidden, hidden, bias=False)
        self.neighbor_projection = nn.Linear(hidden, hidden, bias=False)

    def forward(self, features, neighbors):
        x = self.norm(features)
        keep = neighbors >= 0
        adjacent = x[neighbors.clamp_min(0)] * keep.unsqueeze(-1)
        mean = adjacent.sum(1) / keep.sum(1).clamp_min(1).unsqueeze(-1)
        return features + F.gelu(
            self.self_projection(x) + self.neighbor_projection(mean)
        )


class ContextTransformerLayer(nn.Module):
    def __init__(self, hidden=128, head_dim=64, heads=4):
        super().__init__()
        self.norm1, self.norm2 = (nn.LayerNorm(hidden), nn.LayerNorm(hidden))
        self.qkv = nn.Linear(hidden, 3 * heads * head_dim, bias=False)
        self.output = nn.Linear(heads * head_dim, hidden, bias=False)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )
        self.heads, self.head_dim = (heads, head_dim)

    def forward(self, tokens):
        n, length, _ = tokens.shape
        q, k, v = self.qkv(self.norm1(tokens)).chunk(3, dim=-1)

        def reshape(t):
            return t.reshape(n, length, self.heads, self.head_dim).transpose(1, 2)

        mixed = F.scaled_dot_product_attention(
            reshape(q), reshape(k), reshape(v), dropout_p=0.0
        )
        mixed = mixed.transpose(1, 2).reshape(n, length, -1)
        tokens = tokens + self.output(mixed)
        return tokens + self.feedforward(self.norm2(tokens))


class SpatialContextEncoder(nn.Module):
    def __init__(self, cell_chunk=48):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU())
        self.graph_layers = nn.ModuleList([CoordinateGraphLayer() for _ in range(3)])
        self.sequence = nn.GRU(128, 128, batch_first=True)
        self.transformers = nn.ModuleList([ContextTransformerLayer() for _ in range(3)])
        self.pool_attention = nn.Linear(128, 1, bias=False)
        self.depth_logits = nn.Parameter(torch.zeros(1))
        self.readout = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256))
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.cell_chunk = cell_chunk

    def _chunk(self, sequence, n, count, length):
        states, _ = self.sequence(sequence)
        tokens = states.reshape(n, count, length, -1)[:, :, 1:]
        tokens = tokens.reshape(n, count * (length - 1), -1)
        with torch.autocast(
            device_type=tokens.device.type, enabled=tokens.is_cuda, dtype=torch.bfloat16
        ):
            for layer in self.transformers:
                tokens = layer(tokens)
            weights = torch.softmax(self.pool_attention(tokens), dim=1)
            return self.readout((weights * tokens).sum(1)).float()

    def forward(self, features, paths, neighbors):
        projected = self.project(features)
        for layer in self.graph_layers:
            projected = layer(projected, neighbors[:, :15])
        outputs = []
        for start in range(0, len(paths), self.cell_chunk):
            bank = paths[start : start + self.cell_chunk]
            n, count, length = bank.shape
            sequence = projected[bank].reshape(n * count, length, -1)
            value = self._chunk(sequence, n, count, length)
            outputs.append(value)
        residual = torch.cat(outputs, 0)
        scale = features.std(0, unbiased=False).mean().clamp_min(0.0001).detach()
        return features + torch.sigmoid(self.gate_logit) * scale * residual


class MessagePassing(nn.Module):
    def __init__(self, num_regions, cell_chunk=48):
        super().__init__()
        self.num_regions = num_regions
        self.up = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.down = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.emb = nn.Sequential(
            nn.Linear(257, 192), nn.GELU(), nn.Linear(192, 128), nn.GELU()
        )
        self.context = SpatialContextEncoder(cell_chunk)

    def forward(self, context):
        z_local, z_fused, regions = (context.cells, context.regional, context.regions)
        a = self.up(z_local)
        cnt = (
            z_local.new_zeros(self.num_regions)
            .index_add(0, regions, z_local.new_ones(len(regions)))
            .clamp_min(1.0)
        )
        agg = z_local.new_zeros((self.num_regions, a.shape[1])).index_add(0, regions, a)
        ctx = self.down(z_fused)[regions] + (agg / cnt[:, None])[regions]
        w = torch.sigmoid(
            (z_local * z_fused[regions]).sum(-1, keepdim=True)
            / math.sqrt(z_local.shape[1])
        )
        emb = self.emb(torch.cat([ctx, w], dim=1))
        loss = (1.0 - F.cosine_similarity(z_local, z_fused[regions], dim=-1)).mean()
        return {"emb": emb, "loss": loss}


class LinearDotHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1792, 1)
        self.projection = nn.Linear(256, 16, bias=False)
        with torch.no_grad():
            self.linear.weight[:, 1024:].zero_()

    def forward(self, x):
        a, b = (self.projection(x[:, :256]), self.projection(x[:, 256:512]))
        return self.linear(x) + (a * b).sum(1, keepdim=True) / math.sqrt(16)


class SharedReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = LinearDotHead()
        self.local_cell_fusion = nn.Sequential(
            nn.GELU(), nn.Linear(128, 128), nn.GELU()
        )

    def local(self, context, region_cell, cell_cell):
        z = context.cells
        mean = context.neighborhood_mean
        stats = cell_cell.statistics(z, context.neighbors)
        inputs = torch.cat([z, mean, z - mean, stats], 1)
        weight = torch.cat(
            [
                cell_cell.cell_weight[:, :256],
                region_cell.local_region.weight,
                cell_cell.cell_weight[:, 256:512],
                cell_cell.cell_weight[:, 512:],
            ],
            1,
        )
        return self.local_cell_fusion(F.linear(inputs, weight, cell_cell.bias))

    def forward(self, h, pairs, embeddings):
        if pairs.numel() == 0:
            return h.new_zeros((0,))
        i, j = (pairs[:, 0].long(), pairs[:, 1].long())
        hi, hj = (h[i], h[j])
        blocks = [hi, hj, (hi - hj).abs(), hi * hj]
        for name in ("k1", "local_cell", "k3"):
            e = embeddings[name]
            blocks.extend((e[i], e[j]))
        return self.net(torch.cat(blocks, 1)).squeeze(1)


class SCLF(nn.Module):
    def parameters(self, recurse=True):
        if not recurse:
            yield from super().parameters(recurse=False)
            return
        ordered = list(self.named_parameters())
        for context_parameters in (False, True):
            for name, parameter in ordered:
                if name.startswith("k3.context.") == context_parameters:
                    yield parameter

    def _regional_mean(z, regions, count):
        sums = z.new_zeros((count, z.shape[1])).index_add(0, regions, z)
        sizes = torch.zeros(count, device=z.device, dtype=z.dtype).index_add(
            0, regions, torch.ones_like(regions, dtype=z.dtype)
        )
        if int((sizes == 0).sum()):
            raise ValueError("Spatial partition contains an empty region")
        return sums / sizes.unsqueeze(1)

    _regional_mean = staticmethod(_regional_mean)

    def compact_state(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if name not in ("fixed_features", "ordered_paths")
        }

    def load_weights(self, state):
        expected = self.compact_state()
        if set(state) != set(expected):
            raise ValueError(
                dict(
                    missing=sorted(set(expected) - set(state)),
                    unexpected=sorted(set(state) - set(expected)),
                )
            )
        complete = dict(state)
        complete.update(
            fixed_features=self.fixed_features, ordered_paths=self.ordered_paths
        )
        return self.load_state_dict(complete, strict=True)

    def __init__(self, features, paths, num_regions, k_neg=10, k_pos=15):
        super().__init__()
        self.num_regions = num_regions
        self.lift = nn.Identity()
        with torch.random.fork_rng(devices=[]):
            self.k1 = RegionCell(num_regions)
            self.k2 = CellCell(k_pos=k_pos, k_neg=k_neg)
            self.k3 = MessagePassing(num_regions, 48)
            self.comm = SharedReadout()
        self.register_buffer(
            "fixed_features", torch.as_tensor(features, dtype=torch.float32)
        )
        self.register_buffer("ordered_paths", torch.as_tensor(paths, dtype=torch.long))
        self._initialize_parameters()

    def _initialize_parameters(self):
        with torch.no_grad():

            def reset(module):
                for layer in module.modules():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

            seed = torch.initial_seed()
            torch.empty(131584).uniform_()
            self.k1.cell_proj.reset_parameters()
            torch.empty(2631750).uniform_()
            context = self.k3.context
            reset(context.project)
            reset(context.graph_layers[0])
            reset(context.graph_layers[1])
            torch.empty(65536).uniform_()
            context.sequence.reset_parameters()
            reset(context.transformers[0])
            torch.empty(788352).uniform_()
            context.pool_attention.reset_parameters()
            reset(context.readout)
            for offset, module in (
                (271830, self.comm.net),
                (271831, context.transformers[1]),
                (271832, context.graph_layers[2]),
                (271833, context.transformers[2]),
            ):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + offset)
                    reset(module)
            self.comm.net.linear.weight[:, 1024:].zero_()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 271828)
                reset(self.k1.emb)
                projection = nn.Linear(773, 128)
                self.k1.local_region.weight.copy_(projection.weight[:, 256:512])
                self.k2.cell_weight.copy_(
                    torch.cat(
                        [projection.weight[:, :256], projection.weight[:, 512:]], 1
                    )
                )
                self.k2.bias.copy_(projection.bias)
                self.comm.local_cell_fusion[1].reset_parameters()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 271829)
                reset(self.k3.up)
                reset(self.k3.down)
                reset(self.k3.emb)

    def forward(self, batch, cell_pairs=None, labels=None):
        base = self.fixed_features
        z = self.k3.context(base, self.ordered_paths, batch["knn_idx"])
        h = self.lift(base)
        regional = self._regional_mean(z, batch["regions"], self.num_regions)
        context = SpatialContext(z, batch["knn_idx"], batch["regions"], regional)
        rc, mp = (self.k1(context), self.k3(context))
        cc = self.comm.local(context, self.k1, self.k2)
        cc_loss = self.k2.contrast(cc, batch["knn_idx"])
        embeddings = {"k1": rc["emb"], "local_cell": cc, "k3": mp["emb"]}
        out = dict(
            h=h,
            z_local=base,
            k_input=z,
            embs=embeddings,
            loss_k1=rc["loss"],
            loss_k2=cc_loss,
            loss_k3=mp["loss"],
        )
        if cell_pairs is not None:
            logits = self.comm(h, cell_pairs, embeddings)
            out.update(logits=logits, probs=torch.sigmoid(logits))
            if labels is not None:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                out.update(
                    loss_comm=loss, total_loss=(loss + 0.5 * (rc["loss"] + cc_loss)) / 2
                )
        return out

    def score_splits(self, batch, splits, chunk=32768):
        with torch.no_grad():
            self.eval()
            bank = self(batch)
            result = {}
            for name, (pairs, _) in splits.items():
                scores = []
                for start in range(0, len(pairs), chunk):
                    logits = self.comm(
                        bank["h"], pairs[start : start + chunk], bank["embs"]
                    )
                    scores.append(logits.sigmoid().cpu())
                result[name] = (
                    torch.cat(scores).numpy() if scores else torch.empty(0).numpy()
                )
            return result
