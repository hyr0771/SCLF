import random
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_cluster import radius_graph
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class RegionCellContrast(nn.Module):

    def __init__(self, z_fused_dim,temp=0.2, margin=0.8,memory_size=512,num_regions = 100):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor([temp]))
        self.margin = margin

        self.cell_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )
        self.region_proj = nn.Sequential(
            nn.Linear(z_fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

        self.register_buffer('region_memory', torch.zeros(num_regions, memory_size))

    def _update_region_prototypes(self, z_fused):
        if self.region_memory is None:
            self.region_memory = z_fused.detach()
        else:
            self.region_memory = 0.9 * self.region_memory + 0.1 * z_fused.detach()

    def forward(self, z_local, z_fused, regions):

        h_cell = F.normalize(self.cell_proj(z_local), dim=-1)
        h_region = F.normalize(self.region_proj(z_fused), dim=-1)

        self._update_region_prototypes(h_region)

        sim_matrix = torch.mm(h_cell, self.region_memory.T)
        pos_mask = F.one_hot(regions, num_classes=h_region.size(0)).float()

        neg_mask = (torch.rand_like(sim_matrix) > 0.1).float() * (1 - pos_mask)
        sim_matrix = sim_matrix - (1 - pos_mask) * 1e6

        pos_sim = (sim_matrix * pos_mask).sum(1)
        neg_sim = torch.logsumexp(sim_matrix * neg_mask, dim=1)
        loss = -torch.mean(pos_sim - neg_sim)
        return h_cell, h_region, loss * self.temp.exp()


import torch.nn as nn

from torch_cluster import radius_graph

class CellCellContrast(nn.Module):

    def __init__(self, temp=0.3, alpha=0.8):
        super().__init__()

        self.base_temp = temp
        self.register_buffer('alpha', torch.tensor(alpha))

        self.view_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        self.feature_enhancer = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.temp = nn.Parameter(torch.tensor([temp]))

        self.sample_size = 1024
        self.max_pairs = 50000

    def _build_biological_view(self, edge_index, h, coords):

        if edge_index.numel() == 0:
            return (torch.empty((2, 0), device=h.device),
                    torch.empty((2, 0), device=h.device))
        h = F.normalize(h, p=2, dim=1)

        with torch.no_grad():

            idx = torch.randperm(h.size(0))[:self.sample_size]
            sampled_h = h[idx]
            feat_sim = F.cosine_similarity(
                sampled_h.unsqueeze(1),
                sampled_h.unsqueeze(0),
                dim=-1
            ).mean()
            feat_sim = torch.clamp(feat_sim, min=0.0, max=1.0)

            base_r = 0.15
            r = base_r * (1.2 - 0.4 * feat_sim.item())
            max_neighbors = min(10, int(20 + 10 * feat_sim.item()))

        spatial_edges = radius_graph(coords, r=r, max_num_neighbors=max_neighbors)

        edge_set = set(map(tuple, edge_index.cpu().numpy().T))
        spatial_set = set(map(tuple, spatial_edges.cpu().numpy().T))

        pos_edges = [t for t in spatial_set & edge_set]
        neg_edges = [t for t in spatial_set - edge_set]

        pos_tensor = torch.tensor(pos_edges, device=h.device).t()
        neg_tensor = torch.tensor(neg_edges, device=h.device).t()

        if len(neg_edges) > 3 * len(pos_edges):
            keep_ratio = 3 * len(pos_edges) / len(neg_edges)
            mask = torch.rand(neg_tensor.size(1), device=h.device) < keep_ratio
            neg_tensor = neg_tensor[:, mask]

        return pos_tensor, neg_tensor

    def _build_feature_view(self, h):
        n = h.size(0)
        if n < 2:
            return torch.empty((2, 0), device=h.device), torch.empty((2, 0), device=h.device)

        idx = torch.randint(0, n, (self.max_pairs * 2,), device=h.device)
        row, col = idx[::2], idx[1::2]
        mask = row != col
        row, col = row[mask][:self.max_pairs], col[mask][:self.max_pairs]

        sim = (h[row] * h[col]).sum(dim=1)

        self.alpha = torch.quantile(sim, 0.75) if sim.numel() > 0 else self.alpha

        pos_mask = sim > self.alpha
        neg_mask = sim < (self.alpha - 0.2)

        return torch.stack([row[pos_mask], col[pos_mask]]), \
            torch.stack([row[neg_mask], col[neg_mask]])

    def forward(self, h_cell, edge_index, coords):

        h_proj = self.view_proj(h_cell)

        view1_edges = self._build_feature_view(h_proj)
        view2_edges = self._build_biological_view(edge_index, h_proj, coords)

        loss1 = self._compute_view_loss(h_proj, *view1_edges)
        loss2 = self._compute_view_loss(h_proj, *view2_edges)
        h_enhanced = self.feature_enhancer(h_proj)

        return h_enhanced , 0.5 * (loss1 + loss2)

    def _compute_view_loss(self, h, pos_edges, neg_edges):

        temp = self.temp.clamp(min=0.1, max=1.0)

        pos_term = torch.tensor(0.0, device=h.device)
        if pos_edges.numel() > 0:
            pos_sim = F.cosine_similarity(h[pos_edges[0]], h[pos_edges[1]])
            pos_sim = torch.clamp(pos_sim, min=-1.0 + 1e-6, max=1.0 - 1e-6)
            pos_term = torch.exp(pos_sim / temp).sum()

        neg_term = torch.tensor(0.0, device=h.device)
        if neg_edges.numel() > 0:
            neg_sim = F.cosine_similarity(h[neg_edges[0]], h[neg_edges[1]])
            neg_sim = torch.clamp(neg_sim, min=-1.0 + 1e-6, max=1.0 - 1e-6)
            neg_term = torch.exp(neg_sim / temp).sum()

        denominator = pos_term + neg_term + 1e-8
        loss = - (pos_term / denominator).log()

        if torch.isnan(loss):
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalMessagePassing(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256, num_regions=100):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regions = num_regions

        self.up_aggregation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.up_projection = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim)
        )


        self.down_broadcast = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

        self.raw_residual_weight = nn.Parameter(torch.tensor(0.0))

        self.region_prototypes = nn.Parameter(torch.randn(num_regions, input_dim))
        nn.init.xavier_uniform_(self.region_prototypes)

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.prototype_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, z_local, z_fused, regions):

        original_z_local = z_local
        original_z_fused = z_fused
        residual_weight = torch.sigmoid(self.raw_residual_weight)

        similarity = torch.matmul(z_local, self.region_prototypes.t())
        cell_contributions = F.softmax(similarity / self.temperature, dim=1)

        updated_z_fused = original_z_fused.clone()

        unique_regions = torch.unique(regions)

        for region_id in unique_regions:
            region_mask = (regions == region_id)
            if region_mask.sum() == 0:
                projected_prototype = self.prototype_projection(
                    self.region_prototypes[region_id].unsqueeze(0)
                ).squeeze(0)

                updated_region_feature = self.up_projection(projected_prototype.unsqueeze(0)).squeeze(0)
            else:
                cell_weights = cell_contributions[region_mask, region_id]
                cell_weights = F.softmax(cell_weights, dim=0)
                aggregated = torch.sum(
                    self.up_aggregation(z_local[region_mask]) * cell_weights.unsqueeze(1),
                    dim=0
                )

                updated_region_feature = self.up_projection(aggregated.unsqueeze(0)).squeeze(0)

            updated_z_fused[region_id] = updated_region_feature

        similarity = torch.matmul(z_local, updated_z_fused.t())
        cell_receiving_weights = F.softmax(similarity / self.temperature, dim=1)

        cell_updates = []
        for cell_idx in range(len(z_local)):
            region_id = regions[cell_idx]
            if region_id < 0 or region_id >= len(updated_z_fused):

                cell_updates.append(original_z_local[cell_idx])
                continue

            weight = cell_receiving_weights[cell_idx, region_id]

            broadcasted = self.down_broadcast(updated_z_fused[region_id])
            cell_updates.append(broadcasted * weight)

        updated_z_local = torch.stack(cell_updates)

        if updated_z_local.size(1) != self.input_dim:

            if not hasattr(self, 'dim_adapter'):
                self.dim_adapter = nn.Linear(updated_z_local.size(1), self.input_dim).to(updated_z_local.device)
            updated_z_local = self.dim_adapter(updated_z_local)

        z_fused_residual = residual_weight * updated_z_fused + (1 - residual_weight) * original_z_fused
        z_local_residual = residual_weight * updated_z_local + (1 - residual_weight) * original_z_local

        return z_local_residual, z_fused_residual


class CommunicationModule(nn.Module):

    def __init__(self, comm_head_dim=256, num_heads=4, dropout=0.5):
        super().__init__()
        self.comm_head_dim = comm_head_dim
        self.num_heads = num_heads

        # self.comm_head = nn.Sequential(
        #     nn.Linear(512, 384),
        #     nn.GELU(),
        #     self.ResidualBlock(384, 384),
        #     self.ResidualBlock(384, 256),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, comm_head_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(comm_head_dim)
        # )
        self.comm_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, comm_head_dim),
            nn.GELU(),
            nn.LayerNorm(comm_head_dim)
        )



        self.self_attn = self.MultiHeadAttention(comm_head_dim, num_heads, dropout)
        self.cross_attn = self.MultiHeadAttention(comm_head_dim, num_heads, dropout)

        self.cross_attn2 = self.MultiHeadAttention(comm_head_dim, num_heads, dropout)

        # self.gate_unit = nn.Sequential(
        #     nn.Linear(comm_head_dim * 4, comm_head_dim * 2),
        #     nn.GELU(),
        #     self.ResidualBlock(comm_head_dim * 2, comm_head_dim * 2),  # 新增残差块
        #     nn.LayerNorm(comm_head_dim * 2),
        #     nn.Linear(comm_head_dim * 2, 4),
        #     nn.Sigmoid()
        # )
        self.gate_unit = nn.Sequential(
            nn.Linear(comm_head_dim * 4, comm_head_dim * 2),
            nn.GELU(),
            nn.Linear(comm_head_dim * 2, 4),
            nn.Sigmoid()
        )

        self.lr_adapters = nn.ModuleList([
            nn.Sequential(
                self.ResidualBlock(comm_head_dim, 256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.LayerNorm(128),

                nn.Linear(128, 1)
            )
        ])

        self.feature_enhancer = nn.Sequential(
            nn.Linear(comm_head_dim, comm_head_dim * 2),
            nn.GELU(),
            nn.LayerNorm(comm_head_dim * 2),
            nn.Linear(comm_head_dim * 2, comm_head_dim),
            nn.GELU(),
            nn.LayerNorm(comm_head_dim),
            nn.Dropout(0.2)
        )
        self.specificity_predictor = nn.Sequential(
            nn.Linear(comm_head_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_comm_weights()

    class ResidualBlock(nn.Module):

        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, out_dim)
            self.linear2 = nn.Linear(out_dim, out_dim)
            self.gate = nn.Linear(out_dim, out_dim)
            self.sigmoid = nn.Sigmoid()
            self.layer_norm = nn.LayerNorm(out_dim)

            if in_dim != out_dim:
                self.shortcut = nn.Linear(in_dim, out_dim)
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            out = F.gelu(self.linear1(x))
            out = self.linear2(out)
            out = self.layer_norm(out)
            gate = self.sigmoid(self.gate(out))
            out = gate * out + (1 - gate) * residual
            return out

    class MultiHeadAttention(nn.Module):

        def __init__(self, embed_dim, num_heads, dropout=0.5):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(embed_dim)

        def forward(self, query, key, value):

            if query.dim() == 2:
                query = query.unsqueeze(1)
                key = key.unsqueeze(1)
                value = value.unsqueeze(1)
                remove_dim = True
            else:
                remove_dim = False

            batch_size, seq_len, _ = query.size()

            Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_scores = torch.clamp(attn_scores, -50, 50)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            context = torch.matmul(attn_probs, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            context = self.out_proj(context)
            context = self.layer_norm(context)

            if remove_dim:
                context = context.squeeze(1)

            return context

    def _init_comm_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    # def compute_comm_score(self, data, cell_pairs):
    #
    #     cell_features = data['cell_features']
    #     if cell_features is None or cell_pairs is None or len(cell_pairs) == 0:
    #         return None
    #
    #     num_cells = cell_features.size(0)
    #     valid_mask = (cell_pairs[:, 0] < num_cells) & (cell_pairs[:, 1] < num_cells) & \
    #                  (cell_pairs[:, 0] >= 0) & (cell_pairs[:, 1] >= 0)
    #
    #     if not valid_mask.all():
    #         cell_pairs = cell_pairs[valid_mask]
    #         if len(cell_pairs) == 0:
    #             return torch.zeros(0, device=cell_features.device)
    #
    #     with torch.no_grad():
    #         cell_features = self.comm_head(cell_features)
    #
    #         cell_features = self.feature_enhancer(cell_features)
    #
    #     feat_i = cell_features[cell_pairs[:, 0]]
    #     feat_j = cell_features[cell_pairs[:, 1]]
    #
    #     feat_i = self.self_attn(feat_i, feat_i, feat_i)
    #     feat_j = self.self_attn(feat_j, feat_j, feat_j)
    #
    #     cross_i = self.cross_attn(feat_i, feat_j, feat_j)
    #     cross_j = self.cross_attn(feat_j, feat_i, feat_i)
    #
    #     cross_i2 = self.cross_attn2(cross_i, cross_j, cross_j)
    #     cross_j2 = self.cross_attn2(cross_j, cross_i, cross_i)
    #
    #     cross_i = 0.6 * cross_i + 0.4 * cross_i2
    #     cross_j = 0.6 * cross_j + 0.4 * cross_j2
    #
    #     gate_input = torch.cat([feat_i, feat_j, cross_i, cross_j], dim=1)
    #     gate_weights = self.gate_unit(gate_input)
    #     g1, g2, g3, g4 = gate_weights.chunk(4, dim=1)
    #
    #     fused_feat = torch.zeros_like(feat_i)
    #     fused_feat.add_(g1 * feat_i)
    #     fused_feat.add_(g2 * feat_j)
    #     fused_feat.add_(g3 * cross_i)
    #     fused_feat.add_(g4 * cross_j)
    #
    #     fused_feat = nn.LayerNorm(fused_feat.size(1)).to(fused_feat.device)(fused_feat)
    #
    #     specificity = self.lr_adapters[0](fused_feat)
    #     specificity = torch.clamp(specificity, -10, 10)
    #
    #     return torch.sigmoid(specificity).squeeze(1)
    def compute_comm_score(self, data, cell_pairs):
        cell_features = self.comm_head(data['cell_features'])

        feat_i = cell_features[cell_pairs[:, 0]]
        feat_j = cell_features[cell_pairs[:, 1]]

        attn_i = self.self_attn(feat_i, feat_i, feat_i)
        attn_j = self.self_attn(feat_j, feat_j, feat_j)
        cross_i = self.cross_attn(attn_i, attn_j, attn_j)
        cross_j = self.cross_attn(attn_j, attn_i, attn_i)

        gate_input = torch.cat([attn_i, attn_j, cross_i, cross_j], dim=1)
        gate_weights = self.gate_unit(gate_input)
        g1, g2, g3, g4 = gate_weights.chunk(4, dim=1)

        fused_feat = (
                g1 * attn_i +
                g2 * attn_j +
                g3 * cross_i +
                g4 * cross_j
        )

        return torch.sigmoid(self.specificity_predictor(fused_feat)).squeeze(1)

    def compute_comm_loss(self, data, cell_pairs, labels):

        scores = self.compute_comm_score(data, cell_pairs)
        if scores is None:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)
        if scores.dim() > 1:
            scores = scores.squeeze()
        if scores.dim() == 1 and labels.dim() == 2:
            labels = labels.squeeze(0)
        elif scores.dim() == 0 and labels.dim() == 1:
            scores = scores.unsqueeze(0)
        if scores.size(0) != labels.size(0):
            min_size = min(scores.size(0), labels.size(0))
            scores = scores[:min_size]
            labels = labels[:min_size]

        scores = torch.clamp(scores, 1e-7, 1 - 1e-7)

        logits = torch.log(scores / (1 - scores + 1e-8))
        logits = torch.clamp(logits, -50, 50)

        pos_weight = torch.tensor([4]).to(scores.device)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight
        )
        return loss



class CommunicationTrainer:
    def __init__(self, model, embedding_path=None, device='cuda'):
        self.model = model.to(device)
        if embedding_path is not None:
            model.load_state_dict(torch.load(embedding_path))
        self.device = device
        self.training_records = []
    def train(self, train_loader, valid_loader=None, epochs=1, lr=1e-3, save_interval=100):

        trainable_params = self.model.parameters()
        comm_optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(
            comm_optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True,
            min_lr=1e-7
        )
        all_params = []
        for group in trainable_params:
            if isinstance(group['params'], (list, tuple)):
                all_params.extend(group['params'])
            else:
                all_params.append(group['params'])

        best_valid_loss = float('inf')
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            start_time = time.time()

            for batch in train_loader:

                cell_pairs = batch['cell_pairs'].to(self.device)
                labels = batch['labels'].to(self.device)
                features = batch['features']
                if not features.is_cuda and self.device != 'cpu':
                    features = features.to(self.device)

                comm_optimizer.zero_grad()

                loss = self.model.compute_comm_loss(
                    {'cell_features': features},
                    cell_pairs,
                    labels,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                comm_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            elapsed = time.time() - start_time


            auc, f1, acc, precision, recall = None, None, None, None, None
            valid_loss = None

            if epoch % save_interval == 0 or epoch % epochs == 0:
                if valid_loader is not None:
                    valid_loss = self._compute_validation_loss(valid_loader)

                    auc, f1, acc, precision, recall = self.evaluate(valid_loader)
                    scheduler.step(valid_loss)
                    record = {
                        'epoch': epoch,
                        'loss': avg_loss,
                        'valid_loss': valid_loss,
                        'auc': auc if auc is not None else float('nan'),
                        'acc': acc if acc is not None else float('nan'),
                        'precision': precision if precision is not None else float('nan'),
                        'f1': f1 if f1 is not None else float('nan'),
                        'recall': recall if recall is not None else float('nan'),
                        'time': elapsed,
                        'lr': comm_optimizer.param_groups[0]['lr']
                    }
                    self.training_records.append(record)
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(self.model.state_dict(), "best_comm_model.pth")
                    if epoch % save_interval == 0:
                        print(
                            f"Comm Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                            f"Valid Loss: {valid_loss:.4f} | "
                            f"AUC: {auc:.4f}| Acc: {acc:.4f}| Precision: {precision:.4f}| F1: {f1:.4f}| Recall: {recall:.4f}| "
                            f"LR: {comm_optimizer.param_groups[0]['lr']:.2e} | "
                            f"Time: {elapsed:.1f}s")
                else:
                    print(f"Comm Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        pd.DataFrame(self.training_records).to_csv("training_records.csv", index=False)
        print(f"通信模块训练完成")

    def _compute_validation_loss(self, valid_loader):

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in valid_loader:
                cell_pairs = batch['cell_pairs'].to(self.device)
                labels = batch['labels'].to(self.device)
                features = batch['features']
                if not features.is_cuda and self.device != 'cpu':
                    features = features.to(self.device)

                loss = self.model.compute_comm_loss(
                    {'cell_features': features},
                    cell_pairs,
                    labels,
                )
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float('inf')
    def evaluate(self, comm_data_loader):
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in comm_data_loader:
                cell_pairs = batch['cell_pairs'].to(self.device)
                labels = batch['labels'].to(self.device)
                features = batch['features']


                if not features.is_cuda and self.device != 'cpu':
                    features = features.to(self.device)


                scores = self.model.compute_comm_score(
                    {'cell_features': features},
                    cell_pairs,
                )


                if scores is not None and len(scores) > 0:
                    all_scores.append(scores.cpu())
                    all_labels.append(labels.cpu())

        if len(all_scores) == 0:
            print("警告: 评估时没有有效样本")
            return 0.5, 0.0, 0.0, 0.0, 0.0


        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)


        scores_np = all_scores.numpy()
        labels_np = all_labels.numpy()


        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc


        try:
            roc_auc = roc_auc_score(labels_np, scores_np)
        except ValueError:
            roc_auc = 0.5


        predictions = (all_scores > 0.5).float()


        try:
            f1 = f1_score(labels_np, predictions.numpy())
        except ValueError:
            f1 = 0.0

        acc = accuracy_score(labels_np, predictions.numpy())
        precision = precision_score(labels_np, predictions.numpy())
        recall = recall_score(labels_np, predictions.numpy())

        return roc_auc, f1, acc, precision, recall
class ResNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        self.expand = nn.Linear(input_dim, 1024)
        self.block1 = self._make_res_block(1024, 1024)
        self.block2 = self._make_res_block(1024, 1024)
        self.block3 = self._make_res_block(1024, 1024)
        self.project = nn.Sequential(
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, output_dim)
        )

    def _make_res_block(self, in_dim, out_dim):

        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        identity = self.expand(x)
        x = self.block1(identity) + identity
        x = self.block2(x) + x
        x = self.block3(x) + x
        return self.project(x)
class BioContrastiveModel(nn.Module):
    def __init__(self, z_local_dim,z_fused_dim, num_regions=100):
        super().__init__()
        self.region_cell = RegionCellContrast(z_fused_dim)
        self.cell_cell = CellCellContrast()

        self.spatial_encoder = ResNetEncoder(z_local_dim, output_dim=512)
        self.message_passing = HierarchicalMessagePassing(
            input_dim=512,
            hidden_dim=256,
            num_regions=num_regions
        )
        self.region_projection = ResNetEncoder(z_fused_dim, output_dim=512)

    class ResidualBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, out_dim)
            self.linear2 = nn.Linear(out_dim, out_dim)
            self.gate = nn.Linear(out_dim, out_dim)
            self.sigmoid = nn.Sigmoid()

            if in_dim != out_dim:
                self.shortcut = nn.Linear(in_dim, out_dim)
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            out = F.gelu(self.linear1(x))
            out = self.linear2(out)
            gate = self.sigmoid(self.gate(out))
            out = gate * out + (1 - gate) * residual
            return out

    def forward(self, data):

        z_local = data['z_local']
        z_fused = data['z_fused']
        regions = data['regions']
        coords = data['coords']
        edge_index = data['functional_edges']

        h_local = self.spatial_encoder(z_local)

        h_cell_rc, h_region_rc, rc_loss = self.region_cell(
            h_local,
            z_fused,
            regions
        )

        h_cell_cc, cc_loss = self.cell_cell(
            h_cell_rc,
            edge_index,
            coords=coords
        )
        projected_z_fused = self.region_projection(z_fused)

        h_cell_mp, z_fused_mp = self.message_passing(
            h_cell_cc,
            projected_z_fused,
            regions
        )
        total_loss = 0.5 * rc_loss + 0.5 * cc_loss
        return {
            'total_loss': total_loss,
            'region_loss': rc_loss,
            'cell_loss': cc_loss,
            'cell_features': h_cell_mp.detach(),
            'region_features': z_fused_mp.detach()
        }

import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BioContrastTrainer:
    def __init__(self, model, train_dataset,valid_dataset, device='cuda'):
        self.model = model.to(device)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)

        self.best_loss = float('inf')
        self.train_log = []
        self.training_records = []
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        data = self.train_dataset[0]
        data = {k: v.to(self.device) for k, v in data.items()}

        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = outputs['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        total_loss += loss.item()
        elapsed = time.time() - start_time

        valid_loss = self._evaluate_dataset(self.valid_dataset)

        log_msg = (
            f"Epoch {epoch:03d} | "
            f"Train Loss: {total_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "  
            f"Region Loss: {outputs['region_loss'].item():.4f} | "
            f"Cell Loss: {outputs['cell_loss'].item():.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if epoch % 100 == 0:
            print(log_msg)

        self.scheduler.step(valid_loss)
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            torch.save(self.model.state_dict(), "best_model.pth")

        self.train_log.append({
            'epoch': epoch,
            'total_loss': total_loss,
            'valid_loss': valid_loss,
            'region_loss': outputs['region_loss'].item(),
            'cell_loss': outputs['cell_loss'].item()
        })

    def train(self, epochs=2000):
        print("Initial loss:", self._evaluate_dataset(self.train_dataset))
        print("Initial valid loss:", self._evaluate_dataset(self.valid_dataset))
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)

        self.save_training_log()
        return pd.DataFrame(self.train_log)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            data = self.dataset[0]
            data = {k: v.to(self.device) for k, v in data.items()}
            outputs = self.model(data)
        return outputs['total_loss'].item()

    def _evaluate_dataset(self, dataset):
        self.model.eval()
        with torch.no_grad():
            data = dataset[0]
            data = {k: v.to(self.device) for k, v in data.items()}
            outputs = self.model(data)
        return outputs['total_loss'].item()

    def save_training_log(self):
        pd.DataFrame(self.train_log).to_csv("training_log.csv", index=False)

import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class BioDataset(Dataset):
    def __init__(self, data_paths, split_mask=None):

        self.data_paths = data_paths
        coords_df = pd.read_csv(data_paths['coords'])

        for col in coords_df.columns:
            if coords_df[col].dtype == object:
                try:
                    coords_df[col] = pd.to_numeric(coords_df[col], errors='coerce')
                except:
                    pass
        self.full_coords = torch.FloatTensor(coords_df.values)

        z_local_df = pd.read_csv(data_paths['z_local'])
        self.full_z_local = torch.FloatTensor(z_local_df.values)

        self.full_regions = torch.load(data_paths['regions'])

        z_fused_df = pd.read_csv(data_paths['z_fused'])
        self.z_fused = torch.FloatTensor(z_fused_df.values)

        self.full_edges = torch.load(data_paths['edges']).long()

        self._process_split(split_mask)

    def _process_split(self, split_mask):

        if split_mask is None:
            self.coords = self.full_coords
            self.z_local = self.full_z_local
            self.regions = self.full_regions
            self.valid_indices = torch.arange(len(self.full_z_local))
        else:
            self.valid_indices = torch.where(split_mask)[0]
            self.coords = self.full_coords[self.valid_indices]
            self.z_local = self.full_z_local[self.valid_indices]
            original_regions = self.full_regions[self.valid_indices]
            self.unique_regions = torch.unique(original_regions)
            self.region_mapping = {int(orig): new for new, orig in enumerate(self.unique_regions)}
            self.regions = torch.tensor([self.region_mapping[int(r)] for r in original_regions])

        self._process_edges()

    def _process_edges(self):

        idx_mapping = {int(orig): new for new, orig in enumerate(self.valid_indices.tolist())}

        valid_edges = []
        for edge in self.full_edges.t().cpu().numpy():
            src, dst = edge[0].item(), edge[1].item()
            if src in idx_mapping and dst in idx_mapping:
                valid_edges.append([
                    idx_mapping[src],
                    idx_mapping[dst]
                ])
        self.functional_edges = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'coords': self.coords,
            'z_local': self.z_local,
            'z_fused': self.z_fused,
            'regions': self.regions,
            'functional_edges': self.functional_edges
        }

def load_and_split_cell_pairs(file_path, train_ratio=0.8, random_seed=42):
    cell_pairs = []
    labels = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0] == "to":
                    try:
                        cell_i = int(parts[1])
                        cell_j = int(parts[2])
                        label = int(parts[3])
                        cell_pairs.append([cell_i, cell_j])
                        labels.append(label)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"加载细胞对文件错误: {e}")
        return None, None, None, None

    if len(cell_pairs) == 0:
        print("警告: 没有有效的细胞对可加载")
        return None, None, None, None


    cell_pairs_tensor = torch.tensor(cell_pairs, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    num_pairs = len(cell_pairs_tensor)
    indices = torch.randperm(num_pairs, generator=torch.Generator().manual_seed(random_seed))

    split_idx = int(num_pairs * train_ratio)

    train_pairs = cell_pairs_tensor[indices[:split_idx]]
    train_labels = labels_tensor[indices[:split_idx]]

    valid_pairs = cell_pairs_tensor[indices[split_idx:]]
    valid_labels = labels_tensor[indices[split_idx:]]

    return train_pairs, train_labels, valid_pairs, valid_labels

def create_train_valid_split(dataset, train_ratio=0.8):

    n = len(dataset.full_z_local)

    train_mask = torch.rand(n) < train_ratio
    valid_mask = ~train_mask

    train_dataset = BioDataset(
        data_paths=dataset.data_paths,
        split_mask=train_mask
    )

    valid_dataset = BioDataset(
        data_paths=dataset.data_paths,
        split_mask=valid_mask
    )

    return train_dataset, valid_dataset

class CommDataset(Dataset):

    def __init__(self, features, cell_pairs, labels, batch_size=10000, mode='train'):

        self.cell_features = features['cell_features']
        self.num_cells = self.cell_features.size(0)
        self.cell_pairs = cell_pairs
        self.labels = labels
        self.batch_size = batch_size
        self.mode = mode

        self._filter_invalid_pairs()

        pos_count = (self.labels == 1).sum().item()
        neg_count = (self.labels == 0).sum().item()

    def _filter_invalid_pairs(self):

        valid_mask = (self.cell_pairs[:, 0] < self.num_cells) & (self.cell_pairs[:, 1] < self.num_cells)

        invalid_count = len(self.cell_pairs) - valid_mask.sum().item()
        if invalid_count > 0:
            print(f"过滤掉 {invalid_count} 个无效细胞对 (总细胞数: {self.num_cells})")

        self.cell_pairs = self.cell_pairs[valid_mask]
        self.labels = self.labels[valid_mask]

    def __len__(self):
        return (len(self.cell_pairs) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.cell_pairs))

        if self.mode == 'train' and random.random() < 0.3 and len(self.cell_pairs) > 1:

            neg_indices = torch.where(self.labels == 0)[0]

            if len(neg_indices) > 0:

                neg_idx = random.choice(neg_indices.tolist())
                return {
                    'cell_pairs': self.cell_pairs[neg_idx:neg_idx + 1],
                    'labels': self.labels[neg_idx:neg_idx + 1],
                    'features': self.cell_features
                }

        return {
            'cell_pairs': self.cell_pairs[start_idx:end_idx],
            'labels': self.labels[start_idx:end_idx],
            'features': self.cell_features
        }





