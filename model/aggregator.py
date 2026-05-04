import copy
import time

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch_geometric.data import Data
from torch_geometric.graphgym import SAGEConv
from torch_geometric.nn import GCNConv, GATv2Conv, LayerNorm, TransformerConv
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
import pandas as pd
from torch_geometric.utils import add_self_loops
from datetime import datetime
import torch
from torch import nn, scatter_add
from torch_geometric.nn import GATv2Conv, GraphSAGE, ChebConv
from torch_geometric.data import Data
import os
import json
import pickle
from torch import Tensor
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter
from torch_scatter import scatter_softmax, scatter_mean, scatter_max,scatter_add
from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import connected_components
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import to_undirected
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, JumpingKnowledge
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class TrainingLogger:
    

    def __init__(self):
        self.logs = {
            'timestamp': [],
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

    def record(self, epoch, train_loss, val_loss, lr):
        self.logs['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logs['epoch'].append(epoch + 1)
        self.logs['train_loss'].append(train_loss)
        self.logs['val_loss'].append(val_loss)
        self.logs['lr'].append(lr)

    def save(self, filename="training_log.csv"):
        pd.DataFrame(self.logs).to_csv(filename, index=False)


import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512  
GRAD_ACCUM_STEPS = 4  


def manual_batch_generator(indices, batch_size):
    
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]



def prepare_data(expr, edge_data):
    
    if isinstance(expr, pd.DataFrame):
        x = torch.tensor(expr.values, dtype=torch.float32)
    else:
        x = torch.tensor(expr, dtype=torch.float32)

    
    
    if not isinstance(edge_data, torch.Tensor):
        edge_data = torch.tensor(edge_data, dtype=torch.long)

    
    print(f"边索引形状: {edge_data.shape}")
    print(f"边索引最小值: {edge_data.min().item()}")
    print(f"边索引最大值: {edge_data.max().item()}")

    
    if (edge_data < 0).any():
        print("警告: 边索引包含负值，进行修正...")
        edge_data = torch.clamp(edge_data, min=0)

    
    num_nodes = x.shape[0]
    if (edge_data >= num_nodes).any():
        print(f"警告: 边索引超出节点范围(0-{num_nodes - 1})，进行修正...")
        edge_data = torch.clamp(edge_data, max=num_nodes - 1)

    
    edge_data = edge_data.contiguous()

    return Data(x=x, edge_index=edge_data)


class EnhancedGNN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, 256)
        self.conv2 = GATConv(256, 128, heads=4, concat=True)
        self.conv3 = SAGEConv(128 * 4, 128)
        self.conv4 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        ))

        
        self.jk = JumpingKnowledge(mode='cat')
        self.lin_jk = torch.nn.Linear(256 + 128 * 4 + 128 + 128, 256)

        
        self.res_block1 = self._make_residual_block(256, 256)
        self.res_block2 = self._make_residual_block(256, 256)

        
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 50),  
            torch.nn.BatchNorm1d(50)
        )

        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(50, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256)
        )

        
        self.lin_out1 = torch.nn.Linear(256, 128)
        self.lin_out2 = torch.nn.Linear(128, input_dim)

        
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128 * 4)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn_jk = torch.nn.BatchNorm1d(256)
        self.bn_res = torch.nn.BatchNorm1d(256)

        
        self.res_linear = torch.nn.Linear(input_dim, input_dim)

        
        self.use_checkpoint = True

    def _make_residual_block(self, in_dim, out_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(out_dim),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(out_dim)
        )

    def forward(self, x, adj_t):
        
        x0 = x

        
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            x1 = checkpoint(self._forward_conv1, x, adj_t)
        else:
            x1 = self._forward_conv1(x, adj_t)

        
        if self.use_checkpoint and self.training:
            x2 = checkpoint(self._forward_conv2, x1, adj_t)
        else:
            x2 = self._forward_conv2(x1, adj_t)

        
        if self.use_checkpoint and self.training:
            x3 = checkpoint(self._forward_conv3, x2, adj_t)
        else:
            x3 = self._forward_conv3(x2, adj_t)

        
        if self.use_checkpoint and self.training:
            x4 = checkpoint(self._forward_conv4, x3, adj_t)
        else:
            x4 = self._forward_conv4(x3, adj_t)

        
        jk_input = [x1, x2, x3, x4]
        x_jk = self.jk(jk_input)
        x_jk = F.relu(self.lin_jk(x_jk))
        x_jk = self.bn_jk(x_jk)
        x_jk = F.dropout(x_jk, p=0.2, training=self.training)

        
        x_res = self.res_block1(x_jk)
        x_res = x_res + x_jk  
        x_res = self.res_block2(x_res)
        x_res = x_res + x_jk  
        x_res = self.bn_res(x_res)

        
        z_local = self.projector(x_res)  

        
        x_dec = self.decoder(z_local)

        
        x_out = F.relu(self.lin_out1(x_dec))
        x_out = self.lin_out2(x_out)

        
        res_connection = self.res_linear(x0)
        if x_out.device != res_connection.device:
            res_connection = res_connection.to(x_out.device)
        x_out = x_out + res_connection

        return x_out, z_local  

    def _forward_conv1(self, x, adj_t):
        x = F.relu(self.conv1(x, adj_t))
        return self.bn1(x)

    def _forward_conv2(self, x, adj_t):
        x = F.elu(self.conv2(x, adj_t))
        return self.bn2(x)

    def _forward_conv3(self, x, adj_t):
        x = F.relu(self.conv3(x, adj_t))
        return self.bn3(x)

    def _forward_conv4(self, x, adj_t):
        x = F.relu(self.conv4(x, adj_t))
        return self.bn4(x)


def train_model(expr, edges, num_epochs=200):
    
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

    
    def calculate_metrics(y_true, y_pred):
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'pearson': pearsonr(y_true, y_pred)[0]
        }
        return metrics

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    
    torch.cuda.empty_cache()
    print(
        f"初始GPU内存: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB / {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    try:
        
        print("开始准备数据...")
        data = prepare_data(expr, edges)

        
        print(f"节点数: {data.num_nodes}")
        print(f"边数: {data.edge_index.shape[1]}")

        
        from torch_geometric.utils import remove_self_loops, coalesce
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index = coalesce(data.edge_index)
        print(f"优化后边数: {data.edge_index.shape[1]}")

        
        from torch_sparse import SparseTensor
        adj = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        data.adj_t = adj.t()

        
        
        model = EnhancedGNN(input_dim=data.num_features)
        model = model.to(device)

        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数: {total_params:,}")

        
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)

        
        torch.cuda.empty_cache()
        print(f"数据加载后内存: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0005,
            weight_decay=1e-4
        )

        
        history = {
            'epoch': [],
            'train_loss': [],
            'train_mse': [],
            'train_rmse': [],
            'train_r2': [],
            'train_pearson': [],
            'lr': []
        }

        best_loss = float('inf')
        patience = 20
        no_improve = 0

        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        
        print("开始全图训练...")

        
        accumulation_steps = 8  

        
        final_z_local = None

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            
            recon, z_local = model(data.x, data.adj_t)  

            loss = F.mse_loss(recon, data.x)

            
            loss = loss / accumulation_steps
            loss.backward()

            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            
            avg_loss = loss.item() * accumulation_steps
            history['train_loss'].append(avg_loss)
            history['epoch'].append(epoch + 1)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            
            
            model.eval()
            with torch.no_grad():
                
                recon, z_local = model(data.x, data.adj_t)

                
                if epoch == num_epochs - 1 or no_improve >= patience - 1:
                    final_z_local = z_local.detach().cpu().numpy()

                
                preds = recon.cpu().numpy()
                targets = data.x.cpu().numpy()

                
                train_metrics = calculate_metrics(targets, preds)

            
            history['train_mse'].append(train_metrics['mse'])
            history['train_rmse'].append(train_metrics['rmse'])
            history['train_r2'].append(train_metrics['r2'])
            history['train_pearson'].append(train_metrics['pearson'])

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                
                feature_dim = final_z_local.shape[1] if final_z_local is not None else "N/A"
                print(f'Epoch {epoch + 1:03d}, Loss: {avg_loss:.4f}, '
                      f'MSE: {train_metrics["mse"]:.4f}, RMSE: {train_metrics["rmse"]:.4f}, '
                      f'R²: {train_metrics["r2"]:.4f}, Pearson: {train_metrics["pearson"]:.4f}, '
                      f'Feature Dim: {feature_dim}')

            
            scheduler.step(avg_loss)

            
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'早停在epoch {epoch + 1}')
                    
                    break

        
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history
        }, 'final_model_X.pth')

        
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.to_csv('Mouse_training_history.csv', index=False)
        print("训练历史已保存为CSV文件")

        
        if final_z_local is None:
            with torch.no_grad():
                _, z_local = model(data.x, data.adj_t)
                final_z_local = z_local.detach().cpu()

        
        
        torch.save(final_z_local,'Lung_z_local_50d.pt')
        print(f"✅ 50维特征已保存: {final_z_local.shape}")

        return model, history, final_z_local  

    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

        
        if device.type == 'cuda':
            print("尝试在CPU上运行...")
            device = torch.device('cpu')
            
            model = EnhancedGNN(input_dim=data.num_features).to(device)
            
        raise