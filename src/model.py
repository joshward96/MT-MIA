import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GATv2Conv, to_hetero, GlobalAttention
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.input_proj = torch.nn.LazyLinear(hidden_channels)
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv3 = GATv2Conv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = self.conv1(x, edge_index).relu()        
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x

def generate_ssl_embeddings(model, data_list, batch_size):
    """
    Generate binary classifier logits for a list of HeteroData graphs.
    """
    model.eval()

    clean_data = []
    for d in data_list:
        if hasattr(d, "y"):
            d = d.clone()
            del d.y
        clean_data.append(d)

    loader = DataLoader(clean_data, batch_size=batch_size, shuffle=False)

    all_z = []
    all_z_target = []
    all_z_context = []

    for batch in loader:
        batch = batch.to(device)
        z, z_target, z_context,_ = model(batch)  
        all_z.append(z.cpu())
        all_z_target.append(z_target.cpu())
        all_z_context.append(z_context.cpu())

    return  torch.cat(all_z, dim=0).numpy(), torch.cat(all_z_target, dim=0).numpy(), torch.cat(all_z_context, dim=0).numpy()


def compute_reconstruction_losses(model, data_list, batch_size, entity_unit, parent_weight=0.5, child_weight=0.5):
    """
    Compute per-sample reconstruction losses for a list of HeteroData graphs.

    Returns three 1-D numpy arrays (total, target, context), one value per graph.
    Lower loss means the model reconstructs that sample more accurately, which is
    the membership signal used by the reconstruction-loss attack.
    """
    model.eval()

    clean_data = []
    for d in data_list:
        if hasattr(d, "y"):
            d = d.clone()
            del d.y
        clean_data.append(d)

    loader = DataLoader(clean_data, batch_size=batch_size, shuffle=False)

    all_target_loss = []
    all_context_loss = []
    all_total_loss = []

    for batch in loader:
        batch = batch.to(device)
        z_final, _, _, _ = model(batch)

        # Per-sample target reconstruction loss
        z_target_recon = model.target_decoder(z_final)
        true_target_feat = batch[entity_unit].x
        loss_target = F.mse_loss(z_target_recon, true_target_feat, reduction='none').mean(dim=-1)

        # Per-sample context reconstruction loss
        target_edges = [et for et in batch.edge_types if et[0] == entity_unit]
        if target_edges and hasattr(model, 'context_decoder'):
            edge_type = target_edges[0]
            dst_type = edge_type[2]
            edge_index = batch[edge_type].edge_index
            item_features = batch[dst_type].x
            true_item_sum = torch.zeros(z_final.size(0), item_features.size(1), device=device)
            true_item_sum.index_add_(0, edge_index[0], item_features[edge_index[1]])
            z_context_recon = model.context_decoder(z_final)
            loss_context = F.mse_loss(z_context_recon, true_item_sum, reduction='none').mean(dim=-1)
        else:
            loss_context = torch.zeros(z_final.size(0), device=device)

        loss_total = parent_weight * loss_target + child_weight * loss_context

        all_target_loss.append(loss_target.detach().cpu())
        all_context_loss.append(loss_context.detach().cpu())
        all_total_loss.append(loss_total.detach().cpu())

    return (
        torch.cat(all_total_loss).numpy(),
        torch.cat(all_target_loss).numpy(),
        torch.cat(all_context_loss).numpy(),
    )
        
class sslModel(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, target_node_type, sample_batch):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.target_node_type = target_node_type
        self.node_types, self.edge_types = metadata

        target_feat_dim = sample_batch[target_node_type].x.size(1)
        target_edge = [et for et in self.edge_types if et[0] == target_node_type][0]
        context_node_type = target_edge[2]
        context_feat_dim = sample_batch[context_node_type].x.size(1)
        
        print(f"Inferred Dims -> Target: {target_feat_dim}, Context: {context_feat_dim}")
        # 1. GNN Layers
        self.convs = nn.ModuleList()
        for _ in range(2):
            conv = GNNEncoder(hidden_channels, hidden_channels) 
            conv = to_hetero(conv, metadata, aggr='sum')
            self.convs.append(conv)

        # 2. Context Pooling (Attention based) - Excludes target
        self.attention_pools = nn.ModuleDict({
            node_type: GlobalAttention(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1)
            ))
            for node_type in self.node_types if node_type != target_node_type
        })

        # 3. Normalization and Projection
        self.target_norm = nn.LayerNorm(hidden_channels)
        self.context_norm = nn.LayerNorm(hidden_channels)
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_channels, hidden_channels) 
        )
        self.context_transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid() 
)
        # 4. The Structural Anchor (Decoder)
        self.target_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, target_feat_dim) 
        )

        # Reconstructs the sum of the Items' features
        self.context_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, context_feat_dim)
        )

    def forward(self, data):
        x_dict = data.x_dict
        try:
            edge_index_dict = data.edge_index_dict
        except KeyError:
            edge_index_dict = {}

        # Stage 1: Message Passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: v.relu() for k, v in x_dict.items()}

        # Stage 2: Target Signal Isolation
        z_target = self.target_norm(x_dict[self.target_node_type])

        # Stage 3: Context Signal Isolation
        target_batch = data[self.target_node_type].batch
        batch_size = int(target_batch.max().item() + 1)
        
        context_list = []
        for node_type, pool in self.attention_pools.items():
            if node_type in x_dict and x_dict[node_type].size(0) > 0:
                pooled = pool(x_dict[node_type], data[node_type].batch)
                context_list.append(pooled)
            else:
                context_list.append(torch.zeros(batch_size, self.hidden_channels, device=z_target.device))
        
        z_context_graph = self.context_norm(torch.stack(context_list).sum(dim=0))
        z_context = z_context_graph[target_batch]

        # Stage 4: Fusion and Projection
        z_c_transformed = self.context_transform(z_context)

        # 2. Calculate the gate (how much do we trust the context for this specific user?)
        gate_input = torch.cat([z_target, z_context], dim=-1)
        g = self.gate(gate_input)

        # 3. Apply the Gate and a Residual Connection
        z_final = z_target + (g * z_c_transformed)
        return z_final, z_target, z_context, x_dict


class sslModelNoContext(torch.nn.Module):
    """
    Ablation: MLP autoencoder over parent (target) node features only.

    Mirrors the depth and hidden dimension of sslModel but replaces each
    GATv2Conv with a plain Linear layer, removes context pooling and the
    gated fusion, and keeps only the target reconstruction decoder.

    Mapping to sslModel structure
    ─────────────────────────────
    sslModel has 2 × GNNEncoder blocks.  Each GNNEncoder is:
        LazyLinear(H)          ← input projection
        Linear(H,H) + ReLU     ← conv1
        Linear(H,H)            ← conv2
        Linear(H,H)            ← conv3
    followed by an outer ReLU from sslModel.forward.

    sslModelNoContext replaces each block with the equivalent MLP sequence,
    then applies LayerNorm (= target_norm) and the same target_decoder.

    Return signature matches sslModel: (z_final, z_target, z_context, {})
    so generate_ssl_embeddings and compute_reconstruction_losses work
    without modification.
    """

    def __init__(self, hidden_channels: int, target_node_type: str, target_feat_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.target_node_type = target_node_type

        H = hidden_channels

        # 2 × GNNEncoder blocks, each: input_proj + conv1(relu) + conv2 + conv3 + outer relu
        self.encoder = nn.Sequential(
            # Block 1
            nn.LazyLinear(H),           # input_proj
            nn.Linear(H, H), nn.ReLU(), # conv1
            nn.Linear(H, H),            # conv2
            nn.Linear(H, H), nn.ReLU(), # conv3 + outer relu
            # Block 2
            nn.Linear(H, H),            # input_proj
            nn.Linear(H, H), nn.ReLU(), # conv1
            nn.Linear(H, H),            # conv2
            nn.Linear(H, H), nn.ReLU(), # conv3 + outer relu
        )

        self.target_norm = nn.LayerNorm(H)

        # Identical structure to sslModel.target_decoder
        self.target_decoder = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, target_feat_dim),
        )

    def forward(self, data):
        x = data[self.target_node_type].x
        z = self.encoder(x)
        z_final = self.target_norm(z)
        z_context = torch.zeros_like(z_final)
        return z_final, z_final, z_context, {}
