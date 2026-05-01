import torch
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroGraphSAGE
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.nn import HeteroConv, LayerNorm, GATConv

from .model import PositionalEmbedding, MLPDiffusion
from .transformer import Transformer, Tokenizer, Reconstructor


class HeteroGAT(torch.nn.Module):
    """
    Implementation of heterogeneous GAT.
    """

    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        (channels, channels), channels, heads=1, add_self_loops=False
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, torch.Tensor],
        edge_index_dict: dict[NodeType, torch.Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> dict[NodeType, torch.Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class Embedding(nn.Module):
    def __init__(self, dim_t, max_positions=10000):
        super().__init__()
        self.dim_t = dim_t
        self.embedding = PositionalEmbedding(
            num_channels=dim_t, max_positions=max_positions
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

    def forward(self, timesteps):
        emb = self.embedding(timesteps)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        emb = self.mlp(emb)
        return emb


class NoOp(torch.nn.Module):
    def forward(self, input: torch.Tensor, *args: any, **kwargs: any) -> torch.Tensor:
        return input


class GraphDiff(nn.Module):
    def __init__(
        self,
        d_numerical_dict,
        categories_dict,
        gnn_params,
        dim_t=128,
        model_dim=1024,
        transformer_layers=2,
        d_token=4,
        n_head=1,
        factor=32,
        bias=True,
        zero_init=True,
        proportions_dict: dict[str, list[torch.Tensor]] | None = None,
        order_enc=None,
        use_transformers=True,
    ) -> None:
        super().__init__()

        self.d_numerical_dict = d_numerical_dict
        self.categories_dict = categories_dict
        self.num_features_dict = dict()
        self.token_dim = d_token
        self.order_enc = order_enc
        self.dim_t = dim_t
        self.use_transformers = use_transformers

        self.projs = nn.ModuleDict()
        self.noise_embeds = nn.ModuleDict()
        self.mlp_diff_dict = nn.ModuleDict()
        # TabDiff transformer encoder
        self.tokenizers = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        # TabDiff transformer decoder
        self.decoders = nn.ModuleDict()
        self.detokenizers = nn.ModuleDict()
        # Positional encoding
        if self.order_enc is not None:
            self.order_encoders = nn.ModuleDict()

        for table_name in d_numerical_dict.keys():
            d_numerical = d_numerical_dict[table_name]
            categories = categories_dict[table_name]
            num_features = d_numerical + len(categories)
            self.num_features_dict[table_name] = num_features

            # TabDiff transformer encoder
            self.tokenizers[table_name] = Tokenizer(
                d_numerical, categories, d_token, bias=bias
            )
            self.encoders[table_name] = Transformer(
                transformer_layers, d_token, n_head, d_token, factor
            )
            # ignore the first CLS token.
            d_in = num_features * d_token
            self.projs[table_name] = nn.Linear(d_in, dim_t)
            self.noise_embeds[table_name] = Embedding(dim_t)

            # Positional encoding
            if self.order_enc is not None and table_name in self.order_enc:
                self.order_encoders[table_name] = Embedding(dim_t, max_positions=100)

            if d_numerical + len(categories) == 0:
                # raise NotImplementedError("Foreign-key only table")
                self.mlp_diff_dict[table_name] = NoOp()  # Foreign-key only table
            else:
                self.mlp_diff_dict[table_name] = MLPDiffusion(
                    d_in=dim_t,
                    dim_t=model_dim,
                    d_out=d_in,
                    use_mlp=True,
                )

            # TabDiff transformer decoder
            self.decoders[table_name] = Transformer(
                transformer_layers, d_token, n_head, d_token, factor
            )
            if proportions_dict is not None and len(proportions_dict[table_name]) > 0:
                proportions = torch.cat(proportions_dict[table_name], dim=0)
                log_proportions = proportions.log()
                # set negative infinity (mask state proportions to 0)
                log_proportions[log_proportions == -float("inf")] = 0
                bias_init = log_proportions
            else:
                bias_init = None

            self.detokenizers[table_name] = Reconstructor(
                d_numerical,
                categories,
                d_token,
                zero_init=zero_init,
                bias_init=bias_init,
            )

        gnn_type = gnn_params.pop("type", "HeteroGraphSAGE")
        if gnn_type == "HeteroGraphSAGE":
            self.gnn = HeteroGraphSAGE(
                channels=dim_t,
                **gnn_params,
            )
        elif gnn_type == "HeteroGAT":
            self.gnn = HeteroGAT(
                channels=dim_t,
                **gnn_params,
            )
        elif gnn_type == "no_gnn":
            self.gnn = NoOp()
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x_num_dict, x_cat_dict, time_dict, batch):
        # Transformer encoder
        x_in = dict()
        for table, time in time_dict.items():
            if time.shape[0] == 0:
                x_in[table] = torch.zeros((0, self.dim_t), device=time.device)
                continue
            emb = self.noise_embeds[table](time)
            if self.order_enc is not None and table in self.order_enc:
                emb += self.order_encoders[table](batch[table].order)
            # Tokenize the input
            e = self.tokenizers[table](x_num_dict[table], x_cat_dict[table])
            encoder_input = e[:, 1:, :]  # ignore the first CLS token.
            if self.use_transformers:
                x = self.encoders[table](encoder_input)
            else:
                x = encoder_input
            x = self.projs[table](x.reshape(x.shape[0], -1))
            x_in[table] = x + F.silu(emb)

        # GNN message passing
        gnn_features = self.gnn(x_in, batch.edge_index_dict)

        x_num_out_dict = dict()
        x_cat_out_dict = dict()
        for table_name, gnn_out in gnn_features.items():
            mask = torch.isin(batch[table_name].n_id, batch[table_name].input_id)

            timesteps = time_dict[table_name][mask]
            target_nodes_features = (gnn_out[mask] + x_in[table_name][mask]) * 0.5

            if mask.sum() == 0:
                x_num_out_dict[table_name] = torch.zeros(
                    (0, x_num_dict[table_name].shape[1]), device=timesteps.device
                )
                x_cat_out_dict[table_name] = torch.zeros(
                    (0, x_cat_dict[table_name].shape[1]), device=timesteps.device
                )
                continue
            # Apply Diffusion backbone
            pred_y = self.mlp_diff_dict[table_name](target_nodes_features, timesteps)

            # Transformer decoder (TabDiff)
            if self.use_transformers:
                pred_e = self.decoders[table_name](
                    pred_y.reshape(
                        -1, self.num_features_dict[table_name], self.token_dim
                    )
                )
            else:
                pred_e = pred_y.reshape(
                    -1, self.num_features_dict[table_name], self.token_dim
                )

            # Detokenize the output
            x_num_pred, x_cat_pred = self.detokenizers[table_name](pred_e)
            x_cat_pred = (
                torch.cat(x_cat_pred, dim=-1)
                if len(x_cat_pred) > 0
                else torch.zeros_like(x_cat_dict[table_name][mask]).to(x_num_pred.dtype)
            )

            x_num_out_dict[table_name] = x_num_pred
            x_cat_out_dict[table_name] = x_cat_pred

        return x_num_out_dict, x_cat_out_dict


class PrecondMulti(nn.Module):
    def __init__(
        self,
        denoise_fn,
        sigma_data=0.5,  # Expected standard deviation of the training data.
        net_conditioning="sigma",
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num_dict, x_cat_dict, t_dict, batch, sigma_dict):
        x_in_dict = dict()
        c_skip_dict = dict()
        c_out_dict = dict()
        c_noise_dict = dict()
        mask_dict = dict()
        for table_name, x_num in x_num_dict.items():
            x_num = x_num.to(torch.float32)
            sigma = sigma_dict[table_name].to(torch.float32)
            t = t_dict[table_name]

            assert sigma.ndim == 2
            if (
                sigma.dim() > 1
            ):  # if learnable column-wise noise schedule, sigma conditioning is set to the defaults schedule of rho=7
                sigma_cond = (
                    0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))
                ).pow(7)
            else:
                sigma_cond = sigma
            dtype = torch.float32

            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
            c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
            c_noise = sigma_cond.log() / 4

            mask = torch.isin(batch[table_name].n_id, batch[table_name].input_id)
            mask_dict[table_name] = mask
            c_skip_dict[table_name] = c_skip[mask]
            c_out_dict[table_name] = c_out[mask]
            x_in_dict[table_name] = c_in * x_num
            c_noise_dict[table_name] = c_noise.flatten()

        if self.net_conditioning == "sigma":
            F_x_dict, x_cat_pred_dict = self.denoise_fn_F(
                x_in_dict, x_cat_dict, c_noise_dict, batch=batch
            )
        elif self.net_conditioning == "t":
            F_x_dict, x_cat_pred_dict = self.denoise_fn_F(
                x_in_dict, x_cat_dict, t_dict, batch=batch
            )

        D_x_dict = dict()
        for table_name, x_num in x_num_dict.items():
            mask = mask_dict[table_name]
            F_x = F_x_dict[table_name]
            c_skip = c_skip_dict[table_name]
            c_out = c_out_dict[table_name]
            assert F_x.dtype == dtype
            D_x_dict[table_name] = c_skip * x_num[mask] + c_out * F_x.to(torch.float32)

        return D_x_dict, x_cat_pred_dict


class ModelJoint(nn.Module):
    def __init__(
        self,
        denoise_fn,
        sigma_data=0.5,
        precond=False,
        net_conditioning="sigma",
        **kwargs,
    ):
        super().__init__()
        self.precond = precond
        if precond:
            self.denoise_fn_D = PrecondMulti(
                denoise_fn, sigma_data=sigma_data, net_conditioning=net_conditioning
            )
        else:
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t, batch, sigma=None):
        if self.precond:
            return self.denoise_fn_D(x_num, x_cat, t, batch, sigma)
        else:
            return self.denoise_fn_D(x_num, x_cat, t, batch)
