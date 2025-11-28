import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv

from .bert import BERT


class FinePoiModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embed_dim % config.n_heads == 0

        self.n_layers = config.n_layers
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads

        self.poi_convs = nn.ModuleList([
            GATv2Conv(
                in_channels=config.embed_dim,
                out_channels=config.embed_dim//config.n_heads,
                heads=config.n_heads,
                concat=True,
            ) for _ in range(config.n_layers)
        ])

        self.lns = nn.ModuleList([
            nn.LayerNorm(config.embed_dim) for _ in range(config.n_layers)
        ])

        self.act = nn.ReLU()

    def forward(self, fine_poi_x, edge_index):
        for i in range(self.n_layers):
            fine_poi_x = fine_poi_x + self.poi_convs[i](x=fine_poi_x, edge_index=edge_index)
            fine_poi_x = self.lns[i](fine_poi_x)
            if i != self.n_layers - 1:
                fine_poi_x = self.act(fine_poi_x)

        return fine_poi_x


class CoarsePoiModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_grid_x = config.num_grid_x
        self.num_grid_y = config.num_grid_y
        self.num_poi_cats = config.num_poi_cats

        self.poi_conv = nn.Conv2d(
            in_channels=config.num_poi_cats*config.embed_dim,
            out_channels=config.num_poi_cats*config.embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=config.num_poi_cats,
            bias=False,
        )

        self.mask_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad = False

        self.ln = nn.LayerNorm(config.embed_dim)

    def forward(self, grid_poi_x, grid_poi_mask):
        assert self.num_grid_y * self.num_grid_x == grid_poi_x.size(0)
        assert self.num_poi_cats == grid_poi_x.size(1)

        grid_poi_x = grid_poi_x.view(self.num_grid_y, self.num_grid_x, self.num_poi_cats, self.embed_dim)
        grid_poi_x = grid_poi_x.permute(2, 3, 0, 1)
        grid_poi_x = grid_poi_x.contiguous().view(1, self.num_poi_cats*self.embed_dim, self.num_grid_y, self.num_grid_x)
        grid_poi_x_out = self.poi_conv(grid_poi_x)
        grid_poi_x_out = grid_poi_x_out.view(self.num_poi_cats, self.embed_dim, self.num_grid_y, self.num_grid_x)
        grid_poi_x_out = grid_poi_x_out.permute(2, 3, 0, 1)
        grid_poi_x_out = grid_poi_x_out.contiguous().view(self.num_grid_y*self.num_grid_x, self.num_poi_cats, self.embed_dim)

        grid_poi_mask = grid_poi_mask.float()
        grid_poi_mask = grid_poi_mask.view(self.num_grid_y, self.num_grid_x, self.num_poi_cats)
        grid_poi_mask = grid_poi_mask.permute(2, 0, 1)
        grid_poi_mask = grid_poi_mask.unsqueeze(1)
        grid_poi_mask_out = self.mask_conv(grid_poi_mask)
        grid_poi_mask_out = grid_poi_mask_out.squeeze(1)
        grid_poi_mask_out = grid_poi_mask_out.permute(1, 2, 0)
        grid_poi_mask_out = grid_poi_mask_out.contiguous().view(self.num_grid_y*self.num_grid_x, self.num_poi_cats)

        grid_poi_x_out = torch.sum(grid_poi_x_out, dim=1) / torch.sum(grid_poi_mask_out, dim=1, keepdim=True).clamp_min(1)
        grid_poi_x_out = self.ln(grid_poi_x_out)

        return grid_poi_x_out


class MoEGate(nn.Module):
    def __init__(self, gating_dim, n_routed_experts, top_k, update_rate):
        super().__init__()

        self.gating_dim = gating_dim
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.update_rate = update_rate

        self.gate = nn.Linear(self.gating_dim, self.n_routed_experts)
        self.expert_biases = nn.Parameter(torch.zeros(self.n_routed_experts), requires_grad=False)

    def forward(self, hidden_states, mask):
        B, T, C = hidden_states.size()

        hidden_states = hidden_states.view(-1, C)
        mask = mask.view(-1)

        gate_output = self.gate(hidden_states)
        gate_probs = torch.sigmoid(gate_output)

        gate_logits = gate_output + self.expert_biases
        _, top_k_indices = torch.topk(gate_logits, k=self.top_k, dim=-1, sorted=False)

        top_k_probs = gate_probs.gather(-1, top_k_indices)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        with torch.no_grad():
            expert_counts = torch.bincount(top_k_indices[mask].flatten(), minlength=self.n_routed_experts).float()
            max_count = expert_counts.max()
            avg_count = expert_counts.mean()
            maxvio = (max_count - avg_count) / (avg_count + 1e-5)

            if self.training and self.update_rate > 0:
                error = avg_count - expert_counts.float()
                self.expert_biases += self.update_rate * torch.sign(error)

        return top_k_indices, top_k_probs, maxvio


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w3(self.act(self.w1(x)) * self.w2(x))


class RouteChoiceModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.moe_n_routed_experts = config.moe_n_routed_experts
        self.moe_top_k = config.moe_top_k
        self.moe_update_rate = config.moe_update_rate

        self.route_choice_travel_progress_proj = nn.Linear(1, config.embed_dim)
        self.route_choice_angle_proj = nn.Linear(2, config.embed_dim)
        self.route_choice_trans_prob_proj = nn.Linear(1, config.embed_dim)

        self.q_proj = nn.Linear(config.embed_dim*2, config.embed_dim)

        self.moe_gate = MoEGate(gating_dim=config.embed_dim*2, n_routed_experts=config.moe_n_routed_experts, top_k=config.moe_top_k, update_rate=config.moe_update_rate)
        self.routed_experts = nn.ModuleList([SwiGLU(input_dim=config.embed_dim*3, hidden_dim=config.embed_dim*4, output_dim=config.embed_dim*2) for _ in range(config.moe_n_routed_experts)])
        self.shared_experts = SwiGLU(input_dim=config.embed_dim*3, hidden_dim=config.embed_dim*4, output_dim=config.embed_dim*2)

        self.selected_and_unselected_fusion_mlp = SwiGLU(input_dim=config.embed_dim*2, hidden_dim=config.embed_dim*4, output_dim=config.embed_dim)

    def forward(self, traj_feat, traj_len, adj_feat, route_choice_travel_progress, route_choice_angle, route_choice_trans_prob, route_choice_selected_mask, route_choice_unselected_mask):
        B, T1, T2, C = adj_feat.size()
        assert traj_feat.size() == (B, T1, C)

        current_intersection_feat = torch.concat([
            traj_feat,
            self.route_choice_travel_progress_proj(route_choice_travel_progress.unsqueeze(-1)),
        ], dim=-1)
        adj_choice_feat = torch.concat([
            adj_feat,
            self.route_choice_angle_proj(torch.concat([torch.sin(route_choice_angle.unsqueeze(-1)*math.pi), torch.cos(route_choice_angle.unsqueeze(-1)*math.pi)], dim=-1)),
            self.route_choice_trans_prob_proj(route_choice_trans_prob.unsqueeze(-1)),
        ], dim=-1)

        mask = (torch.arange(T1, dtype=torch.int64, device=traj_len.device).unsqueeze(0) < traj_len.unsqueeze(1))

        topk_idx, topk_weight, maxvio = self.moe_gate(current_intersection_feat, mask)
        topk_idx = topk_idx.unsqueeze(1).expand(-1, T2, -1).contiguous().view(B*T1*T2, self.moe_top_k)
        topk_weight = topk_weight.unsqueeze(1).expand(-1, T2, -1).contiguous().view(B*T1*T2, self.moe_top_k)

        identity = adj_choice_feat

        adj_choice_feat = adj_choice_feat.view(B*T1*T2, -1)
        adj_choice_feat = adj_choice_feat.repeat_interleave(self.moe_top_k, dim=0)
        flat_topk_idx = topk_idx.view(-1)

        sorted_flat_topk_idx = torch.argsort(flat_topk_idx)
        sorted_adj_choice_feat = adj_choice_feat[sorted_flat_topk_idx]

        tokens_per_expert = torch.bincount(flat_topk_idx, minlength=self.moe_n_routed_experts)
        expert_offsets = torch.cumsum(tokens_per_expert, dim=0) - tokens_per_expert

        sorted_outputs = torch.empty((B*T1*T2*self.moe_top_k, C*2), dtype=adj_choice_feat.dtype, device=adj_choice_feat.device)
        for i, expert in enumerate(self.routed_experts):
            count = tokens_per_expert[i].item()
            if count == 0:
                continue

            start = expert_offsets[i]
            end = start + count

            expert_input = sorted_adj_choice_feat[start:end]
            expert_output = expert(expert_input)

            sorted_outputs[start:end] = expert_output

        y = torch.empty_like(sorted_outputs)
        y[sorted_flat_topk_idx] = sorted_outputs

        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(B, T1, T2, -1)

        y = y + self.shared_experts(identity)

        q = self.q_proj(current_intersection_feat).unsqueeze(2)
        k, v = y.split(self.embed_dim, dim=-1)

        assert torch.all(torch.sum(route_choice_selected_mask, dim=-1) <= 1)

        selected_emb = torch.sum(v * route_choice_selected_mask.float().unsqueeze(-1), dim=2)

        unselected_attn = ((q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))).squeeze(2)
        unselected_attn = unselected_attn.masked_fill(route_choice_unselected_mask.logical_not(), float('-inf'))
        unselected_attn = F.softmax(unselected_attn, dim=-1)
        unselected_attn = unselected_attn.masked_fill(torch.all(route_choice_unselected_mask.logical_not(), dim=-1, keepdim=True), 0.0)

        unselected_emb = (unselected_attn.unsqueeze(2) @ v).squeeze(2)

        out = self.selected_and_unselected_fusion_mlp(torch.concat([selected_emb, unselected_emb], dim=-1))
        return out, maxvio


class Core(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert len(config.road_type) == len(config.road_length_norm) == len(config.road_out_degree) == len(config.road_in_degree) == len(config.road_poi_hidden_states) == config.road_id_padding_idx
        assert len(config.grid_poi_hidden_states) == config.grid_id_padding_idx
        assert config.n_weekday_embed-1 == config.weekday_padding_idx
        assert config.n_time_of_day_embed-1 == config.time_of_day_padding_idx

        self.embed_dim = config.embed_dim

        self.cls_embed = nn.Parameter(torch.randn(1, config.embed_dim))

        self.register_buffer('road_type', config.road_type, persistent=False)
        self.register_buffer('road_length_norm', config.road_length_norm, persistent=False)
        self.register_buffer('road_out_degree', config.road_out_degree, persistent=False)
        self.register_buffer('road_in_degree', config.road_in_degree, persistent=False)

        self.type_embed = nn.Embedding(num_embeddings=config.n_type_embed, embedding_dim=config.embed_dim)
        self.length_embed = nn.Linear(1, config.embed_dim)
        self.out_degree_embed = nn.Embedding(num_embeddings=config.n_out_degree_embed, embedding_dim=config.embed_dim)
        self.in_degree_embed = nn.Embedding(num_embeddings=config.n_in_degree_embed, embedding_dim=config.embed_dim)
        self.road_feat_before_fusion_ln = nn.LayerNorm(config.embed_dim)

        self.register_buffer('road_poi_hidden_states', config.road_poi_hidden_states, persistent=False)
        self.register_buffer('edge_index', config.edge_index, persistent=False)

        self.fine_poi_model = FinePoiModel(config.fine_poi_model_config)

        self.register_buffer('grid_poi_hidden_states', config.grid_poi_hidden_states, persistent=False)
        self.register_buffer('grid_poi_mask', config.grid_poi_mask, persistent=False)

        self.coarse_poi_model = CoarsePoiModel(config.coarse_poi_model_config)

        self.fine_poi_gate = nn.Sequential(
            nn.Linear(config.embed_dim*2, config.embed_dim),
            nn.SiLU()
        )
        self.coarse_poi_gate = nn.Sequential(
            nn.Linear(config.embed_dim*2, config.embed_dim),
            nn.SiLU()
        )

        self.road_feat_after_fusion_ln = nn.LayerNorm(config.embed_dim)

        self.route_choice_model = RouteChoiceModel(config.route_choice_model_config)
        self.spatial_feat_ln = nn.LayerNorm(config.embed_dim)

        self.weekday_embed = nn.Embedding(num_embeddings=config.n_weekday_embed, embedding_dim=config.embed_dim, padding_idx=config.weekday_padding_idx)
        self.time_of_day_embed = nn.Embedding(num_embeddings=config.n_time_of_day_embed, embedding_dim=config.embed_dim, padding_idx=config.time_of_day_padding_idx)
        self.temporal_feat_ln = nn.LayerNorm(config.embed_dim)

        self.bert = BERT(config.bert_config)

        self.projection_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim*4),
            nn.ReLU(),
            nn.Linear(config.embed_dim*4, config.embed_dim)
        )

    def forward(self, traj_road_id, traj_grid_id, adj_road_id, adj_grid_id, route_choice_travel_progress, route_choice_angle, route_choice_trans_prob, route_choice_selected_mask, route_choice_unselected_mask, traj_weekday, traj_time_of_day, traj_len):
        all_road_type_feat = self.type_embed(self.road_type)
        all_road_length_feat = self.length_embed(self.road_length_norm.unsqueeze(-1))
        all_road_out_degree_feat = self.out_degree_embed(self.road_out_degree)
        all_road_in_degree_feat = self.in_degree_embed(self.road_in_degree)

        all_road_feat = self.road_feat_before_fusion_ln(all_road_type_feat + all_road_length_feat + all_road_out_degree_feat + all_road_in_degree_feat)

        all_road_poi_feat = self.fine_poi_model(self.road_poi_hidden_states, self.edge_index)
        all_grid_poi_feat = self.coarse_poi_model(self.grid_poi_hidden_states, self.grid_poi_mask)

        all_road_feat = torch.concat([
            all_road_feat,
            torch.zeros((1, all_road_feat.size(-1)), dtype=all_road_feat.dtype, device=all_road_feat.device),
        ], dim=0)
        all_road_poi_feat = torch.concat([
            all_road_poi_feat,
            torch.zeros((1, all_road_poi_feat.size(-1)), dtype=all_road_poi_feat.dtype, device=all_road_poi_feat.device),
        ], dim=0)
        all_grid_poi_feat = torch.concat([
            all_grid_poi_feat,
            torch.zeros((1, all_grid_poi_feat.size(-1)), dtype=all_grid_poi_feat.dtype, device=all_grid_poi_feat.device),
        ], dim=0)

        traj_road_feat = all_road_feat[traj_road_id]
        traj_road_poi_feat = all_road_poi_feat[traj_road_id]
        traj_grid_poi_feat = all_grid_poi_feat[traj_grid_id]

        adj_road_feat = all_road_feat[adj_road_id]
        adj_road_poi_feat = all_road_poi_feat[adj_road_id]
        adj_grid_poi_feat = all_grid_poi_feat[adj_grid_id]

        traj_feat = traj_road_feat + self.fine_poi_gate(torch.concat([traj_road_feat, traj_road_poi_feat], dim=-1)) * traj_road_poi_feat + self.coarse_poi_gate(torch.concat([traj_road_feat, traj_grid_poi_feat], dim=-1)) * traj_grid_poi_feat
        traj_feat = self.road_feat_after_fusion_ln(traj_feat)

        adj_feat = adj_road_feat + self.fine_poi_gate(torch.concat([adj_road_feat, adj_road_poi_feat], dim=-1)) * adj_road_poi_feat + self.coarse_poi_gate(torch.concat([adj_road_feat, adj_grid_poi_feat], dim=-1)) * adj_grid_poi_feat
        adj_feat = self.road_feat_after_fusion_ln(adj_feat)

        spatial_feat, maxvio = self.route_choice_model(traj_feat, traj_len, adj_feat, route_choice_travel_progress, route_choice_angle, route_choice_trans_prob, route_choice_selected_mask, route_choice_unselected_mask)
        spatial_feat = self.spatial_feat_ln(spatial_feat)

        temporal_feat = self.weekday_embed(traj_weekday) + self.time_of_day_embed(traj_time_of_day)
        temporal_feat = self.temporal_feat_ln(temporal_feat)

        feat = spatial_feat + temporal_feat
        feat = torch.concat([
            self.cls_embed.unsqueeze(0).expand(feat.size(0), -1, -1),
            feat,
        ], dim=1)

        x = self.bert(feat, traj_len+1)

        traj_embedding, hidden_states = x[:, 0, :], x[:, 1:, :]
        return traj_embedding, hidden_states, maxvio

    def pretrain(self, traj_road_id, traj_grid_id, adj_road_id, adj_grid_id, route_choice_travel_progress, route_choice_angle, route_choice_trans_prob, route_choice_selected_mask, route_choice_unselected_mask, traj_weekday, traj_time_of_day, traj_len):
        traj_embedding, hidden_states, maxvio = self.forward(traj_road_id, traj_grid_id, adj_road_id, adj_grid_id, route_choice_travel_progress, route_choice_angle, route_choice_trans_prob, route_choice_selected_mask, route_choice_unselected_mask, traj_weekday, traj_time_of_day, traj_len)
        proj_traj_embedding = self.projection_head(traj_embedding)

        return proj_traj_embedding, hidden_states, maxvio
    
    def get_road_emb(self, road_id, grid_id):
        all_road_type_feat = self.type_embed(self.road_type)
        all_road_length_feat = self.length_embed(self.road_length_norm.unsqueeze(-1))
        all_road_out_degree_feat = self.out_degree_embed(self.road_out_degree)
        all_road_in_degree_feat = self.in_degree_embed(self.road_in_degree)

        all_road_feat = self.road_feat_before_fusion_ln(all_road_type_feat + all_road_length_feat + all_road_out_degree_feat + all_road_in_degree_feat)

        all_road_poi_feat = self.fine_poi_model(self.road_poi_hidden_states, self.edge_index)
        all_grid_poi_feat = self.coarse_poi_model(self.grid_poi_hidden_states, self.grid_poi_mask)

        all_road_feat = torch.concat([
            all_road_feat,
            torch.zeros((1, all_road_feat.size(-1)), dtype=all_road_feat.dtype, device=all_road_feat.device),
        ], dim=0)
        all_road_poi_feat = torch.concat([
            all_road_poi_feat,
            torch.zeros((1, all_road_poi_feat.size(-1)), dtype=all_road_poi_feat.dtype, device=all_road_poi_feat.device),
        ], dim=0)
        all_grid_poi_feat = torch.concat([
            all_grid_poi_feat,
            torch.zeros((1, all_grid_poi_feat.size(-1)), dtype=all_grid_poi_feat.dtype, device=all_grid_poi_feat.device),
        ], dim=0)

        road_feat = all_road_feat[road_id]
        road_poi_feat = all_road_poi_feat[road_id]
        grid_poi_feat = all_grid_poi_feat[grid_id]

        out = road_feat + self.fine_poi_gate(torch.concat([road_feat, road_poi_feat], dim=-1)) * road_poi_feat + self.coarse_poi_gate(torch.concat([road_feat, grid_poi_feat], dim=-1)) * grid_poi_feat
        out = self.road_feat_after_fusion_ln(out)

        return out
    
    def get_road_and_poi_emb(self, road_id, grid_id):
        all_road_type_feat = self.type_embed(self.road_type)
        all_road_length_feat = self.length_embed(self.road_length_norm.unsqueeze(-1))
        all_road_out_degree_feat = self.out_degree_embed(self.road_out_degree)
        all_road_in_degree_feat = self.in_degree_embed(self.road_in_degree)

        all_road_feat = self.road_feat_before_fusion_ln(all_road_type_feat + all_road_length_feat + all_road_out_degree_feat + all_road_in_degree_feat)

        all_road_poi_feat = self.fine_poi_model(self.road_poi_hidden_states, self.edge_index)
        all_grid_poi_feat = self.coarse_poi_model(self.grid_poi_hidden_states, self.grid_poi_mask)

        all_road_feat = torch.concat([
            all_road_feat,
            torch.zeros((1, all_road_feat.size(-1)), dtype=all_road_feat.dtype, device=all_road_feat.device),
        ], dim=0)
        all_road_poi_feat = torch.concat([
            all_road_poi_feat,
            torch.zeros((1, all_road_poi_feat.size(-1)), dtype=all_road_poi_feat.dtype, device=all_road_poi_feat.device),
        ], dim=0)
        all_grid_poi_feat = torch.concat([
            all_grid_poi_feat,
            torch.zeros((1, all_grid_poi_feat.size(-1)), dtype=all_grid_poi_feat.dtype, device=all_grid_poi_feat.device),
        ], dim=0)

        road_feat = all_road_feat[road_id]
        road_poi_feat = all_road_poi_feat[road_id]
        grid_poi_feat = all_grid_poi_feat[grid_id]

        return road_feat, road_poi_feat, grid_poi_feat
