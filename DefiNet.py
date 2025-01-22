# %%
import math
import torch
from torch import nn
from torch_scatter import scatter
from graph_utils import ScaledSiLU, AtomEmbedding, RadialBasis
from graph_utils import vector_norm
from torch_geometric.nn import global_mean_pool

class VirtualScalar(nn.Module):
    def __init__(self, in_feats, out_feats, residual=False):
        super(VirtualScalar, self).__init__()
        # Add residual connection or not
        self.residual = residual

        self.mlp_vn = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU())
        
        self.mlp_node = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU())

    def update_node_emb(self, x, batch, vx):
        h = self.mlp_node(torch.cat([x, vx[batch]], dim=1)) + x

        return h, vx

    def update_vn_emb(self, x, batch, vx):
        vx_temp = self.mlp_vn(torch.cat([global_mean_pool(x, batch), vx], dim=-1)) 

        if self.residual:
            vx = vx + vx_temp
        else:
            vx = vx_temp

        return vx

class VirtualVector(nn.Module):
    def __init__(self, in_feats, out_feats, residual=False):
        super(VirtualVector, self).__init__()
        # Add residual connection or not
        self.residual = residual

        self.mlp_vn = nn.Linear(in_feats, out_feats, bias=False)
        self.mlp_node = nn.Linear(in_feats, out_feats, bias=False)
 
    def update_node_emb(self, vec, batch, vvec):
        hvec = self.mlp_node(vec + vvec[batch]) + vec

        return hvec, vvec

    def update_vn_emb(self, vec, batch, vvec):
        vvec_temp = self.mlp_vn(scatter(vec, batch, dim=0, reduce='mean', dim_size=vvec.size(0)) + vvec) 

        if self.residual:
            vvec = vvec + vvec_temp
        else:
            vvec = vvec_temp

        return vvec

class VectorActivation(nn.Module):
    def __init__(
        self,
        hidden_channels
    ):
        super().__init__()
        self.lin_vec = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.lin_gvec = nn.Linear(hidden_channels, 1, bias=False)

    def forward(self, vec):
        gvec = self.lin_gvec(vec)
        lvec = self.lin_vec(vec)

        dotprod = (lvec * gvec).sum(1, keepdim=True)
        mask = (dotprod >= 0).float()

        vec = mask * lvec + (1 - mask) * (lvec + gvec) / 2

        return vec

class DefiNet(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        num_layers=3,
        num_rbf=128,
        cutoff=6.,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        num_elements=83
    ):
        super(DefiNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.defect_emb = AtomEmbedding(hidden_channels, 3)
        self.lin_comb = nn.Linear(hidden_channels*2, hidden_channels)
        
        self.vn_emb = nn.Embedding(1, hidden_channels)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.virtual_scalar_layers = nn.ModuleList()
        self.virtual_vector_layers = nn.ModuleList()
        self.message_passing_layers = nn.ModuleList()
        self.message_update_layers = nn.ModuleList()
        self.coord_update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.virtual_scalar_layers.append(VirtualScalar(hidden_channels, hidden_channels, residual=True))
            self.virtual_vector_layers.append(VirtualVector(hidden_channels, hidden_channels, residual=True))
            self.message_passing_layers.append(
                MessagePassing(hidden_channels, num_rbf)
            )
            self.message_update_layers.append(MessageUpdating(hidden_channels))
            self.coord_update_layers.append(CoordUpdate(hidden_channels, num_rbf))

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data):
        pos = data.pos_u
        cell= data.cell_u
        batch = data.batch
        defect = data.defect

        cell_offsets = data.cell_offsets.float()
        edge_index = data.edge_index

        neighbors = data.neighbors
        z = data.x.long()
        assert z.dim() == 1 and z.dtype == torch.long

        x_atom = self.atom_emb(z)
        x_defect = self.defect_emb(defect)
        x = self.lin_comb(torch.cat([x_atom, x_defect], dim=-1))
        
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        vx = self.vn_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        vvec = torch.zeros(
                batch[-1].item() + 1, 3, x.size(1)).to(edge_index.dtype).to(edge_index.device)
        #### Interaction blocks ###############################################
        for n in range(self.num_layers):
            j, i = edge_index
            abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)
            pos_diff = pos[j] + torch.einsum("a p, a p v -> a v", cell_offsets, abc_unsqueeze) - pos[i]
            edge_dist = vector_norm(pos_diff, dim=-1)
            edge_vector = -pos_diff / edge_dist.unsqueeze(-1)
            edge_rbf = self.radial_basis(edge_dist)  # rbf * evelope
            edge_feat = edge_rbf

            x, vx = self.virtual_scalar_layers[n].update_node_emb(x, batch, vx)
            vec, vvec = self.virtual_vector_layers[n].update_node_emb(vec, batch, vvec)

            dx, dvec = self.message_passing_layers[n](
                x, x_defect, vec, edge_index, edge_feat, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.message_update_layers[n](x, vec)
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            pos = self.coord_update_layers[n](x, x_defect, vec, pos, edge_index, edge_feat, edge_vector)

            vx = self.virtual_scalar_layers[n].update_vn_emb(x, batch, vx)
            vvec = self.virtual_vector_layers[n].update_vn_emb(vec, batch, vvec)

        return pos
    
class CoordUpdate(nn.Module):
    def __init__(
        self,
        hidden_channels,
        edge_feat_channels
    ):
        super().__init__()

        self.lin_vec2coord = nn.Linear(hidden_channels, 1, bias=False)

        self.defect_proj = nn.Linear(hidden_channels, hidden_channels)
        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.edge_proj = nn.Linear(edge_feat_channels, hidden_channels)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, x_defect, vec, pos, edge_index, edge_feat, edge_vector):
        coord_from_vec = self.lin_vec2coord(vec).squeeze(-1)

        j, i = edge_index
        x_h = self.x_proj(x)
        rbf_h = self.edge_proj(edge_feat)
        defect_h =self.defect_proj(x_defect)

        weighted_diff = self.coord_mlp(x_h[j] * rbf_h * (defect_h[j] + defect_h[i])) * edge_vector
        coord_from_diff = scatter(weighted_diff, i, dim=0, dim_size=x.size(0))

        return pos + (coord_from_vec + coord_from_diff) / 2

class MessagePassing(nn.Module):
    def __init__(
        self,
        hidden_channels,
        edge_feat_channels,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.defect_proj = nn.Linear(hidden_channels, hidden_channels*3)
        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            ScaledSiLU(),
            nn.Linear(hidden_channels//2, hidden_channels*3),
        )
        self.edge_proj = nn.Linear(edge_feat_channels, hidden_channels*3)

        self.vec_activation = VectorActivation(hidden_channels)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

    def forward(self, x, x_defect, vec, edge_index, edge_feat, edge_vector):
        j, i = edge_index

        rbf_h = self.edge_proj(edge_feat)

        x_defect = self.defect_proj(x_defect)

        x_h = self.x_proj(x)
        x_ji1, x_ji2, x_ji3 = torch.split(x_h[j] * (x_defect[j] + x_defect[i]) * rbf_h * self.inv_sqrt_3, self.hidden_channels, dim=-1)

        vec_ji = x_ji1.unsqueeze(1) * vec[j] + x_ji2.unsqueeze(1) * edge_vector.unsqueeze(2)
        vec_ji = vec_ji * self.inv_sqrt_h

        d_vec = scatter(vec_ji, index=i, dim=0, dim_size=x.size(0)) 
        d_vec = self.vec_activation(d_vec)

        d_x = scatter(x_ji3, index=i, dim=0, dim_size=x.size(0))
        
        return d_x, d_vec
    
class MessageUpdating(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels*2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels*3),
        )

        self.vec_activation = VectorActivation(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )

        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        gate = torch.tanh(xvec3)
        dx = xvec2 * self.inv_sqrt_2 + x * gate

        dvec = self.vec_activation(xvec1.unsqueeze(1) * vec1)

        return dx, dvec
    
# %%
