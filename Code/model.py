import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv

class Contact_MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, attn_dropout=0.1):
        super(Contact_MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.eps = 1e-6
        self.log_lambda = nn.Parameter(torch.tensor(-2.0))

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, esm_contact, mask):

        B, N, d = x.shape

        Q = self.W_q(x)  # [B, N, d]
        K = self.W_k(x)  # [B, N, d]
        V = self.W_v(x)  # [B, N, d]

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d_h]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d_h]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d_h]

        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, h, N, N]

        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]
        esm_contact = esm_contact.clamp(min=0.0, max=1.0)
        esm_contact = esm_contact.unsqueeze(1)

        lambda_ = F.softplus(self.log_lambda)
        bias = torch.clamp(torch.log(esm_contact + self.eps), min =-5.0)
        bias = bias - bias.mean(dim=-1, keepdim=True)
        scores_content = scores / math.sqrt(self.head_dim)
        bias_term = lambda_ * bias
        scores = scores_content + bias_term

        scores = scores.masked_fill(mask_2d.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)  # [B, h, N, N]

        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, V)  # [B, h, N, d_h]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)

        attn_out = self.out_proj(attn_out)  # [B, N, d]
        attn_out = attn_out * mask.unsqueeze(-1).to(attn_out.dtype)

        return attn_out



class Contact_Attention_Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = Contact_MultiHeadAttention(hidden_dim, num_heads, attn_dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, contacts, mask):
        # Attention + residual
        x = x + self.dropout1(
            self.attn(self.norm1(x), contacts, mask)
        )
        # FFN + residual
        x = x + self.dropout2(
            self.ffn(self.norm2(x))
        )
        return x



class Contact_Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.attention_block1 = Contact_Attention_Block(hidden_dim, num_heads, dropout, attn_dropout)
        self.attention_block2 = Contact_Attention_Block(hidden_dim, num_heads, dropout, attn_dropout)

    def forward(self, x, contacts, mask):
        out = self.linear1(x)
        out = self.attention_block1(out, contacts, mask)
        out = self.attention_block2(out, contacts, mask)

        return out



class attn_model(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.contact_attention = Contact_Attention(embed_dim, hidden_dim, num_heads, dropout, attn_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64)
        )

        prior = 0.421711541759561
        init_bias = math.log(prior/(1-prior))
        with torch.no_grad():
            self.classifier[-1].bias.fill_(init_bias)



    def forward(self, x, contacts, mask):

        out = self.contact_attention(x, contacts, mask)

        mask_f = mask.unsqueeze(-1).to(out.dtype)              # [B,N,1]
        K_feat = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        logits = self.classifier(K_feat).squeeze(-1)  # [B]
        return logits



class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_layers=2, dropout=0.3,
                 heads=4, gat_concat=True, att_dropout=0.0, L=35, center_local=17):
        super().__init__()
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.num_layers = num_layers
        self.dropout = dropout
        self.L = L
        self.center_local = center_local

        out_dim = hidden_channels
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if gat_concat:
                if hidden_channels % heads == 0:
                    out_per_head = hidden_channels // heads
                else:
                    out_per_head = hidden_channels
                conv = GATv2Conv(out_dim, out_per_head, heads=heads, concat=True, dropout=att_dropout)
                out_dim = out_per_head * heads
            else:
                conv = GATv2Conv(out_dim, hidden_channels, heads=heads, concat=False, dropout=att_dropout)
                out_dim = hidden_channels
            self.convs.append(conv)

        self.project = nn.Linear(out_dim, hidden_channels) if out_dim != hidden_channels else None

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64)
        )

    def forward(self, x, edge_index, batch, node_mask):
        x = self.input_lin(x)
        layer_features = []

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x * node_mask.unsqueeze(-1).float()
            layer_features.append(x)


        node_repr = torch.stack(layer_features, dim=0).mean(dim=0)

        if self.project is not None:
            node_repr = self.project(node_repr)

        batch_size = int(batch.max().item() + 1)
        device = node_repr.device
        graph_idx = torch.arange(batch_size, device=device)
        center_global_idx = graph_idx * self.L + self.center_local
        center_feats = node_repr[center_global_idx]

        out = self.mlp(center_feats)
        return out.squeeze(1)
    


# Ensemble
class ensemble_model(nn.Module):
    def __init__(self, GAT_in_channels, Atten_embed_dim, Atten_hidden_dim,
                 GAT_hidden_channels=512, GAT_num_layers=2, GAT_dropout=0.3,
                 GAT_heads=4, GAT_gat_concat=True, GAT_att_dropout=0.0, GAT_L=35, GAT_center_local=17,
                 Atten_num_heads=4, Atten_dropout=0.1, Atten_attn_dropout=0.1):
        super().__init__()
        self.gat_model = GAT(GAT_in_channels, GAT_hidden_channels, GAT_num_layers, GAT_dropout,
                             GAT_heads, GAT_gat_concat, GAT_att_dropout, GAT_L, GAT_center_local)
        self.atten_model = attn_model(Atten_embed_dim, Atten_hidden_dim, Atten_num_heads, Atten_dropout, Atten_attn_dropout)
        
        self.linear_g = nn.Sequential(nn.ReLU(),
            nn.Dropout(GAT_dropout),
            nn.Linear(64, 1))
        self.linear_a = nn.Sequential(nn.ReLU(),
            nn.Dropout(Atten_dropout),
            nn.Linear(64, 1))

        self.sigmoid = nn.Sigmoid()

        self.s_g = nn.Parameter(torch.tensor(1.0))
        self.s_a = nn.Parameter(torch.tensor(1.0))


    def forward(self, gat_x, gat_edge_index, gat_batch, gat_node_mask,
                atten_x, atten_contacts, atten_mask):
        g = self.gat_model(gat_x, gat_edge_index, gat_batch, gat_node_mask)#.view(-1)
        a = self.atten_model(atten_x, atten_contacts, atten_mask)#.view(-1)

        g = self.linear_g(g).squeeze(-1)
        a = self.linear_a(a).squeeze(-1)
        
        g1 = self.sigmoid(g)
        a1 = self.sigmoid(a)

        sg = self.s_g**2 / (self.s_g**2 + self.s_a**2)
        sa = self.s_a**2 / (self.s_g**2 + self.s_a**2)

        combined_logits = sg * g + sa * a

        return combined_logits