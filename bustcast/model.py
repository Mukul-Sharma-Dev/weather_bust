"""
Architecture (see README for the ASCII diagram):

  node features (per point)
        |
  Graph-Attention spatial encoder (kNN, k from geo.choose_k)     -> per-point embeddings h_i
        |
  region pooling (KMeans regions, precomputed pool matrix)       -> compressed spatial tokens (Sec.25)
        |
  4 lag snapshots per region (t0 state, prev1, rollmean3, rollmean7)  <- pseudo-temporal axis;
        |                                                              see docstring at the bottom
  Temporal Transformer Encoder (geo + temporal positional embeddings)
        |
  7 causal lead-query tokens  --cross-attn-->  Causal Transformer Decoder
        |
  broadcast region embedding back to points (pool^T)             -> "900-location output reconstruction"
        |
  concat with point-level GAT embedding (skip connection)
        |
  Head 1: Huber error regression   |   Head 2: sigmoid bust probability
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(x, n_freqs):
    """x: [...,] in roughly [-1,1] -> [..., 2*n_freqs] sin/cos positional features."""
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device).float()
    ang = x[..., None] * freqs * math.pi
    return torch.cat([ang.sin(), ang.cos()], -1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_in, d_out, heads, edge_dim):
        super().__init__()
        self.h, self.dk = heads, d_out // heads
        self.q = nn.Linear(d_in, d_out)
        self.k = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_out)
        self.edge_proj = nn.Linear(edge_dim, heads)
        self.out = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, nbr, edge_feat):
        # x: [B,N,d_in]  nbr: [N,K] (shared across batch)  edge_feat: [N,K,edge_dim]
        B, N, _ = x.shape
        K = nbr.shape[1]
        q = self.q(x).view(B, N, self.h, self.dk)
        k_all = self.k(x).view(B, N, self.h, self.dk)
        v_all = self.v(x).view(B, N, self.h, self.dk)
        k_n = k_all[:, nbr]              # [B,N,K,h,dk]
        v_n = v_all[:, nbr]              # [B,N,K,h,dk]
        scores = torch.einsum("bnhd,bnkhd->bnkh", q, k_n) / math.sqrt(self.dk)
        scores = scores + self.edge_proj(edge_feat)[None]           # geo bias, broadcast over batch
        attn = torch.softmax(scores, dim=2)                          # over K neighbours
        agg = torch.einsum("bnkh,bnkhd->bnhd", attn, v_n).reshape(B, N, -1)
        return self.norm(self.skip(x) + self.out(agg))


class SpatialEncoder(nn.Module):
    def __init__(self, d_in, d_node, heads, n_layers, edge_dim=3):
        super().__init__()
        self.inp = nn.Linear(d_in, d_node)
        self.layers = nn.ModuleList([GraphAttentionLayer(d_node, d_node, heads, edge_dim)
                                      for _ in range(n_layers)])

    def forward(self, x, nbr, edge_feat):
        h = F.gelu(self.inp(x))
        for layer in self.layers:
            h = layer(h, nbr, edge_feat)
        return h  # [B,N,d_node]


class TemporalViews(nn.Module):
    """Turns one node embedding (already built from a feature vector that contains prev1/
    rollmean3/rollmean7/delta/spatial-anomaly sub-blocks -- see dataset.py's `hist` feature
    group) into T learned 'read-out' tokens. Each view is a small linear head that the network
    is free to specialise toward a different lag scale (recent vs. weekly, say); giving the
    Temporal Transformer several distinct tokens to attend across, without re-running the
    (expensive) spatial encoder once per lag as a literal 4-pass pipeline would require."""
    def __init__(self, d_node, n_views):
        super().__init__()
        self.views = nn.ModuleList([nn.Linear(d_node, d_node) for _ in range(n_views)])

    def forward(self, h):
        return torch.stack([v(h) for v in self.views], dim=1)   # [B,T,N,d_node]


class TemporalEncoder(nn.Module):
    def __init__(self, d_node, d_model, heads, n_layers, ff_mult, dropout, geo_freqs):
        super().__init__()
        self.proj = nn.Linear(d_node, d_model)
        self.time_emb = nn.Embedding(8, d_model)     # a handful of pseudo-timesteps is plenty
        self.geo_mlp = nn.Linear(4 * geo_freqs, d_model)
        self.geo_freqs = geo_freqs
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * ff_mult, dropout,
                                            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, tokens, geo_norm_xy):
        # tokens: [B,T,R,d_node]  geo_norm_xy: [R,2] normalised lat/lon of region centroids
        B, T, R, _ = tokens.shape
        h = self.proj(tokens)
        geo_feat = torch.cat([sinusoidal(geo_norm_xy[:, 0], self.geo_freqs),
                               sinusoidal(geo_norm_xy[:, 1], self.geo_freqs)], -1)   # [R,4*freqs]
        h = h + self.geo_mlp(geo_feat)[None, None]
        h = h + self.time_emb(torch.arange(T, device=tokens.device))[None, :, None]
        h = h.reshape(B, T * R, -1)
        h = self.encoder(h)
        return h.reshape(B, T, R, -1)


class CausalDecoder(nn.Module):
    def __init__(self, d_model, heads, n_layers, ff_mult, dropout, n_leads):
        super().__init__()
        self.n_leads = n_leads
        self.lead_tokens = nn.Parameter(torch.randn(n_leads, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, heads, d_model * ff_mult, dropout,
                                            batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        mask = torch.triu(torch.full((n_leads, n_leads), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, memory):
        # memory: [B, T*R, d_model] encoder output (all GFS-derived context, legitimately
        # available at init time -- see README section on causal masking)
        B = memory.shape[0]
        tgt = self.lead_tokens[None].expand(B, -1, -1)
        return self.decoder(tgt, memory, tgt_mask=self.causal_mask)   # [B,n_leads,d_model]


class BustCastModel(nn.Module):
    """
    x: [B,N,F] point-level features at t0 (see dataset.py)
    x_prev1, x_roll3, x_roll7: [B,N,F] the same feature block computed from the lag columns
        (this IS the model's pseudo-temporal axis -- see module docstring). If you later have
        genuinely separate daily raw sequences to feed in, swap TemporalEncoder's `tokens` input
        for a real [B,T,R,d_node] stack; nothing else in the architecture needs to change.
    """
    def __init__(self, n_hist_feat, n_lead_feat, n_points, n_leads, n_error_vars, cfg,
                 coords_norm, region_pool, n_temporal_views=3):
        super().__init__()
        m = cfg["model"]
        self.n_leads = n_leads
        self.spatial = SpatialEncoder(n_hist_feat, m["d_node"], m["heads"], m["gat_layers"])
        self.temporal_views = TemporalViews(m["d_node"], n_temporal_views)
        self.register_buffer("region_pool", torch.from_numpy(region_pool).float(), persistent=False)  # [R,N]
        self.register_buffer("region_pool_T", torch.from_numpy(region_pool.T).float(), persistent=False)
        region_xy = coords_norm  # [R,2] already normalised centroids, passed in
        self.register_buffer("region_xy", torch.from_numpy(region_xy).float(), persistent=False)

        self.temporal = TemporalEncoder(m["d_node"], m["d_model"], m["heads"], m["enc_layers"],
                                         m["ff_mult"], m["dropout"], m["geo_freqs"])
        self.decoder = CausalDecoder(m["d_model"], m["heads"], m["dec_layers"], m["ff_mult"],
                                      m["dropout"], n_leads)
        self.lead_embed = nn.Embedding(n_leads, m["d_model"])
        # per-lead GFS forecast state -- legitimately available at init time (spec Sec.10) --
        # is injected here, at the decoder/head stage, rather than through the causal decoder
        # queries themselves.
        self.gfs_lead_proj = nn.Linear(n_lead_feat, m["d_model"])

        head_in = m["d_model"] + m["d_node"] + m["d_model"]   # lead_ctx + node_h + gfs_lead
        self.err_head = nn.Sequential(nn.Linear(head_in, m["d_model"]), nn.GELU(),
                                       nn.Linear(m["d_model"], 1))
        self.errvar_head = nn.Sequential(nn.Linear(head_in, m["d_model"]), nn.GELU(),
                                          nn.Linear(m["d_model"], n_error_vars))
        self.bust_head = nn.Sequential(nn.Linear(head_in, m["d_model"]), nn.GELU(),
                                        nn.Linear(m["d_model"], 1))
        # Decoding strategy: PARALLEL / non-autoregressive. The 7 lead tokens are learned
        # queries, not shifted teacher-forced error values, so there is nothing for the causal
        # mask to leak -- it is kept purely so a later teacher-forcing variant (feeding
        # Error_D1..D6 as decoder input, as in the spec's Sec.11) can be dropped in without
        # touching the encoder. See README "Decoding strategy" for the full justification.

    def encode(self, x_hist, nbr, edge_feat):
        """x_hist: [B,N,Fh] lead-independent history/spatial feature block (see dataset.py)."""
        node_h = self.spatial(x_hist, nbr, edge_feat)                      # [B,N,d_node]
        views = self.temporal_views(node_h)                                # [B,T,N,d_node]
        region_tokens = torch.einsum("rn,btnd->btrd", self.region_pool, views)   # [B,T,R,d_node]
        enc = self.temporal(region_tokens, self.region_xy)                 # [B,T,R,d_model]
        memory = enc.reshape(enc.shape[0], -1, enc.shape[-1])              # [B,T*R,d_model]
        return memory, node_h

    def decode_and_predict(self, memory, node_h, x_lead):
        """
        dec: [B, n_leads, d_model] -- ONE embedding per lead, produced by causal cross-attention
        over the full T*R spatiotemporal memory (built purely from ERA5 history + spatial
        neighbourhood -- never from GFS, so this half of the model cannot leak the forecast
        itself). This is a deliberate simplification: the decoder carries "how is the atmosphere
        evolving, at this lead" as a global-context vector rather than a per-region one (the
        causal masking is about lead-dependence and leakage-safety; see README). Per-point,
        per-region granularity in the final [N,7] output comes from concatenating this lead
        context with (a) each point's own GAT embedding (node_h) and (b) that point's own
        lead-specific GFS forecast state (x_lead) before the prediction heads -- FiLM-style
        conditioning rather than routing spatial/forecast detail through the decoder itself.
        x_lead: [B,N,L,Fl] per-point, per-lead GFS state + forecast-cycle-revision features.
        """
        dec = self.decoder(memory)                                                 # [B,L,d_model]
        B, N, L, _ = x_lead.shape
        point_feat = node_h[:, None, :, :].expand(-1, L, -1, -1)                   # [B,L,N,d_node]
        lead_ctx = dec[:, :, None, :].expand(-1, -1, N, -1)                        # [B,L,N,d_model]
        gfs_lead = self.gfs_lead_proj(x_lead).transpose(1, 2)                      # [B,L,N,d_model]
        combo = torch.cat([lead_ctx, point_feat, gfs_lead], -1)

        err = self.err_head(combo).squeeze(-1)            # [B,L,N]
        errvar = self.errvar_head(combo)                  # [B,L,N,n_error_vars]
        bust_logit = self.bust_head(combo).squeeze(-1)     # [B,L,N]
        return err.transpose(1, 2), errvar.transpose(1, 2), bust_logit.transpose(1, 2)  # ->[B,N,L(,V)]

    def forward(self, x_hist, x_lead, nbr, edge_feat):
        memory, node_h = self.encode(x_hist, nbr, edge_feat)
        return self.decode_and_predict(memory, node_h, x_lead)
