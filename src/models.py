"""
Model architectures for transformer-based DDoS detection.

Contains:
- BaselineTransformer: single-transformer multiclass classifier
- TemporalAttentionTransformer: temporal-focused expert
- ProtocolAttentionTransformer: protocol-focused expert
- FeatureAttentionTransformer: feature-focused expert
- CrossModalAttentionTransformer: cross-modality attention
- AdvancedEnsembleTransformer: 3-expert ensemble with gating + uncertainty
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Baseline Transformer
# =============================================================================

class BaselineTransformer(nn.Module):
    """Single transformer encoder for multiclass DDoS classification."""

    def __init__(self, feature_dim, embed_dim=128, num_heads=8,
                 transformer_depth=4, num_classes=13, dropout=0.1,
                 max_seq_len=128):
        super().__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.protocol_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4),
        )
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4),
        )
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        fusion_dim = embed_dim + embed_dim // 4 + embed_dim // 4
        self.embedding_projection = nn.Linear(fusion_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, protocols, temporal):
        batch_size, seq_len, _ = features.shape
        feat_emb = self.feature_embedding(features)
        proto_emb = self.protocol_embedding(protocols.unsqueeze(-1))
        temp_emb = self.temporal_embedding(temporal.unsqueeze(-1))
        pos_emb = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([feat_emb, proto_emb, temp_emb], dim=-1)
        x = self.embedding_projection(combined) + pos_emb
        x = self.dropout(x)
        x = self.transformer(x)

        pq = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.attention_pool(pq, x, x)
        return self.classifier(pooled.squeeze(1))


# =============================================================================
# Individual Attention Experts (used standalone for ablation)
# =============================================================================

class TemporalAttentionTransformer(nn.Module):
    """Transformer with an extra temporal attention layer before the encoder."""

    def __init__(self, feature_dim, embed_dim=128, num_heads=8,
                 transformer_depth=4, num_classes=13, dropout=0.1,
                 max_seq_len=128):
        super().__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.protocol_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4),
        )
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout,
        )
        self.temporal_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )

        fusion_dim = embed_dim + embed_dim // 4 + embed_dim // 2
        self.embedding_projection = nn.Linear(fusion_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )
        self.multiclass_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, protocols, temporal, output_type='all'):
        B, S, _ = features.shape
        feat_emb = self.feature_embedding(features)
        proto_emb = self.protocol_embedding(protocols.unsqueeze(-1))
        temp_emb = self.temporal_embedding(temporal.unsqueeze(-1))

        combined = torch.cat([feat_emb, proto_emb, temp_emb], dim=-1)
        x = self.embedding_projection(combined)
        x = x + self.positional_encoding[:S].unsqueeze(0).expand(B, -1, -1)
        x = self.dropout(x)

        attended, weights = self.temporal_attention(x, x, x)
        x = x + self.temporal_enhancer(attended)
        x = self.transformer(x)

        pq = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.attention_pool(pq, x, x)
        pooled = pooled.squeeze(1)

        outputs = {}
        if output_type in ('binary', 'all'):
            outputs['binary'] = self.binary_head(pooled)
        if output_type in ('multiclass', 'all'):
            outputs['multiclass'] = self.multiclass_head(pooled)
        if output_type in ('anomaly', 'all'):
            outputs['anomaly'] = self.anomaly_head(pooled)
        return outputs, {'temporal_weights': weights}


class ProtocolAttentionTransformer(nn.Module):
    """Transformer with protocol-focused attention and feature fusion."""

    def __init__(self, feature_dim, embed_dim=128, num_heads=8,
                 transformer_depth=4, num_classes=13, dropout=0.1,
                 max_seq_len=128):
        super().__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.protocol_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
        )
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4),
        )
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        self.protocol_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout,
        )
        self.protocol_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.protocol_feature_fusion = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(),
        )

        fusion_dim = embed_dim + embed_dim // 2 + embed_dim // 4
        self.embedding_projection = nn.Linear(fusion_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )
        self.multiclass_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, protocols, temporal, output_type='all'):
        B, S, _ = features.shape
        feat_emb = self.feature_embedding(features)
        proto_emb = self.protocol_embedding(protocols.unsqueeze(-1))
        temp_emb = self.temporal_embedding(temporal.unsqueeze(-1))

        pf_combined = torch.cat([feat_emb, proto_emb], dim=-1)
        pf_enhanced = self.protocol_feature_fusion(pf_combined)

        combined = torch.cat([pf_enhanced, proto_emb, temp_emb], dim=-1)
        x = self.embedding_projection(combined)
        x = x + self.positional_encoding[:S].unsqueeze(0).expand(B, -1, -1)
        x = self.dropout(x)

        attended, weights = self.protocol_attention(x, x, x)
        x = x + self.protocol_enhancer(attended)
        x = self.transformer(x)

        pq = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.attention_pool(pq, x, x)
        pooled = pooled.squeeze(1)

        outputs = {}
        if output_type in ('binary', 'all'):
            outputs['binary'] = self.binary_head(pooled)
        if output_type in ('multiclass', 'all'):
            outputs['multiclass'] = self.multiclass_head(pooled)
        if output_type in ('anomaly', 'all'):
            outputs['anomaly'] = self.anomaly_head(pooled)
        return outputs, {'protocol_weights': weights}


class FeatureAttentionTransformer(nn.Module):
    """Transformer with feature importance gating and self-attention."""

    def __init__(self, feature_dim, embed_dim=128, num_heads=8,
                 transformer_depth=4, num_classes=13, dropout=0.1,
                 max_seq_len=128):
        super().__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.protocol_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4),
        )
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4),
        )
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        self.feature_self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout,
        )
        self.feature_importance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid(),
        )
        self.feature_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
        )

        fusion_dim = embed_dim + embed_dim // 4 + embed_dim // 4
        self.embedding_projection = nn.Linear(fusion_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )
        self.multiclass_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, protocols, temporal, output_type='all'):
        B, S, _ = features.shape
        feat_emb = self.feature_embedding(features)
        proto_emb = self.protocol_embedding(protocols.unsqueeze(-1))
        temp_emb = self.temporal_embedding(temporal.unsqueeze(-1))

        # Feature importance gating
        importance = self.feature_importance(feat_emb)
        weighted_feat = feat_emb * importance

        combined = torch.cat([weighted_feat, proto_emb, temp_emb], dim=-1)
        x = self.embedding_projection(combined)
        x = x + self.positional_encoding[:S].unsqueeze(0).expand(B, -1, -1)
        x = self.dropout(x)

        attended, weights = self.feature_self_attention(x, x, x)
        x = x + self.feature_enhancer(attended)
        x = self.transformer(x)

        pq = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.attention_pool(pq, x, x)
        pooled = pooled.squeeze(1)

        outputs = {}
        if output_type in ('binary', 'all'):
            outputs['binary'] = self.binary_head(pooled)
        if output_type in ('multiclass', 'all'):
            outputs['multiclass'] = self.multiclass_head(pooled)
        if output_type in ('anomaly', 'all'):
            outputs['anomaly'] = self.anomaly_head(pooled)
        return outputs, {'feature_weights': weights, 'importance_weights': importance}


class CrossModalAttentionTransformer(nn.Module):
    """Transformer with bidirectional cross-attention between modalities."""

    def __init__(self, feature_dim, embed_dim=128, num_heads=8,
                 transformer_depth=4, num_classes=13, dropout=0.1,
                 max_seq_len=128):
        super().__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.protocol_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
        )
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
        )
        self.protocol_enhancer = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.temporal_enhancer = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        half_heads = max(1, num_heads // 2)
        self.feat_to_proto = nn.MultiheadAttention(embed_dim, half_heads, batch_first=True, dropout=dropout)
        self.feat_to_temp = nn.MultiheadAttention(embed_dim, half_heads, batch_first=True, dropout=dropout)
        self.proto_to_feat = nn.MultiheadAttention(embed_dim, half_heads, batch_first=True, dropout=dropout)
        self.temp_to_feat = nn.MultiheadAttention(embed_dim, half_heads, batch_first=True, dropout=dropout)
        self.proto_to_temp = nn.MultiheadAttention(embed_dim, half_heads, batch_first=True, dropout=dropout)
        self.temp_to_proto = nn.MultiheadAttention(embed_dim, half_heads, batch_first=True, dropout=dropout)

        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )
        self.multiclass_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, protocols, temporal, output_type='all'):
        B, S, _ = features.shape
        feat_emb = self.feature_embedding(features)
        proto_emb = self.protocol_enhancer(self.protocol_embedding(protocols.unsqueeze(-1)))
        temp_emb = self.temporal_enhancer(self.temporal_embedding(temporal.unsqueeze(-1)))

        fp, _ = self.feat_to_proto(feat_emb, proto_emb, proto_emb)
        ft, _ = self.feat_to_temp(feat_emb, temp_emb, temp_emb)
        pf, _ = self.proto_to_feat(proto_emb, feat_emb, feat_emb)
        tf, _ = self.temp_to_feat(temp_emb, feat_emb, feat_emb)
        pt, _ = self.proto_to_temp(proto_emb, temp_emb, temp_emb)
        tp, _ = self.temp_to_proto(temp_emb, proto_emb, proto_emb)

        feature_cross = (fp + ft) / 2
        protocol_cross = (pf + pt) / 2
        temporal_cross = (tf + tp) / 2

        fused = self.cross_modal_fusion(
            torch.cat([feature_cross, protocol_cross, temporal_cross], dim=-1)
        )
        x = fused + self.positional_encoding[:S].unsqueeze(0).expand(B, -1, -1)
        x = self.dropout(x)
        x = self.transformer(x)

        pq = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.attention_pool(pq, x, x)
        pooled = pooled.squeeze(1)

        outputs = {}
        if output_type in ('binary', 'all'):
            outputs['binary'] = self.binary_head(pooled)
        if output_type in ('multiclass', 'all'):
            outputs['multiclass'] = self.multiclass_head(pooled)
        if output_type in ('anomaly', 'all'):
            outputs['anomaly'] = self.anomaly_head(pooled)
        return outputs, {}


# =============================================================================
# Advanced Ensemble (Final Architecture)
# =============================================================================

class _ExpertNetwork(nn.Module):
    """Single expert: transformer encoder + attention pooling + specialization."""

    def __init__(self, embed_dim, num_heads, depth, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.specialization = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.transformer(x)
        B = x.size(0)
        pq = self.pool_query.expand(B, -1, -1)
        pooled, attn = self.attention_pool(pq, out, out)
        return self.specialization(pooled.squeeze(1)), attn


class AdvancedEnsembleTransformer(nn.Module):
    """
    3-expert ensemble with dynamic gating, meta-learning, and uncertainty.

    Experts:
        0 — Temporal  (8 heads, depth+1 layers)
        1 — Protocol  (4 heads, depth layers)
        2 — Feature   (16 heads, depth+2 layers)
    """

    def __init__(self, feature_dim, embed_dim=128, num_heads=8,
                 transformer_depth=6, num_classes=13, dropout=0.1,
                 max_seq_len=128, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim

        # Shared embeddings (deeper than baseline)
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embed_dim * 3), nn.LayerNorm(embed_dim * 3),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 3, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.protocol_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
        )
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.LayerNorm(embed_dim // 2),
        )

        fusion_dim = embed_dim + embed_dim // 2 + embed_dim // 2
        self.embedding_projection = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        # Expert networks
        expert_configs = [
            {'heads': 8,  'depth': transformer_depth + 1},   # Temporal
            {'heads': 4,  'depth': transformer_depth},        # Protocol
            {'heads': 16, 'depth': transformer_depth + 2},    # Feature
        ]
        self.experts = nn.ModuleList([
            _ExpertNetwork(embed_dim, cfg['heads'], cfg['depth'], dropout)
            for cfg in expert_configs
        ])

        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, num_experts),
            nn.Softmax(dim=-1),
        )

        # Meta-learner
        self.meta_learner = nn.Sequential(
            nn.Linear(embed_dim * num_experts, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
        )

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 1), nn.Sigmoid(),
        )

        # Output heads (input = gated + meta + gate_input + uncertainty = 3*embed+1)
        final_dim = embed_dim * 3 + 1

        self.binary_head = nn.Sequential(
            nn.Linear(final_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )
        self.multiclass_head = nn.Sequential(
            nn.Linear(final_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(final_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, protocols, temporal, output_type='all'):
        B, S, _ = features.shape

        feat_emb = self.feature_embedding(features)
        proto_emb = self.protocol_embedding(protocols.unsqueeze(-1))
        temp_emb = self.temporal_embedding(temporal.unsqueeze(-1))

        combined = torch.cat([feat_emb, proto_emb, temp_emb], dim=-1)
        x = self.embedding_projection(combined)
        x = x + self.positional_encoding[:S].unsqueeze(0).expand(B, -1, -1)
        x = self.dropout(x)

        # Expert processing
        expert_outputs, expert_attentions = [], []
        for expert in self.experts:
            out, attn = expert(x)
            expert_outputs.append(out)
            expert_attentions.append(attn)

        # Dynamic gating
        gate_input = torch.mean(x, dim=1)
        gate_weights = self.gating_network(gate_input)  # (B, num_experts)

        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, E, D)
        gated = torch.sum(expert_stack * gate_weights.unsqueeze(-1), dim=1)

        # Meta-learner
        meta_in = torch.cat(expert_outputs, dim=-1)
        meta_out = self.meta_learner(meta_in)

        # Uncertainty
        uncertainty = self.uncertainty_head(gated)

        # Final representation
        final = torch.cat([gated, meta_out, gate_input, uncertainty], dim=-1)

        outputs = {}
        if output_type in ('binary', 'all'):
            outputs['binary'] = self.binary_head(final)
        if output_type in ('multiclass', 'all'):
            outputs['multiclass'] = self.multiclass_head(final)
        if output_type in ('anomaly', 'all'):
            outputs['anomaly'] = self.anomaly_head(final)
        outputs['uncertainty'] = uncertainty

        attention_info = {
            'gate_weights': gate_weights,
            'expert_attentions': expert_attentions,
            'uncertainty': uncertainty,
        }
        return outputs, attention_info
