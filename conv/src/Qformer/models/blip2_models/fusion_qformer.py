# Copyright (c) 2025 Zhenye Yang
# Author: Zhenye Yang
# All rights reserved.

import torch
import torch.nn as nn
from Qformer.models.base_model import all_gather_with_grad, concat_all_gather
from Qformer.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
)

from torch.nn import functional as F


class F_Former(Blip2Base):

    def __init__(
            self,
            entity_size=768,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=768,
            max_txt_len=256,
            use_margin_loss=True,      # whether to enable margin loss module
            margin_loss_weight=0.1,    # weight for margin loss
            margin=0.1,                # margin hyperparameter
            use_aux_etm=True,          # whether to enable auxiliary ITM task
            aux_etm_weight=0.2,        # weight for auxiliary ITM loss
            triplet_weight=0.2,        # weight for triplet loss
            use_mem=True,              # whether to enable memory module
            mem_ratio=0.15,            # ratio for memory sampling
            mem_weight=0.2,            # weight for memory reconstruction loss
    ):
        super().__init__()

        # Initialize tokenizer and Q-former
        self.tokenizer = self.init_tokenizer()
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, entity_size, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        # Initialize query parameters from original model weights
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # Projection layers for entity and text features
        self.entity_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # Image-text matching head
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # Temperature parameter for contrastive losses
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        # Margin loss parameters
        self.use_margin_loss = use_margin_loss
        self.margin_loss_weight = margin_loss_weight
        self.margin = margin

        # Auxiliary ITM task parameters
        self.use_aux_etm = use_aux_etm
        self.aux_etm_weight = aux_etm_weight

        # Triplet loss weight
        self.triplet_weight = triplet_weight

        # Local text projection (for memory-aware tasks)
        self.local_text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # Memory module parameters
        self.use_mem = use_mem
        self.mem_ratio = mem_ratio
        self.mem_weight = mem_weight
        # Predictor mapping memory back to original feature space
        self.mem_pred = nn.Linear(embed_dim, embed_dim)

    def forward(self, entity, context, rec_feats):

        # entity: [B, L, D] tensor of entity embeddings
        bs = entity.size(0)  # batch size
        entity_embeds = entity

        # Attention mask for entities: [B, L]
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long, device=entity.device)

        # Expand query tokens to batch dimension: [B, Q, D]
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        # === Branch 1: Entity attends to KG (original branch) ===
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            output_hidden_states=True,
            return_dict=True,
            is_decoder=False
        )
        last_hidden = query_output.hidden_states[-1]

        # Project and normalize to get entity features: [B, Q, D]
        entity_feats_1 = F.normalize(self.entity_proj(last_hidden), dim=-1)

        # === Text encoding ===
        text_tokens = self.tokenizer(
            context,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        text_tokens = {k: v.to(entity.device) for k, v in text_tokens.items()}
        text_output = self.Qformer.bert(
            input_ids=text_tokens["input_ids"],
            attention_mask=text_tokens["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
            is_decoder=False
        )
        text_last_hidden = text_output.hidden_states[-1]

        # CLS token representation for text: [B, D]
        text_feat = F.normalize(
            self.text_proj(text_last_hidden[:, 0, :]), dim=-1
        )

        # === Branch 2: Entity attends to Text (reverse branch) ===
        entity_as_query = entity_feats_1  # [B, Q, D]
        reverse_output = self.Qformer.bert(
            query_embeds=entity_as_query,
            encoder_hidden_states=text_last_hidden,
            encoder_attention_mask=text_tokens["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
            is_decoder=False
        )
        reverse_last_hidden = reverse_output.hidden_states[-1]

        # Project and normalize reverse branch features: [B, Q, D]
        entity_feats_2 = F.normalize(self.entity_proj(reverse_last_hidden), dim=-1)

        # === Fuse both branches ===
        entity_feats = (entity_feats_1 + entity_feats_2) / 2

        loss_total = 0.0  # accumulator for main and auxiliary losses

        if self.use_aux_etm and rec_feats is not None:
            # Global fused representation: [B, D]
            entity_repr = entity_feats.mean(dim=1)
            # Cosine embedding loss with positive target
            loss_aux = F.cosine_embedding_loss(entity_repr, rec_feats, torch.ones(bs, device=entity.device))
            loss_total = loss_total + self.aux_etm_weight * loss_aux

        # === Contrastive Loss with Batch-Hard Negative Mining ===
        # Compute similarity: [B, B, Q]
        sim_q2t = torch.matmul(entity_feats, text_feat.t().unsqueeze(0))
        sim_q2t = sim_q2t.permute(0, 2, 1)
        # Max over query tokens: [B, B]
        sim_i2t, _ = sim_q2t.max(dim=-1)
        sim_i2t = sim_i2t / self.temp

        # Compute text-to-image similarities: [B, B]
        sim_matrix = torch.einsum('id,jqd->ijq', text_feat, entity_feats)
        sim_t2i, _ = sim_matrix.max(dim=-1)
        sim_t2i = sim_t2i / self.temp

        # Mask out diagonal for negative mining
        mask = torch.eye(bs, dtype=torch.bool, device=entity.device)
        sim_i2t_masked = sim_i2t.masked_fill(mask, -float('inf'))
        hardest_neg_i2t, _ = sim_i2t_masked.max(dim=1)
        positive_i2t = sim_i2t.diag()

        sim_t2i_masked = sim_t2i.masked_fill(mask, -float('inf'))
        hardest_neg_t2i, _ = sim_t2i_masked.max(dim=1)
        positive_t2i = sim_t2i.diag()

        targets = torch.arange(bs, device=entity.device)
        # Cross-entropy loss for both directions
        loss_ce = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
            F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        loss_margin = 0.0
        if self.use_margin_loss:
            # Compute margin-based hardest negative loss
            loss_margin_i2t = F.relu(self.margin + hardest_neg_i2t - positive_i2t)
            loss_margin_t2i = F.relu(self.margin + hardest_neg_t2i - positive_t2i)
            loss_margin = (loss_margin_i2t.mean() + loss_margin_t2i.mean()) / 2

        loss_total = loss_total + loss_ce + self.margin_loss_weight * loss_margin

        if rec_feats is not None:
            # Triplet loss on pooled features
            entity_repr = entity_feats.mean(dim=1)
            entity_norm = F.normalize(entity_repr, dim=-1)
            rec_norm = F.normalize(rec_feats, dim=-1)
            sim = torch.matmul(entity_norm, rec_norm.t())
            pos_sim = sim.diag()
            sim_neg = sim.masked_fill(mask, -float('inf'))
            hardest_neg_sim, _ = sim_neg.max(dim=1)
            loss_triplet = F.relu(hardest_neg_sim - pos_sim + self.margin).mean()
            loss_total = loss_total + self.triplet_weight * loss_triplet

        return entity_feats, text_feat, loss_total

    def forward_entity(self, entity):
        # Process single-entity input to get top slot feature
        entity_embeds = entity.unsqueeze(1)
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long, device=entity.device)
        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            return_dict=True,
        )
        entity_feats = self.entity_proj(query_output.last_hidden_state)
        # Cosine similarity between slot features and original entity
        cosine_sim = torch.matmul(entity_feats, entity.unsqueeze(-1)).squeeze(-1)
        _, max_indices = torch.max(cosine_sim, dim=1)
        # Select slot with highest similarity
        final_entity_feats = torch.stack([entity_feats[i, idx] for i, idx in enumerate(max_indices)], dim=0)
        return final_entity_feats

    def forward_text(self, text_tokens):
        # Encode text input and return CLS feature
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def user_embed_generate(self, entity_embeds, context):
        # Generate entity slot features and text feature for user embedding
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long, device=entity_embeds.device)
        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            return_dict=True,
        )
        entity_feats = self.entity_proj(query_output.last_hidden_state)
        text_tokens = self.tokenizer(
            context,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(entity_embeds.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])
        return entity_feats, text_feat

    def fuse_embeds(self, entity, text):
        # Fuse entity slots with text representation to get final entity feature
        entity_embeds = entity
        bs = entity_embeds.size(0)
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long, device=entity.device)
        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            use_cache=True,
            return_dict=True,
        )
        entity_feats = self.entity_proj(query_output.last_hidden_state)
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(entity.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])
        # Repeat text feature for each query slot
        text_feat = text_feat.repeat(bs, 1)
        cosine_sim = torch.matmul(entity_feats, text_feat.unsqueeze(-1)).squeeze(-1)
        _, max_indices = torch.max(cosine_sim, dim=1)
        final_entity_feats = torch.stack([entity_feats[i, idx] for i, idx in enumerate(max_indices)], dim=0)
        return final_entity_feats
