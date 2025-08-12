import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv

# F-former (fusion Q-former)
from Qformer.models.blip2_models.fusion_qformer import F_Former


class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None, prompt_max_length=50, n_examples=3,
        entity_hidden_size=128,
        # F-former options
        use_fformer=True, num_query_token=32, fformer_max_txt_len=256
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        self.use_fformer = use_fformer
        self.num_query_token = num_query_token

        self.pad_entity_id = 31161
        self.prompt_max_length = prompt_max_length
        self.n_examples = n_examples

        # ---- KG encoder (featureless R-GCN) ----
        # Featureless mode: set in_channels = n_entity and pass x=None at forward
        self.kg_encoder = RGCNConv(n_entity, entity_hidden_size, num_relations=num_relations, num_bases=num_bases)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)

        # Placeholder (currently unused) node embeddings
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)

        # Project entity embeddings to model hidden_size
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        # Token-side projection
        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

        # ==== F-former ====
        # Important: align entity_size/embed_dim to hidden_size to avoid shape mismatch
        self.Qformer = F_Former(
            entity_size=hidden_size,
            embed_dim=hidden_size,
            num_query_token=num_query_token,
            max_txt_len=fformer_max_txt_len,
            # Keep margin-based objectives; adjust weights below as needed
            use_margin_loss=True,
        )
        # These weights can be tuned during training/inference
        self.Qformer.aux_etm_weight = 0.0
        self.Qformer.triplet_weight = 0.0

        # Convert slot features [B, Q, H] to token-level embeddings that can be concatenated
        self.slot_to_token = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        # Featureless R-GCN: pass x=None
        entity_embeds = self.kg_encoder(None, self.edge_index, self.edge_type)  # [n_entity, entity_hidden_size]
        # If stronger capacity is needed, re-enable the residual MLP below:
        # entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        # entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds  # Return low-dim; project to hidden_size inside forward

    def forward(
        self,
        entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
        use_conv_prefix=False, retrieved_entity_ids=None, word_embeddings=None, mapping=True,
        context_input_embeddings=None, attention_mask=None, context_str=None, rec_feats=None, use_fformer=False
    ):
        if use_fformer is None:
            use_fformer = self.use_fformer

        batch_size, entity_embeds, entity_len = None, None, None
        retrieved_entity_embeds = None

        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds_all = self.get_entity_embeds()                       # [N_entity, ent_dim]
            ent_low = entity_embeds_all[entity_ids]                            # [B, L, ent_dim]
            # Project to hidden_size for F-former / legacy path
            ent_low = self.entity_proj1(ent_low) + ent_low
            entity_embeds = self.entity_proj2(ent_low)                          # [B, L, hidden_size]

        if retrieved_entity_ids is not None:
            retrieved_entity_embeds = entity_embeds_all[retrieved_entity_ids]   # Backward-compatible output

        # ===== Path 1: F-former-generated prefix (recommended) =====
        if use_fformer and (entity_embeds is not None) and (context_str is not None) and (not output_entity):
            # Run F-former to get slot features [B, Q, H]
            # rec_feats is typically None in conversational stage
            slot_feats, text_feat, loss_ff = self.Qformer(entity_embeds, context_str, rec_feats)

            # Convert to token-level prefix embeddings
            prompt_tokens = self.slot_to_token(slot_feats)                      # [B, Q, H]
            prompt_attention_mask = torch.ones((batch_size, prompt_tokens.size(1)), device=prompt_tokens.device)

            # Optionally prepend custom conversational prefix
            if self.n_prefix_conv is not None and use_conv_prefix:
                prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds  # [P, H]
                prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)                     # [B, P, H]
                prompt_tokens = torch.cat([prefix_embeds, prompt_tokens], dim=1)
                p = prefix_embeds.size(1)
                prefix_mask = torch.ones((batch_size, p), device=prompt_tokens.device)
                prompt_attention_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)

            # Concatenate prefix to input embeddings and attention mask
            context_input_embeddings = torch.cat([prompt_tokens, context_input_embeddings], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            assert context_input_embeddings.shape[1] == attention_mask.shape[1]

            return context_input_embeddings, attention_mask, None, retrieved_entity_embeds, loss_ff

        # ===== Path 2: legacy mapping-based prefix (fallback / ablation) =====
        if not output_entity:
            assert token_embeds is not None, "Mapping path requires token_embeds"
            bs, token_len = token_embeds.shape[:2]
            # Take the first prompt_max_length and tile by n_examples
            prompt_embeds = token_embeds[:, :self.prompt_max_length, :]
            try:
                prompt_embeds = prompt_embeds.contiguous().view(
                    bs, self.n_examples, self.prompt_max_length, self.hidden_size
                )
                prompt_embeds = prompt_embeds.view(bs, self.n_examples * self.prompt_max_length, self.hidden_size)
            except Exception as e:
                print("token_embeds shape:", token_embeds.shape)
                print("reshape target:", bs, self.n_examples, self.prompt_max_length)
                print("prompt_embeds shape:", prompt_embeds.shape)
                raise e

            if mapping:
                # Original implementation: use linear map as query over word_embeddings
                # word_embeddings expected shape: [V, H] (or [B, V, H]; here we assume [V, H])
                affinity_scores = self.cross_attn(prompt_embeds) @ word_embeddings.T
                affinity_scores = affinity_scores / self.hidden_size
                prompt_embeds = torch.softmax(affinity_scores, dim=-1) @ word_embeddings

            prompt_attention_mask = torch.ones((bs, self.n_examples * self.prompt_max_length), device=prompt_embeds.device)

            # Optionally prepend conversational prefix here as well
            if self.n_prefix_conv is not None and use_conv_prefix:
                prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
                prefix_embeds = prefix_embeds.unsqueeze(0).expand(bs, -1, -1)
                prefix_mask = torch.ones((bs, prefix_embeds.size(1)), device=prompt_embeds.device)
                prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
                prompt_attention_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)

            context_input_embeddings = torch.cat([prompt_embeds, context_input_embeddings], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            assert context_input_embeddings.shape[1] == attention_mask.shape[1]

            return context_input_embeddings, attention_mask, None, retrieved_entity_embeds, 0.0

        # ===== output_entity=True: return intermediate entity representations =====
        else:
            return entity_embeds, entity_embeds_all, retrieved_entity_embeds
