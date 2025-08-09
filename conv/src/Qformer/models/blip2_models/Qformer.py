# Copyright (c) 2025 Zhenye Yang
# Author: Zhenye Yang
# All rights reserved.

import math
import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaEmbeddings as _RobertaEmbeddings,
    RobertaPooler,
    RobertaLMHead,
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)


class RobertaQEmbeddings(_RobertaEmbeddings):
    """
    Extends RobertaEmbeddings to prepend `query_embeds` when provided.
    """
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length: int = 0,
    ):
        # 1) standard token + position embeddings
        if input_ids is not None:
            seq_len = input_ids.size(1)
            if position_ids is None:
                position_ids = self.position_ids[
                    :, past_key_values_length : seq_len + past_key_values_length
                ]
            embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        else:
            embeddings = None

        # 2) if query_embeds present:
        if query_embeds is not None:
            if embeddings is None:
                embeddings = query_embeds
            else:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaSelfAttention(nn.Module):
    def __init__(self, config: RobertaConfig, is_cross_attention: bool):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "Hidden size must be divisible by number of heads"
            )
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_cross_attention = is_cross_attention

    def transpose_for_scores(self, x: Tensor):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        is_cross = encoder_hidden_states is not None

        if is_cross:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        past_key_value = (key_layer, value_layer)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, value_layer)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(*hidden_states.size()[:-1], self.all_head_size)
        )
        if output_attentions:
            return context, weights, past_key_value
        return context, past_key_value


class RobertaSelfOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden = self.dense(hidden_states)
        hidden = self.dropout(hidden)
        hidden = self.LayerNorm(hidden + input_tensor)
        return hidden


class RobertaAttention(nn.Module):
    def __init__(self, config: RobertaConfig, is_cross_attention: bool = False):
        super().__init__()
        self.self = RobertaSelfAttention(config, is_cross_attention)
        self.output = RobertaSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        if output_attentions:
            context, attn, pkv = self_outputs
            attention_output = self.output(context, hidden_states)
            return attention_output, attn, pkv
        else:
            context, pkv = self_outputs
            attention_output = self.output(context, hidden_states)
            return attention_output, pkv


class RobertaIntermediate(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        return self.act(self.dense(x))


class RobertaOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden = self.dense(hidden_states)
        hidden = self.dropout(hidden)
        hidden = self.LayerNorm(hidden + input_tensor)
        return hidden


class RobertaLayer(nn.Module):
    def __init__(self, config: RobertaConfig, layer_idx: int):
        super().__init__()
        self.attention = RobertaAttention(config, is_cross_attention=False)
        self.has_cross_attention = config.add_cross_attention and (layer_idx % config.cross_attention_freq == 0)
        if self.has_cross_attention:
            self.crossattention = RobertaAttention(config, is_cross_attention=True)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        # for query tokens
        self.intermediate_query = RobertaIntermediate(config)
        self.output_query = RobertaOutput(config)
        self.chunk_size = config.chunk_size_feed_forward
        self.seq_dim = 1

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length: int = 0,
    ):
        self_attn_pkv = past_key_value[:2] if past_key_value is not None else None
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_pkv,
        )
        attn_output = self_outputs[0]
        present = self_outputs[-1]
        attn_weights = self_outputs[1] if output_attentions else None

        if query_length > 0:
            q_out = attn_output[:, :query_length]
            txt_out = attn_output[:, query_length:]
            if self.has_cross_attention:
                q_attn = self.crossattention(
                    q_out,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                q_out = q_attn[0]
            q_ffn = apply_chunking_to_forward(
                self.feed_forward_query, self.chunk_size, self.seq_dim, q_out
            )
            t_ffn = apply_chunking_to_forward(
                self.feed_forward, self.chunk_size, self.seq_dim, txt_out
            )
            layer_output = torch.cat((q_ffn, t_ffn), dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward, self.chunk_size, self.seq_dim, attn_output
            )

        outputs = (layer_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs

    def feed_forward(self, x: Tensor):
        return self.output(self.intermediate(x), x)

    def feed_forward_query(self, x: Tensor):
        return self.output_query(self.intermediate_query(x), x)


class RobertaEncoder(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.layer = nn.ModuleList([RobertaLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length: int = 0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_past = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                layer_past,
                output_attentions,
                query_length,
            )
            hidden_states = layer_outputs[0]
            present = layer_outputs[1]
            if use_cache:
                next_cache += (present,)
            if output_attentions:
                all_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            outputs = (hidden_states, next_cache, all_hidden_states, all_attentions)
            return outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=None,
        )


class RobertaQFormer(RobertaPreTrainedModel):
    """
    Roberta-based Q-Former with support for:
     - prepending learnable query tokens
     - cross-attention at configurable layers
     - acting as encoder and decoder (for generation)
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config: RobertaConfig, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaQEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

    def bert(self, *args, **kwargs):
        """
        Alias to forward(), to allow calls like self.Qformer.bert(...)
        """
        return self.forward(*args, **kwargs)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _invert_attention_mask(self, mask: Tensor):
        return (1.0 - mask[:, None, None, :]) * -10000.0

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape,
        device,
        is_decoder=False,
        has_query=False,
    ):
        if attention_mask.dim() == 3:
            ext_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if is_decoder:
                batch, seq = input_shape
                ids = torch.arange(seq, device=device)
                causal = ids[None, None, :].repeat(batch, seq, 1) <= ids[None, :, None]
                causal = causal.to(attention_mask.dtype)
                if has_query:
                    qlen = attention_mask.shape[1] - seq
                    causal = torch.cat(
                        [
                            torch.zeros((batch, qlen, seq), device=device, dtype=causal.dtype),
                            causal,
                        ],
                        dim=1,
                    )
                ext_mask = causal[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                ext_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask {attention_mask.shape}")
        ext_mask = ext_mask.to(dtype=self.dtype)
        return (1.0 - ext_mask) * -10000.0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        past_len = past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        qlen = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            past_key_values_length=past_len,
        )

        batch, seq = embedding_output.size()[:-1]
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones((batch, seq), device=device)
        ext_attn = self.get_extended_attention_mask(
            attention_mask, (batch, seq), device, is_decoder, has_query=(qlen > 0)
        )

        if encoder_hidden_states is not None:
            enc_ext_attn = self._invert_attention_mask(encoder_attention_mask) if encoder_attention_mask is not None else None
        else:
            enc_ext_attn = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=ext_attn,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=enc_ext_attn,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=qlen,
        )

        sequence_output = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
            return outputs

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaForMaskedLM(RobertaQFormer):
    """
    Q-Former + LM head for MLM and autoregressive generation.
    """
    def __init__(self, config: RobertaConfig):
        super().__init__(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=True,
        reduction="mean",
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]
        if query_embeds is not None:
            sequence_output = sequence_output[:, query_embeds.shape[1]:, :]

        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            shifted_logits = prediction_scores[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            lm_loss = loss_fct(
                shifted_logits.view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
            )
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )




