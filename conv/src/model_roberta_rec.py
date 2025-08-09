import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from modeling_roberta import RobertaLMHead, RobertaModel
import numpy as np
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
import logging
logger = logging.getLogger(__name__)

@dataclass
class MultiOutput(ModelOutput):
    rec_loss: Optional[torch.FloatTensor] = None
    rec_logits: Optional[torch.FloatTensor] = None

class RobertaForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.tokenizer = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        entity_demonstrations = None,
        entity_embeds=None,
        rec_labels = None
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_demonstrations=entity_demonstrations,
        )[0]

        cls_tokens = outputs[:, 0, :]
        rec_logits = torch.matmul(cls_tokens, entity_embeds.permute(1,0))
        if rec_labels is not None:
            # loss_fct = CrossEntropyLoss()
            rec_loss = F.cross_entropy(rec_logits, rec_labels)
    
        return MultiOutput(
            rec_loss=rec_loss,
            rec_logits=rec_logits,
        )