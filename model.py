# Auth: Joshua Jansen van Vueren
# Date: 2024

from transformers import Wav2Vec2ForAudioFrameClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from typing import Optional,Union,Tuple
import torch
from torch import nn

_HIDDEN_STATES_START_POSITION = 2


class Wav2Vec2ForVAD(Wav2Vec2ForAudioFrameClassification):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        self.projection = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.gelu = nn.GELU()

        self.stacked_outputs = 5

        self.classifier = nn.Linear(config.hidden_size // 2, config.num_labels)

        self.avg_pool = torch.nn.AvgPool2d((self.stacked_outputs,1),stride=(self.stacked_outputs,1),ceil_mode=True)

        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        self.init_weights()

    def freeze_base_model(self,req_grad=False):
        for param in self.wav2vec2.parameters():
            param.requires_grad = req_grad

        # always freeze the feature extractor 
        self.freeze_feature_encoder()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        #######################
        # stack output states #
        #######################

        hidden_states = self.avg_pool(hidden_states)

        hidden_states = self.projection(hidden_states)

        hidden_states = self.gelu(hidden_states)

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # NOTE: we do not have to control for padded tokens
            # because during training we elected to fix input
            # length to 20 seconds.

            flattened_targets = labels.squeeze()
            flattened_logits = logits.squeeze()

            # calculate binary cross entropy loss
            loss = self.bce_loss(flattened_logits,
                            flattened_targets)  

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
