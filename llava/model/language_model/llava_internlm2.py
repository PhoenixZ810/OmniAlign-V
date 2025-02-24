#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
from llava.model.language_model.internlm2.modeling_internlm2 import InternLM2ForCausalLM, InternLM2Model
from llava.model.language_model.internlm2.configuration_internlm2 import InternLM2Config

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.utils import rank0_print
# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaInternlm2Config(InternLM2Config):
    model_type = "llava_internlm2"


class LlavaInternlm2Model(LlavaMetaModel, InternLM2Model):
    config_class = LlavaInternlm2Config

    def __init__(self, config: InternLM2Config):
        super(LlavaInternlm2Model, self).__init__(config)


class LlavaInternlm2ForCausalLM(InternLM2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaInternlm2Config

    def __init__(self, config):
        # super(InternLM2ForCausalLM, self).__init__(config)
        InternLM2ForCausalLM.__init__(self, config)
        self.model = LlavaInternlm2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.datatype_loss = config.datatype_loss if hasattr(config, "datatype_loss") else False
        if self.datatype_loss:
            rank0_print("Logging per datatype loss")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        modalities: Optional[List[str]] = ["image"],
        data_type: Optional[str] = "normal",
        return_dict: Optional[bool] = None,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
                )
            )
        if not self.datatype_loss:
            if dpo_forward:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                hidden_states = outputs[0]
                logits = self.output(hidden_states)
                return logits, labels
            else:
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.output(hidden_states)
            logits = logits.float()

            loss = None
            per_sample_losses = None

            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                # loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # loss = loss_fct(shift_logits, shift_labels)

                ##### Compute per sample loss #####
                # Compute the token-level loss
                loss_fct = CrossEntropyLoss(reduction="none")  # "none" for token-level losses
                token_losses = loss_fct(shift_logits, shift_labels)  # Shape: [batch_size * seq_len]

                # Reshape token losses to [batch_size, seq_len - 1]
                token_losses = token_losses.view(-1, shift_logits.size(0) // inputs_embeds.size(0))
                # batch_size = inputs_embeds.size(0)
                # seq_len = inputs_embeds.size(1)
                # token_losses = token_losses.view(batch_size, seq_len - 1)

                # Mask out padding tokens
                active_tokens = (shift_labels != -100).view(-1, token_losses.size(1))
                token_losses *= active_tokens

                # Compute per-sample losses by summing over the sequence length
                per_sample_losses = token_losses.sum(dim=1) / active_tokens.sum(dim=1).clamp(min=1)

                # Compute overall loss as the mean of per-sample losses
                loss = per_sample_losses.mean()

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            output['logits'] = output['logits'].to(device)

            output["per_sample_losses"] = per_sample_losses  # Include per-sample losses in the output

            return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
                )
            )
        else:
            inputs_embeds = self.get_model().get_input_embeddings()(inputs)

        return super().generate(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_internlm2", LlavaInternlm2Config)
AutoModelForCausalLM.register(LlavaInternlm2Config, LlavaInternlm2ForCausalLM)
