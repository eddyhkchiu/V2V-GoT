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

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM





# MY_CODE
# maybe_zero_3
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        #print('maybe_zero_3 1') # hit
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            #print('maybe_zero_3 2') # hit
            if not ignore_status:
                #print('maybe_zero_3 3') # not hit
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            #print('maybe_zero_3 4') # hit
            param = param.data.detach().cpu().clone()
    else:
        #print('maybe_zero_3 5') # not hit
        param = param.detach().cpu().clone()

    #assert False
    return param

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    print('get_peft_state_non_lora_maybe_zero_3 start')
    to_return = {k: t for k, t in named_params if "lora_" not in k}

    for k, v in to_return.items():
        if 'projector' in k:
            print('k: ', k)
            print('v.shape: ', v.shape)


    #print('g1')
    if require_grad_only:
        print('g2')
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
        print('g3') # y
    for k, v in to_return.items():
        if 'projector' in k:
            print('k: ', k)
            print('v.shape: ', v.shape)


    # all empty shape above
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    # starting from here, we have correct shape
    for k, v in to_return.items():
        if 'projector' in k:
            print('k: ', k)
            print('v.shape: ', v.shape)



    #print('g4') # n
    print('get_peft_state_non_lora_maybe_zero_3 end')
    #assert False
    return to_return




class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config
            # MY_CODE
            # more model args from train.py args
            ,
            my_model_config=None
            ):
        self.my_model_config = my_model_config
        print('my_model_config: ', my_model_config)     
        config.mm_scene_projector_input_size = my_model_config['mm_scene_projector_input_size']    
        config.object_level_only = my_model_config['object_level_only']    
        config.scene_feature_mode = my_model_config['scene_feature_mode']    
        config.object_feature_mode = my_model_config['object_feature_mode']    
        config.ego_only = my_model_config['ego_only']    
        #assert False    

            # Overwrite the token feature size before projector
        #    config.mm_hidden_size = model_args.mm_point_hidden_size

        #if model_args.mm_point_hidden_size is not None
        #    print('here')
        #    print('model_args.mm_point_hidden_size: ', model_args.mm_point_hidden_size)
        #  # 3072
        #  # overwrite the original config from model checkpoint
        #  config.mm_hidden_size = model_args.mm_hidden_size

        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        #print('config: ', config)

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
        return_dict: Optional[bool] = None,
        # MY_CODE
        scene_point_feature_map: Optional[torch.FloatTensor] = None,
        regression_map: Optional[torch.FloatTensor] = None,
        classification_map: Optional[torch.FloatTensor] = None,
        detection_box_score: Optional[torch.FloatTensor] = None,
        object_features: Optional[torch.FloatTensor] = None,
        active_agent_mask: Optional[torch.FloatTensor] = None,
        # MY_DEBUG
        i: Optional[torch.FloatTensor] = None,
        global_timestamp_index: Optional[torch.FloatTensor] = None,
        local_timestamp_index: Optional[torch.FloatTensor] = None,
        qa_sub_type: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # MY_DEBUG
        #print('i: ', i)
        #print('global_timestamp_index: ', global_timestamp_index)
        #print('local_timestamp_index: ', local_timestamp_index)
        #print('qa_sub_type: ', qa_sub_type)

        # MY_DEBUG
        #print('input_ids: ', input_ids)
        # [[    1,  -200,   447,   688,  3391,   373,   263,   521,  8233,  4315,
        #  10348,  2909,    13]] # len()==13
        #print('attention_mask: ', attention_mask) # 13 True
        #print('position_ids: ', position_ids) # None
        #print('past_key_values: ', past_key_values) # None
        #print('inputs_embeds: ', inputs_embeds) # None
        #print('labels: ', labels)
        # [[ -100,  -100,   447,   688,  3391,   373,   263,   521,  8233,  4315,
        # 10348,  2909,    13]]
        #print('use_cache: ', use_cache) # None
        #print('output_attentions: ', output_attentions) # None
        #print('output_hidden_states: ', output_hidden_states) # None
        #print('images[0].shape: ', images[0].shape) # [3, 336, 336]
        #print('image_sizes: ', image_sizes) # None
        #print('return_dict: ', return_dict) # None
        #if scene_point_feature_map is not None:
          #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape) 
          # torch.Size([2, 256, 50, 88]) when batch size 2

        #print('object_feature_only: ', object_feature_only)
        #print('object_features.shape: ', object_features.shape)

        # MY_DEBUG
        #model = self.model
        # check how this non_lora_trainables load works
        #for name, param in model.named_parameters():
        #    if 'mm_projector' in name or 'mm_scene_projector' in name:
        #        print('name: ', name)
        #        print('param.data.shape: ', param.data.shape)
        #        # shape of weights are torch.Size([0])
        #        print("torch.sum(param.data): ", torch.sum(param.data))

        #non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        #    model.named_parameters()
        #)

        #for name, param in non_lora_state_dict.items():
        #    if 'mm_projector' in name or 'mm_scene_projector' in name:
        #        print('name: ', name)
        #        print('param.shape: ', param.shape)
        #        # shape of weights
        #        print("torch.sum(param): ", torch.sum(param))

        #assert False


        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                # MY_CODE
                self.my_model_config,
                scene_point_feature_map,
                regression_map,
                classification_map,
                detection_box_score,
                object_features,
                active_agent_mask
            )
        # HERE
        #assert False

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
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        # MY_CODE
        scene_point_feature_map: Optional[torch.Tensor] = None,
        regression_map: Optional[torch.Tensor] = None,
        classification_map: Optional[torch.Tensor] = None,
        detection_box_score: Optional[torch.Tensor] = None,
        object_features: Optional[torch.Tensor] = None,
        active_agent_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                # MY_CODE
                my_model_config=self.my_model_config,
                scene_point_feature_map=scene_point_feature_map,
                regression_map=regression_map,
                classification_map=classification_map,
                detection_box_score=detection_box_score,
                object_features=object_features,
                active_agent_mask=active_agent_mask
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        # https://github.com/haotian-liu/LLaVA/issues/1492
        # https://github.com/haotian-liu/LLaVA/issues/1448
        if 'cache_position' in inputs:
            inputs.pop("cache_position")

        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
