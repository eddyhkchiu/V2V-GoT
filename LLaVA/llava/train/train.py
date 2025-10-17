# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image

# MY_CODE
import numpy as np


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # MY_CODE
    mm_scene_projector_input_size: Optional[int] = field(default=None)
    scene_level_only: bool = field(default=False)
    object_level_only: bool = field(default=False)
    scene_feature_mode: Optional[str] = field(default="shallow")
    object_feature_mode: Optional[str] = field(default="shallow")
    num_input_frames: Optional[int] = field(default=-1)
    ego_only: bool = field(default=False)
    feature_source: Optional[str] = field(default="no_fusion_keep_all") # or cobevt
    dataset_source: Optional[str] = field(default="v2v4real") # or v2xreal


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    # MY_CODE
    train_data_split: str = 'train'
    eval_data_split: str = 'val'
    seq_eval_mode: str = 'all' # or one of ['0000', '0001', ...]
    v2v4real_config_path: str = './playground/data/V2V4Real/data.json'
    simplified_object_feature: int = 0 
    # 0: original: dmstrack coordinate system
    # 8: [h, w, l, x, y, z, a, s], 3: [x, y, s], in v2v4real coordinate


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        # MY_CODE
        default=16,
        #default=8,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    # MY_CODE
    from_scratch: bool = False


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


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


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
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'mm_scene_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # MY_CODE
        # not tested
        assert False

        # Only save Adapter
        keys_to_match = ['mm_projector', 'mm_scene_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    #print('sources: ', sources)
    # [[{'from': 'human', 'value': 'Render a clear and concise summary of the photo.\n<image>'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]]
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # DEFAULT_IMAGE_TOKEN = "<image>"
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    #print('here 1') # no hit
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                #print('here 2') # no hit
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # MY_CODE
    #print('sources: ', sources)
    # [[{'from': 'human', 'value': '<image>\nRender a clear and concise summary of the photo.'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]]
    #assert False
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # MY_CODE
    #print('sources: ', sources)
    # llava sample
    # sources:  [[{'from': 'human', 'value': '<image>\nGive a short and clear explanation of the subsequent image.'}, {'from': 'gpt', 'value': 'an index card that says, to be a favorite, this is what i make up the menu'}]]
    
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    #print('conversations: ', conversations)
    # conversations:  ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nGive a short and clear explanation of the subsequent image. ASSISTANT: an index card that says, to be a favorite, this is what i make up the menu</s>"]
    #print('has_image: ', has_image)

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        #print('input_ids: ', input_ids)
        # MY_DEBUG
        #assert False
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    #print('input_ids: ', input_ids)
    #print('input_ids.shape: ', input_ids.shape)
    # llava: [1, 75]
    # my v2v4real: [1, 1353]
    #assert False
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        #print('rounds: ', rounds)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            #print('instruction_len: ', instruction_len)
            # 1197

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    #print('targets: ', targets)
    #print('targets.shape: ', targets.shape)
    # 1353
    #print('torch.sum(targets > 0): ', torch.sum(targets > 0))
    # 155
    # The first 55 are set to -100, probably roughly instruction_len
    #assert False
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # MY_CODE
    # MY_COMMENT
    # this preprocess_plain() ignores text input, only keep <image>
    # because the pre-train stage is only for image captioning
    #assert False
    #print('sources: ', sources)
    # [[{'from': 'human', 'value': '<image>\nRender a clear and concise summary of the photo.'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]]

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        #print('conversation: ', conversation)
        conversations.append(conversation)
    #print('conversations: ', conversations)
    # ['<image>select luxury furniture 3 - inch gel memory foam mattress topper\n']
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    #print('input_ids: ', input_ids)
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        #print('target: ', target)
        #print('source: ', source)
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        #print('tokenized_len: ', tokenized_len)
        target[:tokenized_len] = IGNORE_INDEX
        #print('target')

    # MY_CODE
    #print('input_ids: ', input_ids)
    #print('targets: ', targets)
    # input_ids and targets are the same except
    # for the beginning <image>'s token IGNORE_INDEX
    #assert False
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """


    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        #print('here 1') # here, default pretrain script
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        #print('here 2')
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        #print('here 3') # here, pretrain script with --version v1
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        #print('here 4')
        return preprocess_mpt(sources, tokenizer, has_image=has_image)

    # MY_CODE
    # not hit
    assert False


    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        #print('len(list_data_dict): ', len(list_data_dict))
        # 558128
        #print('list_data_dict[0].keys(): ', list_data_dict[0].keys())
        # (['id', 'image', 'conversations']
        #print('list_data_dict[0]["id"]: ', list_data_dict[0]["id"])
        # 004539375
        #print('list_data_dict[0]["image"]: ', list_data_dict[0]["image"])
        # 00453/004539375.jpg
        #print('list_data_dict[0]["conversations"]: ', list_data_dict[0]["conversations"])
        # [{'from': 'human', 'value': 'Render a clear and concise summary of the photo.\n<image>'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}] 

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        # MY_CODE
        #self.__getitem__(0)
        #assert False

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # MY_CODE
        #print('sources: ', sources)
        # {'id': '004539375', 'image': '00453/004539375.jpg', 'conversations': [{'from': 'human', 'value': 'Render a clear and concise summary of the photo.\n<image>'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]}

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                # MY_CODE
                #print('image.shape: ', image.shape)
                # [3, 336, 336]
                #assert False
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            # [[{'from': 'human', 'value': '<image>\nRender a clear and concise summary of the photo.'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]]
            # MY_CODE
            #assert False
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        # MY_CODE
        #assert False
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        #print('data_dict: ', data_dict)
        # 'input_ids': tensor([    1,  -200,  1831, 21684,  2857, 15252, 17252, 29871, 29941,   448,
        #  297,   305,  9127,  3370,  1701,   314,  1775,   509,   404,   304,
        #  2496,    13]), 
        #   'labels': tensor([ -100,  -100,  1831, 21684,  2857, 15252, 17252, 29871, 29941,   448,
        #  297,   305,  9127,  3370,  1701,   314,  1775,   509,   404,   304,
        # 2496,    13]), 
        # 'image': # [3, 336, 336]
        #assert False
        return data_dict


class V2V4RealDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 dataset_for: str): # 'train' or 'eval'
        super(V2V4RealDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        #print('list_data_dict: ', list_data_dict)
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

        # MY_DEBUG
        #print('MY_DEBUG single data sample.')
        #self.list_data_dict = list_data_dict[:128]


        self.data_args = data_args
        # DataArguments(data_path='./playground/data/V2V4Real/data.json', lazy_preprocess=True, is_multimodal=True, image_folder=None, image_aspect_ratio='square', train_data_split='train', eval_data_split='val', seq_eval_mode='all')
        self.model_args = model_args
        if self.model_args.feature_source in ['cobevt', 'early', 'v2xvit', 'attfuse', 'no_fusion']: # and other intermediate fusion, early fusion
            assert(self.model_args.ego_only)
        if self.model_args.scene_level_only:
            assert(not self.model_args.object_level_only)
        if self.model_args.object_level_only:
            assert(not self.model_args.scene_level_only)

        # MY_CODE
        self.dataset_for = dataset_for  # 'train' or 'eval'
        self.get_config(self.data_args.v2v4real_config_path)

        # MY_DEBUG
        #self.__len__()
        #self.__getitem__(0)
        #assert False


    def get_config(self, v2v4real_config_path):
        print('V2V4RealDataset get_config()')
        # Similar to DMSTrack's get_config()
        # determine which split of v2v4real to use
        # get only one seq when in debugging mode
        self.data_split = self.data_args.train_data_split if self.dataset_for == 'train' else self.data_args.eval_data_split
        #print('self.data_split: ', self.data_split)

        data_config = json.load(open(v2v4real_config_path, "r"))

        #data_config = {
        #  'train' : {
        #    'seq_eval' : [
        #      '0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009',
        #      '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019',
        #      '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029',
        #      '0030', '0031'
        #    ],
        #    'len_record' : [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648, 3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]
        #  },
        #  'val' : {
        #    'seq_eval' : ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008'],
        #    'len_record' : [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
        #  }
        #}

        # instead of updating and reading from config, directly set the path?
        #self.llm_data_path = data_config[self.data_split]['llm_data_path'][self.model_args.feature_source]
        if self.data_split == 'train':
          self.llm_data_path = os.path.join('../DMSTrack/V2V4Real/official_models/', 'train_' + self.model_args.feature_source, 'npy/co_llm')
        else:  
          self.llm_data_path = os.path.join('../DMSTrack/V2V4Real/official_models/', self.model_args.feature_source, 'npy/co_llm')

        self.seq_eval = data_config[self.data_split]['seq_eval'] if self.data_args.seq_eval_mode == 'all' else [self.data_args.seq_eval_mode]
        self.len_record = data_config[self.data_split]['len_record']
        print('self.llm_data_path: ', self.llm_data_path)
        # ../DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm
        # ../DMSTrack/V2V4Real/official_models/train_cobevt/npy/co_llm
        print('self.seq_eval: ', self.seq_eval)
        print('self.len_record: ', self.len_record)
        #assert False
        return


    def __len__(self):
        # revert back to the original llava format
        #print('len(self.list_data_dict): ', len(self.list_data_dict))
        # 7105 for train
        return len(self.list_data_dict)

        # my old code
        if len(self.seq_eval) == 1:
          # only run 1 seq, for example: ['0003'] 
          seq_id = int(self.seq_eval[0][-1])
          if seq_id == 0:
            length = self.len_record[0]
          else:
            length = self.len_record[seq_id] - self.len_record[seq_id-1]
        else: # all seqs
          length = self.len_record[-1]
        #print('length: ', length)
        #assert False
        # MY_DEBUG
        #single frame
        #print('MY_DEBUG single frame')
        #assert False
        #return 1
        return length

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list


    def generate_sources(self, list_data_dict, i):
        # not implemented
        assert False
        # targeted format
        # {'id': '004539375', 'image': '00453/004539375.jpg', 'conversations': [{'from': 'human', 'value': 'Render a clear and concise summary of the photo.\n<image>'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]}

        sources  = {
          'id': i,
          'conversations': [{
              'from': 'human',
              #'value': 'Generate the cooperative detection result based on the following individual detection result.\n',
            }, {
              'from': 'gpt',
              #'value': 'The cooperative detection result is\n'
            }
          ]
        }
        #print('sources: ', sources)
        human_input = 'Generate the cooperative detection result based on the following individual detection result.\n'
        gpt_output = 'The cooperative detection result is\n'


        #print('i: ', i)
        #print('list_data_dict: ', list_data_dict)
        cav_ids = ['ego', '1']
        for cav_id in cav_ids:
          individual_detection_file = os.path.join(self.llm_data_path, cav_id, '%04d_detection_llm.txt'%i)
          #print('individual_detection_file: ', individual_detection_file)
          with open(individual_detection_file, 'r') as f:
            individual_detection = f.read()
          individual_detection = individual_detection.replace(',', ', ')
          #print('individual_detection: ', individual_detection)
          human_input += 'Agent ' + cav_id + "'s detection result:\n"
          human_input += individual_detection
        #print('human_input: ', human_input)
        sources['conversations'][0]['value'] = human_input

        gt_detection_file = os.path.join(self.llm_data_path, '%04d_gt_llm.txt'%i)
        #print('gt_detection_file: ', gt_detection_file)
        with open(gt_detection_file, 'r') as f:
          gt_detection = f.read()
        gt_detection = gt_detection.replace(',', ', ')
        #print('gt_detection: ', gt_detection)
        gpt_output += gt_detection
        #print('gpt_output: ', gpt_output)
        sources['conversations'][1]['value'] = gpt_output

        #print('sources: ', sources)
        #assert False
        return sources


    def load_single_frame_feature_map(self, cav_ids, feature_map_name, single_frame_global_timestamp_index):
        #print('feature_map_name: ', feature_map_name)
        #print('single_frame_global_timestamp_index: ', single_frame_global_timestamp_index)
        #print('cav_ids: ', cav_ids)

        scene_spatial_features_2d_all = []
        feature_shape = [1, 1, 1, 1]
        for cav_id in cav_ids:
            scene_spatial_features_2d_file = os.path.join(self.llm_data_path, cav_id, '%04d_%s.npy' % (single_frame_global_timestamp_index, feature_map_name))
            # v2xreal, not every frame has every agent
            try:
              scene_spatial_features_2d = np.load(scene_spatial_features_2d_file)
              feature_shape = scene_spatial_features_2d.shape
            except FileNotFoundError:
              scene_spatial_features_2d = np.zeros(feature_shape)
            #print('scene_spatial_features_2d.shape: ', scene_spatial_features_2d.shape)
            # (1, 256, 50, 88) no_fusion_keep_all
            # (1, 256, 48, 128) cobevt
            scene_spatial_features_2d = torch.from_numpy(scene_spatial_features_2d[0])
            scene_spatial_features_2d_all.append(scene_spatial_features_2d)
        scene_spatial_features_2d = torch.stack(scene_spatial_features_2d_all, dim=0)    
        #print('scene_spatial_features_2d.shape: ', scene_spatial_features_2d.shape)
        # [2, 256, 50, 88]: [cav_id, feature_size, spatial_dim_0, spatial_dim_1]
        # [1, 256, 48, 128] cobevt
        # v2xreal: [4, 384, 100, 176]

        return scene_spatial_features_2d


    def load_single_frame_object_features(self, cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav):
        # pick Option 3 for now and continue develop, 
        # we can still switch later

        # Option 1: BEV local feature map (from DMSTrack)
        object_features_all = []
        feature_shape = [1,1]
        for cav_id in cav_ids:
            object_features_file =  os.path.join(self.llm_data_path, '../', cav_id, '%04d_feature.npy' % single_frame_global_timestamp_index)
            try:
              object_features = np.load(object_features_file)
              #print('object_features.shape: ', object_features.shape)
              # (num_boxes=20, 256, 5, 5)
              # just pick the center feature vector
              object_features = object_features[:, :, 2, 2]
              #print('object_features.shape: ', object_features.shape)
              # (num_boxes=20, 256)
              feature_shape = [max_num_boxes_per_cav, object_features.shape[1]]
            except FileNotFoundError:
              object_features = np.zeros(feature_shape)  

            if object_features.shape[0] > max_num_boxes_per_cav:
              object_features = object_features[:max_num_boxes_per_cav]  

            #q = max_num_boxes_per_cav - object_features.shape[0]
            #print('max_num_boxes_per_cav - object_features.shape[0]: ', max_num_boxes_per_cav - object_features.shape[0])
            #if q < 0:
            #  print('q < 0')  
            #  assert False

            if object_features.shape[0] < max_num_boxes_per_cav:
              object_features = np.pad(
                object_features,
                ((0, max_num_boxes_per_cav - object_features.shape[0]), (0, 0)),
                'constant', constant_values=(0, 0)
              )

            #print('object_features.shape: ', object_features.shape)
            # (50, 256): (max_num_boxes_per_cav, feature size)
            #print('object_features[:10]: ', object_features[:10])
            object_features = torch.from_numpy(object_features)
            object_features_all.append(object_features)

        object_features = torch.stack(object_features_all, dim=0)
        #print('object_features.shape: ', object_features.shape)
        # [2, 50, 256]:  (num_cavs, max_num_boxes_per_cav, feature_size)
        # [1, 50, 256] cobevt
        # v2xreal [4, 100 , 384]
        return object_features


    def load_single_frame_detection_box_score(self, cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav):
        detection_box_score_all = []
        feature_shape = None
        for cav_id in cav_ids:
            detection_box_score_file = os.path.join(self.llm_data_path, cav_id, '%04d_detection_box_score.npy' % single_frame_global_timestamp_index)
            try:
              detection_box_score = np.load(detection_box_score_file)
              feature_shape = detection_box_score.shape
            except FileNotFoundError:
              detection_box_score = np.zeros([max_num_boxes_per_cav, feature_shape[1]])  
            #print('detection_box_score.shape: ', detection_box_score.shape)
            #print('detection_box_score: ', detection_box_score)
            # (num_boxes=20, 8)
            # v2xreal (num_boxes, 9) has object class idx

            # current values are in dmstrack coordinate system
            # need to swap y and z to make it in v2v4real coordinate system

            # in v2v4real coordinate system
            if self.data_args.simplified_object_feature == 8:
              # [h, w, l, x, y, z, a, s]  
              # swap y, z
              detection_box_score = np.concatenate([
                detection_box_score[:, 0:4],
                detection_box_score[:, 5:6],
                detection_box_score[:, 4:5],
                detection_box_score[:, 6:8]
              ], axis=1)
              #print('detection_box_score: ', detection_box_score)
            elif self.data_args.simplified_object_feature == 3:
              # [x, y, s]  
              detection_box_score = np.concatenate([
                detection_box_score[:, 3:4],
                detection_box_score[:, 5:6],
                detection_box_score[:, 7:8]
              ], axis=1)
              #print('detection_box_score: ', detection_box_score)
            elif self.data_args.simplified_object_feature == 2:
              # [x, y]  
              # [x, y, s]  
              detection_box_score = np.concatenate([
                detection_box_score[:, 3:4],
                detection_box_score[:, 5:6]
              ], axis=1)
              #print('detection_box_score: ', detection_box_score)

            if detection_box_score.shape[0] > max_num_boxes_per_cav:
              detection_box_score = detection_box_score[:max_num_boxes_per_cav]  
            if detection_box_score.shape[0] < max_num_boxes_per_cav:  
              detection_box_score = np.pad(
                detection_box_score, 
                ((0, max_num_boxes_per_cav - detection_box_score.shape[0]), (0, 0)),
                'constant', constant_values=(0, 0)
              )
            #print('detection_box_score.shape: ', detection_box_score.shape)
            # (50, 8): (max_num_boxes_per_cav, self.data_args.simplified_object_feature)
            #print('detection_box_score[:10]: ', detection_box_score[:10])
            detection_box_score = torch.from_numpy(detection_box_score)
            detection_box_score_all.append(detection_box_score)

        detection_box_score = torch.stack(detection_box_score_all, dim=0)
        #print('detection_box_score.shape: ', detection_box_score.shape)
        # [2, 50, 8]:  (num_cavs, max_num_boxes_per_cav, self.data_args.simplified_object_feature)
        # [1, 50, 8] cobevt
        # v2xreal [4, 100, 9]
        #assert False
        return detection_box_score


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #print('__getitem__ i: ', i)
        # revert back to original llava format
        sources = self.list_data_dict[i]
        # my old code
        #sources = self.generate_sources(self.list_data_dict, i)
        #has_image = 'image' in sources
        #print('__getitem__: ', i)
        #print('sources: ', sources)
        global_timestamp_index = sources['global_timestamp_index']
        local_timestamp_index = sources['local_timestamp_index']

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # MY_CODE
        # include v2v4real individual point cloud feature map
        # Before generating the formal QA dataset
        # modify the input data here to continue development
        # remove the question input individual detection boxes for now
        # after we re-grenerate the v2v4real dataset, we can update this code
        # And we still need to add the '<image>\n'
        # to insert point feature tokens
        # full scene cooperative detection experiment
        #sources[0]['conversations'][0]['value'] = '<image>\n Generate the cooperative detection result based on the scene-level point cloud feature map and the object-level feature vectors.'
        # 3d grounding experiment
        sources[0]['conversations'][0]['value'] = '<image> \n ' + sources[0]['conversations'][0]['value']

        if '<image>' in sources[0]['conversations'][0]['value']:
        #if  'image' in sources[0]:
            # MY_CODE comment out image loading and processing
            # only keep the tokenizer step: '<image>\n' -> IMAGE_TOKEN_INDEX = -200
            #image_file = self.list_data_dict[i]['image']
            #image_folder = self.data_args.image_folder
            #processor = self.data_args.image_processor
            #image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            #if self.data_args.image_aspect_ratio == 'pad':
            #    def expand2square(pil_img, background_color):
            #        width, height = pil_img.size
            #        if width == height:
            #            return pil_img
            #        elif width > height:
            #            result = Image.new(pil_img.mode, (width, width), background_color)
            #            result.paste(pil_img, (0, (width - height) // 2))
            #            return result
            #        else:
            #            result = Image.new(pil_img.mode, (height, height), background_color)
            #            result.paste(pil_img, ((height - width) // 2, 0))
            #            return result
            #    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            #    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            #else:
            #    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            # We should hit here when using scene-level point feature map
            #print('sources: ', sources)
        else:
            # MY_COMMENT
            # we do not have image or point cloud feature map for now
            sources = copy.deepcopy([e["conversations"] for e in sources])
            #print('sources: ', sources)


        data_dict = preprocess(
            sources,
            self.tokenizer,
            # MY_CODE
            has_image=('image' in self.list_data_dict[i]) or '<image>' in sources[0][0]['value'])
        #print('data_dict: ', data_dict)
        #assert False
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # here


        # MY_CODE
        # include v2v4real individual point cloud feature map
        # Before generating the formal QA dataset
        # modify the input data here to continue development
        # get ego's feature map for now
        # Maybe also include regression head and classification head result feature map??

        # introduce frame dimention in the begining of data tensor
        num_input_frames = self.model_args.num_input_frames
        #print('num_input_frames: ', num_input_frames)

        if self.model_args.ego_only:
          cav_ids = ['ego']
        else:  
          cav_ids = ['ego', '1']

        # Scene-level feature map
        # Option 1: feature map before regression head and classification head
        # Option 2: regression result map and classification result map

        for feature_map_name in ['spatial_features_2d', 'regression_map', 'classification_map']:
            feature_map_all_frames = []
            for past_idx in range(num_input_frames):
                if local_timestamp_index < past_idx:  # no past frames:
                    single_frame_global_timestamp_index = global_timestamp_index
                else:    
                    single_frame_global_timestamp_index = global_timestamp_index - past_idx

                single_frame_feature_map = self.load_single_frame_feature_map(cav_ids, feature_map_name, single_frame_global_timestamp_index)
                feature_map_all_frames.append(single_frame_feature_map)

            feature_map_all_frames = torch.stack(feature_map_all_frames, dim=0)    
            if feature_map_name == 'spatial_features_2d':
              data_dict['scene_point_feature_map'] = feature_map_all_frames
            else:  
              data_dict[feature_map_name] = feature_map_all_frames

            #print('feature_map_name: ', feature_map_name)
            #print('data_dict[feature_map_name].shape: ', data_dict[feature_map_name].shape)
            # [2, 2, 256, 50, 88]
            # [2, 2,  14, 50, 88]
            # [2, 2,   2, 50, 88]
            # [num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1]

        # Object-level feature vectors
        max_num_boxes_per_cav = 50

        # Option 1: BEV local feature map (from DMSTrack)
        object_features_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_object_features = self.load_single_frame_object_features(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            object_features_all_frames.append(single_frame_object_features)

        object_features_all_frames = torch.stack(object_features_all_frames, dim=0)
        #print('object_features_all_frames.shape: ', object_features_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=256]
        data_dict['object_features'] = object_features_all_frames


        # Option 2: detection box 8 corners 
        #detection_result_file =  os.path.join(self.llm_data_path, '../', 'ego', '%04d_pred.npy' % global_timestamp_index)
        #detection_result = np.load(detection_result_file)
        #print('detection_result.shape: ', detection_result.shape)
        # (20, 8, 3) # 8 corners coordinate in 3d space
        #print('detection_result: ', detection_result)


        # Option 3: box [h, w, l, x, y, z, a] and score [s]
        # Directly pad to max number of boxes per CAV (50)
        # [max_num_boxes_per_cav * num_cavs, 8]
        detection_box_score_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_detection_box_score = self.load_single_frame_detection_box_score(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            detection_box_score_all_frames.append(single_frame_detection_box_score)

        detection_box_score_all_frames = torch.stack(detection_box_score_all_frames, dim=0)
        #print('detection_box_score_all_frames.shape: ', detection_box_score_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=8]
        data_dict['detection_box_score'] = detection_box_score_all_frames

        # try if we can pass this args to 
        # LLaVA/llava/model/language_model/llava_llama.py 
        #data_dict['object_feature_only'] = True


        data_dict['active_agent_mask'] =  torch.from_numpy(np.ones([num_input_frames, 2]))
        data_dict['i'] = torch.from_numpy(np.array(i))
        data_dict['global_timestamp_index'] = torch.from_numpy(np.array(global_timestamp_index))
        data_dict['local_timestamp_index'] = torch.from_numpy(np.array(local_timestamp_index))
        data_dict['qa_sub_type'] = torch.from_numpy(np.array(-1))

        return data_dict


class V2XRealDataset(V2V4RealDataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 dataset_for: str): # 'train' or 'eval'
        # this __init__() will call V2XRealDataset's get_config(), which is what we want
        super(V2XRealDataset, self).__init__(data_path, tokenizer, data_args, model_args, dataset_for)
        # MY_DEBUG
        print('self.__len__()', self.__len__())
        #self.__getitem__(0)
        #self.__getitem__(420)
        #assert False


    def get_config(self, v2v4real_config_path):
        print('V2XRealDataset get_config()')
        # Similar to DMSTrack's get_config()
        # determine which split of v2v4real to use
        # get only one seq when in debugging mode
        self.data_split = self.data_args.train_data_split if self.dataset_for == 'train' else self.data_args.eval_data_split
        #print('self.data_split: ', self.data_split)

        #data_config = json.load(open(v2v4real_config_path, "r"))

        data_config = {
          'train' : {
            'seq_eval' : [
              '0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009',
              '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019',
              '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029',
              '0030', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039',
              '0040', '0041', '0042'
            ],
            'len_record' : [104, 226, 389, 560, 782, 911, 1150, 1343, 1465, 1588, 1722, 1829, 1953, 2064, 2187, 2325, 2427, 2591, 2717, 2868, 3031, 3129, 3328, 3456, 3585, 3710, 3838, 3964, 4068, 4243, 4353, 4470, 4594, 4700, 4919, 5025, 5142, 5239, 5355, 5458, 5566, 5666, 5772]
          },
          'val' : {
            'seq_eval' : ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008'],
            'len_record' : [119, 359, 509, 636, 759, 857, 1044, 1150, 1253]
          }
        }

        # instead of updating and reading from config, directly set the path?
        #self.llm_data_path = data_config[self.data_split]['llm_data_path'][self.model_args.feature_source]
        if self.data_split == 'train':
          self.llm_data_path = os.path.join('../V2X-Real/my_models/', 'train_' + self.model_args.feature_source, 'npy/co_llm')
        else:  
          self.llm_data_path = os.path.join('../V2X-Real/my_models/', self.model_args.feature_source, 'npy/co_llm')

        self.seq_eval = data_config[self.data_split]['seq_eval'] if self.data_args.seq_eval_mode == 'all' else [self.data_args.seq_eval_mode]
        self.len_record = data_config[self.data_split]['len_record']
        print('self.llm_data_path: ', self.llm_data_path)
        # ../V2X-Real/my_models/train_no_fusion_keep_all/npy/co_llm
        # ../V2X-Real/my_models/train_cobevt/npy/co_llm
        print('self.seq_eval: ', self.seq_eval)
        print('self.len_record: ', self.len_record)
        #assert False

        active_agent_mask_save_file = os.path.join(self.llm_data_path, 'active_agent_mask.npy')
        active_agent_mask = np.load(active_agent_mask_save_file)
        #print('active_agent_mask.shape: ', active_agent_mask.shape)
        #print('active_agent_mask[:5]: ', active_agent_mask[:5])
        self.active_agent_mask = active_agent_mask

        active_agent_ids_save_file = os.path.join(self.llm_data_path, 'active_agent_ids.json')
        with open(active_agent_ids_save_file, "r") as f:
          active_agent_ids = json.load(f)
        #print('len(active_agent_ids): ', len(active_agent_ids))
        #print('active_agent_ids["0"]: ', active_agent_ids["0"])
        self.active_agent_ids = active_agent_ids
        #assert False
        return


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #print('__getitem__ i: ', i)
        # revert back to original llava format
        sources = self.list_data_dict[i]
        # my old code
        #sources = self.generate_sources(self.list_data_dict, i)
        #has_image = 'image' in sources
        #print('__getitem__: ', i)
        #print('sources: ', sources)
        global_timestamp_index = sources['global_timestamp_index']
        local_timestamp_index = sources['local_timestamp_index']
        qa_sub_type = sources['qa_sub_type']

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # MY_CODE
        # include v2v4real individual point cloud feature map
        # Before generating the formal QA dataset
        # modify the input data here to continue development
        # remove the question input individual detection boxes for now
        # after we re-grenerate the v2v4real dataset, we can update this code
        # And we still need to add the '<image>\n'
        # to insert point feature tokens
        # full scene cooperative detection experiment
        #sources[0]['conversations'][0]['value'] = '<image>\n Generate the cooperative detection result based on the scene-level point cloud feature map and the object-level feature vectors.'
        # 3d grounding experiment
        sources[0]['conversations'][0]['value'] = '<image> \n ' + sources[0]['conversations'][0]['value']

        if '<image>' in sources[0]['conversations'][0]['value']:
        #if  'image' in sources[0]:
            # MY_CODE comment out image loading and processing
            # only keep the tokenizer step: '<image>\n' -> IMAGE_TOKEN_INDEX = -200
            #image_file = self.list_data_dict[i]['image']
            #image_folder = self.data_args.image_folder
            #processor = self.data_args.image_processor
            #image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            #if self.data_args.image_aspect_ratio == 'pad':
            #    def expand2square(pil_img, background_color):
            #        width, height = pil_img.size
            #        if width == height:
            #            return pil_img
            #        elif width > height:
            #            result = Image.new(pil_img.mode, (width, width), background_color)
            #            result.paste(pil_img, (0, (width - height) // 2))
            #            return result
            #        else:
            #            result = Image.new(pil_img.mode, (height, height), background_color)
            #            result.paste(pil_img, ((height - width) // 2, 0))
            #            return result
            #    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            #    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            #else:
            #    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            # We should hit here when using scene-level point feature map
            #print('sources: ', sources)
        else:
            # MY_COMMENT
            # we do not have image or point cloud feature map for now
            sources = copy.deepcopy([e["conversations"] for e in sources])
            #print('sources: ', sources)


        data_dict = preprocess(
            sources,
            self.tokenizer,
            # MY_CODE
            has_image=('image' in self.list_data_dict[i]) or '<image>' in sources[0][0]['value'])
        #print('data_dict: ', data_dict)
        #assert False
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # here


        # MY_CODE
        # include v2v4real individual point cloud feature map
        # Before generating the formal QA dataset
        # modify the input data here to continue development
        # get ego's feature map for now
        # https://github.com/eddyhkchiu/my_co_llm_driver/blob/f9b102cb864d2cc611712e8e6f693bbf83ffd8ad/DMSTrack/V2V4Real/opencood/models/point_pillar.py#L62
        # Maybe also include regression head and classification head result feature map??

        # introduce frame dimention in the begining of data tensor
        num_input_frames = self.model_args.num_input_frames
        #print('num_input_frames: ', num_input_frames)


        # for v2xreal, agent ids are ['ego', '2', '-1', '-2']
        # we may have 2,3, or 4 agents
        if self.model_args.ego_only:
          cav_ids = ['ego']
        else:  
          #cav_ids = ['ego', '1']
          cav_ids = ['ego', '2', '-1', '-2']

        # Scene-level feature map
        # Option 1: feature map before regression head and classification head
        # Option 2: regression result map and classification result map

        for feature_map_name in ['spatial_features_2d', 'regression_map', 'classification_map']:
            feature_map_all_frames = []
            for past_idx in range(num_input_frames):
                if local_timestamp_index < past_idx:  # no past frames:
                    single_frame_global_timestamp_index = global_timestamp_index
                else:    
                    single_frame_global_timestamp_index = global_timestamp_index - past_idx

                single_frame_feature_map = self.load_single_frame_feature_map(cav_ids, feature_map_name, single_frame_global_timestamp_index)
                feature_map_all_frames.append(single_frame_feature_map)

            feature_map_all_frames = torch.stack(feature_map_all_frames, dim=0)    
            if feature_map_name == 'spatial_features_2d':
              data_dict['scene_point_feature_map'] = feature_map_all_frames
            else:  
              data_dict[feature_map_name] = feature_map_all_frames

            #print('feature_map_name: ', feature_map_name)
            #print('data_dict[feature_map_name].shape: ', data_dict[feature_map_name].shape)
            # [2, 2, 256, 50, 88]
            # [2, 2,  14, 50, 88]
            # [2, 2,   2, 50, 88]
            # [num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1]



        # Object-level feature vectors
        max_num_boxes_per_cav = 100

        # Option 1: BEV local feature map (from DMSTrack)
        object_features_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_object_features = self.load_single_frame_object_features(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            object_features_all_frames.append(single_frame_object_features)

        object_features_all_frames = torch.stack(object_features_all_frames, dim=0)
        #print('object_features_all_frames.shape: ', object_features_all_frames.shape)
        # [num_input_frames=2, num_cavs=4, num_max_boxes_per_cav=100, feature_size=384]
        data_dict['object_features'] = object_features_all_frames


        # Option 2: detection box 8 corners 
        #detection_result_file =  os.path.join(self.llm_data_path, '../', 'ego', '%04d_pred.npy' % global_timestamp_index)
        #detection_result = np.load(detection_result_file)
        #print('detection_result.shape: ', detection_result.shape)
        # (20, 8, 3) # 8 corners coordinate in 3d space
        #print('detection_result: ', detection_result)


        # Option 3: box [h, w, l, x, y, z, a] and score [s]
        # Directly pad to max number of boxes per CAV (50)
        # [max_num_boxes_per_cav * num_cavs, 8]
        detection_box_score_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_detection_box_score = self.load_single_frame_detection_box_score(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            detection_box_score_all_frames.append(single_frame_detection_box_score)

        detection_box_score_all_frames = torch.stack(detection_box_score_all_frames, dim=0)
        #print('detection_box_score_all_frames.shape: ', detection_box_score_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=8]
        data_dict['detection_box_score'] = detection_box_score_all_frames

        # try if we can pass this args to 
        # LLaVA/llava/model/language_model/llava_llama.py 
        #data_dict['object_feature_only'] = True



        active_agent_mask_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx
            single_frame_active_agent_mask = self.active_agent_mask[single_frame_global_timestamp_index]    
            single_frame_active_agent_mask = torch.from_numpy(single_frame_active_agent_mask)
            active_agent_mask_all_frames.append(single_frame_active_agent_mask)
        active_agent_mask_all_frames = torch.stack(active_agent_mask_all_frames, dim=0)    
        #print('active_agent_mask_all_frames: ', active_agent_mask_all_frames)
        data_dict['active_agent_mask'] = active_agent_mask_all_frames
        # [num_input_frames, num_cavs]


        # datasample id for debug
        data_dict['i'] = torch.from_numpy(np.array(i))
        data_dict['global_timestamp_index'] = torch.from_numpy(np.array(global_timestamp_index))
        data_dict['local_timestamp_index'] = torch.from_numpy(np.array(local_timestamp_index))
        data_dict['qa_sub_type'] = torch.from_numpy(np.array(qa_sub_type))


        return data_dict




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # MY_CODE
        # can do for loop over the ke words:
        # ['image', 'scene_point_feature_map', ...]
        #print('instances: ', instances)        
        for data_feature_name in ['scene_point_feature_map', 'regression_map', 'classification_map', 'detection_box_score', 'object_features', 'active_agent_mask', 'i', 'global_timestamp_index', 'local_timestamp_index', 'qa_sub_type']:
            if data_feature_name in instances[0]:
                images = [instance[data_feature_name] for instance in instances]
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch[data_feature_name] = torch.stack(images)
                else:
                    batch[data_feature_name] = images

        #print("batch['scene_point_feature_map'].shape: ", batch['scene_point_feature_map'].shape)
        # v2xreal [32, 1, 4, 384, 100, 176]

        #print("batch['regression_map'].shape: ", batch['regression_map'].shape)
        # [32, 2, 14, 50, 88]
        # v2xreal [32, 1, 4, 42, 100, 176]
        # [batch_size, num_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1]
        #print("batch['classification_map'].shape: ", batch['classification_map'].shape)
        # v2xreal [32, 1, 4, 18, 100, 176]

        #print("batch['detection_box_score'].shape: ", batch['detection_box_score'].shape)
        # [32, 2, 50, 8]
        # [batch_size, num_frames, num_cavs, max_num_boxes_per_cav, feature_size]
        # v2xreal [32, 1, 4, 100, 9]

        #print("batch['object_features'].shape: ", batch['object_features'].shape)
        # [32, 2, 50, 256]
        # [batch_size, num_frames, num_cavs, max_num_boxes_per_cav, feature_size]
        # v2xreal [32, 1, 4, 100, 384]

        #print("batch['active_agent_mask']: ", batch['active_agent_mask'])
        #print("batch['active_agent_mask'].shape: ", batch['active_agent_mask'].shape)
        # v2xreal [32, 1, 4]
        # [batch_size, num_frames, num_cavs]


        #print("batch['i']: ", batch['i'])
        #print("batch['global_timestamp_index']: ", batch['global_timestamp_index'])
        #print("batch['local_timestamp_index']: ", batch['local_timestamp_index'])
        #assert False
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # MY_CODE
    #print('data_args: ', data_args)
    # llava:
    # data_args:  DataArguments(data_path='./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json', lazy_preprocess=True, is_multimodal=True, image_folder='./playground/data/LLaVA-Pretrain/images', image_aspect_ratio='square')
    if 'V2V4Real' in data_args.data_path:
      train_dataset = V2V4RealDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                model_args=model_args,
                                dataset_for='train') # 'train' or 'eval'
    elif 'V2X-Real' in data_args.data_path:  
      train_dataset = V2XRealDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                model_args=model_args,
                                dataset_for='train') # 'train' or 'eval'
    else:
      train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

# MY_CODE
def initialize_mm_scene_projector(model):
    # Even for mm_projector, tensor is tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True) ???
    #for name, param in model.named_parameters():
    #    if 'projector' in name:
    #        print('name: ', name)
    #        print('param: ', param)

    #for name, param in model.model.mm_projector.named_parameters():
    #    print('name: ', name)
    #    print('param: ', param)

    #for name, param in model.model.mm_scene_projector.named_parameters():
    #    print('name: ', name)
    #    print('param: ', param)


    mm_scene_projector_state_dict = dict()

    # TODO: how get_peft_state_non_lora_maybe_zero_3 get non-empty tensor for mm_projector
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters())
    #print('non_lora_state_dict: ', non_lora_state_dict)
    #for key, value in non_lora_state_dict.items():
    #    if 'mm_projector' in key or 'mm_scene_projector' in key: # or 'proj' in key:    
    #        pass
            #print('key: ', key)
            #print('value.shape: ', value.shape)
            #print('torch.sum(value): ', torch.sum(value))

    #assert False


    #print("non_lora_state_dict['model.mm_projector.0.weight'].shape: ", non_lora_state_dict['model.mm_projector.0.weight'].shape)
    mm_scene_projector_0_weight = non_lora_state_dict['model.mm_projector.0.weight'].repeat(1, 3)
    mm_scene_projector_0_weight /= 3
    #print('mm_scene_projector_0_weight.shape: ', mm_scene_projector_0_weight.shape)

    #print("torch.sum(non_lora_state_dict['model.mm_projector.0.weight']): ", torch.sum(non_lora_state_dict['model.mm_projector.0.weight']))
    #print("torch.sum(non_lora_state_dict['model.mm_scene_projector.0.weight']): ", torch.sum(non_lora_state_dict['model.mm_scene_projector.0.weight']))

    mm_scene_projector_state_dict = {
        '0.weight': mm_scene_projector_0_weight,
        '0.bias': non_lora_state_dict['model.mm_projector.0.bias'],
        '2.weight': non_lora_state_dict['model.mm_projector.2.weight'],
        '2.bias': non_lora_state_dict['model.mm_projector.2.bias'],
    }


    #for name, param in model.model.mm_scene_projector.named_parameters():
    #    print('name: ', name)
    #    print('param: ', param)

    #model.model.load_state_dict(mm_scene_projector_state_dict)
    #model.mm_projector.load_state_dict(mm_scene_projector_state_dict)
    model.model.mm_scene_projector.load_state_dict(mm_scene_projector_state_dict)

    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters())
    #print('non_lora_state_dict: ', non_lora_state_dict)
    #for key, value in non_lora_state_dict.items():
    #    if 'mm_projector' in key or 'mm_scene_projector' in key: # or 'proj' in key:    
    #        pass
            #print('key: ', key)
            #print('value.shape: ', value.shape)


    #print("torch.sum(non_lora_state_dict['model.mm_projector.0.weight']): ", torch.sum(non_lora_state_dict['model.mm_projector.0.weight']))
    #print("torch.sum(non_lora_state_dict['model.mm_scene_projector.0.weight']): ", torch.sum(non_lora_state_dict['model.mm_scene_projector.0.weight']))
    #assert False

    return



def train(attn_implementation=None):
    print('my_v2vllm_graph train.py')

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    my_model_config = {
        'mm_scene_projector_input_size': model_args.mm_scene_projector_input_size,
        'scene_level_only': model_args.scene_level_only,
        'object_level_only': model_args.object_level_only,
        'scene_feature_mode': model_args.scene_feature_mode,
        'object_feature_mode': model_args.object_feature_mode,
        'num_input_frames': model_args.num_input_frames,
        'ego_only': model_args.ego_only,
        'feature_source': model_args.feature_source,
        'dataset_source': model_args.dataset_source,
    }
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # OneLogger
    output_name = training_args.output_dir.split('/')[-1][len('llava-v1.5-7b-task-lora_'):]
    #print('output_name: ', output_name)
    exp_name= '_'.join(output_name.split('_')[3:8])
    #print('exp_name: ', exp_name)
    app_tag = '_'.join([model_args.dataset_source, exp_name])
    #print('app_tag: ', app_tag)
    app_tag_run_name = 'v2vllm'
    #print('app_tag_run_name: ', app_tag_run_name)
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    #print('world_size: ', world_size)

    data_qa_id = 0
    if 'all' in app_tag:
      data_qa_id = 0
    elif 'v2' in app_tag:
      data_qa_id = 1
    elif 'v4' in app_tag:
      data_qa_id = 2
    elif 'v5' in app_tag:
      data_qa_id = 3
    elif 'v6' in app_tag:
      data_qa_id = 4
    elif 'v7' in app_tag:
      data_qa_id = 5
    else:
      assert False


    # for onelogger, but may be useful
    # [all, v2s, v4bs, v5bs, v6sdn, v7sd]
    train_iterations_target_info = {
      'v2v4real': [1678, 1387, 1400, 570, 490, 490],
      'v2xreal': [5508, 3870, 1316, 2250, 500, 500]
    }
    train_samples_target_info = {
      'v2v4real': [429439, 354820, 35700, 14339, 12290, 12290],
      'v2xreal': [704272, 495290, 167694, 28740, 6274, 6274]
    }
    train_iterations_target = train_iterations_target_info[model_args.dataset_source][data_qa_id]
    train_samples_target = train_samples_target_info[model_args.dataset_source][data_qa_id]




    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        # MY_CODE
        # not tested
        assert False
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector", "mm_scene_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            # MY_CODE
            # pass additional model_args about point cloud feature
            # from finetune script to model's __init__()
            #print('model 1') # here
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                # MY_CODE
                #model_args.mm_point_hidden_size, # 3072
                # pass full model_args
                #model_args,
                # for point feature, unable to load projector weights
                # because we have different feature size before projector
                ignore_mismatched_sizes=False,
                # MY_CODE END
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args,
                my_model_config=my_model_config
            )
            # MY_CODE
            # try to initialize mm_scene_projector from LLaVA's pretrained mm_projector
            #initialize_mm_scene_projector(model)
            #assert False
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            # MY_CODE    
            for p in model.get_model().mm_scene_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            # MY_CODE
            # not tested
            assert False
            for p in model.get_model().mm_scene_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            # MY_CODE
            # not tested
            assert False
            model.get_model().mm_scene_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # MY_CODE
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              # MY_CODE
                                              model_args=model_args)


    # MY_CODE
    # for training from scratch ablation, init llm trainable parameter
    # Custom initialization function
    import torch.nn as nn
    def init_weights(m):
        if isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Normal initialization for embeddings
        elif isinstance(m, nn.Linear):
            # MY_CODE
            if len(m.weight.shape) >= 2:
              torch.nn.init.xavier_uniform_(m.weight)  # Xavier uniform for linear layers
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)  # Biases initialized to zero
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)  # Scale initialized to 1 for LayerNorm
            torch.nn.init.zeros_(m.bias)  # Bias initialized to zero
    if training_args.from_scratch:
        print('Training from scratch: init trainable parameters')
        model.get_model().apply(init_weights)
        print('init finished')
    #assert False


    # Existing code without OneLogger
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        #print('checkpoint')
        #assert False
        trainer.train(resume_from_checkpoint=True)
    else:
        #print('no checkpoint')
        #assert False # here
        # MY_DEBUG
        # do not update trainable parameters,
        # directly save LLaVA's pretrained model weights    
        #pass
        trainer.train()
        # MY_DEBUG
        # save 1 checkpoint before training
        # does not work
        #trainer._save_checkpoint(model, None)


    trainer.save_state()


    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
