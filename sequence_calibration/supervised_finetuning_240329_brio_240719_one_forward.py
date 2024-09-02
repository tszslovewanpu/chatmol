# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, LLaMA, Bloom, ...) on a json file or a dataset.

part of code is modified from https://github.com/shibing624/textgen
"""
import math
import os
from dataclasses import dataclass, field
from glob import glob
from types import MethodType
# from typing import Literal, Optional, Tuple, List, Dict, Sequence
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    BloomForCausalLM,
    AutoModel,
    AutoModelForCausalLM,
    # LlamaTokenizer, ### biubiubiu 35不能用这个LlamaTokenizer，得用AutoTokenizer
    LlamaForCausalLM,
    BloomTokenizerFast,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
# from transformers.models.llama import modeling_llama
# from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv




### biubiubiu
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaFlashAttention2,
    Cache
)
### biubiubiu
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother

### biubiubiu
from transformers.utils.versions import require_version
### biubiubiu

#######biubiubiu#######
### pandas
### .models 就是 transformers.models
import torch.nn.functional as F
import numpy as np
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import sys
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)
import shutil
import torch.distributed as dist
import time
import pandas as pd
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    # is_torch_tpu_available,
    is_torch_xla_available,
    strtobool,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
    # find_subsequence,
    # extract_properties,
)

if is_datasets_available():
    import datasets
import inspect

import logging
from logging import FileHandler
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from packaging import version
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
# if is_accelerate_available():
#     from accelerate import Accelerator, skip_first_batches
#     from accelerate import __version__ as accelerate_version
#     from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
if is_accelerate_available(): ### 35 update biubiubiu
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )
    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper
# from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_less_than_1_11
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13

# if is_torch_tpu_available(check_device=False):
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
#####biubiubiu#####

###### biubiubiu35 ######
from transformers.trainer import _is_peft_model
from transformers.integrations.tpu import tpu_spmd_dataloader
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False
###### biubiubiu35 ######

### biubiubiu 35
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled
is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False
### biubiubiu 35

# try:
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import pad_input, unpad_input
# except ImportError:
#     print("FlashAttention-2 is not installed, ignore this if you are not using FlashAttention.")

### biubiubiu 35
from template import get_conv_template
### biubiubiu 35


### biubiubiu 35 one forward
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""
_CONFIG_FOR_DOC = "LlamaConfig"
class LlamaForCausalLMBrio(LlamaForCausalLM):
    def __init__(self, config, num_candidates, train_batch_size, brio_log_file_path, candidate_label_pad_token_id):
        super().__init__(config)
        # custom parts
        self.num_candidates = num_candidates
        self.train_batch_size = train_batch_size
        self._setup_brio_logger(brio_log_file_path)
        self.candidate_label_pad_token_id = candidate_label_pad_token_id
        # self.extra_layer = nn.Linear(config.hidden_size, config.hidden_size)
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward( ### step1 先来到了这里进行forward的配置
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions ### None 然后是 后面self的值，False
        output_hidden_states = ( ### None 然后是，最终选了这个self.config.output_hidden_states=False
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict ### None，然后是，最终选了这个self.config.use_return_dict=True
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
            cache_position=cache_position,
        ) ### outputs.last_hidden_state.shape = torch.Size([21, 212, 4096])

        hidden_states = outputs[0] ### here torch.Size([3, 184, 4096]) ### one forward hidden_states.shape = torch.Size([21, 212, 4096]) 
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states) ### here， torch.Size([3, 184, 128256])    self.lm_head = Linear(in_features=4096, out_features=128256, bias=False) ### logits.shape = torch.Size([21, 212, 128256]) one forward 

            # logits
            # tensor([[[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
            #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
            #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
            #         ...,
            #         [-4.0938,  0.0334,  8.8750,  ..., -3.5000, -3.5000, -3.5000],
            #         [-4.0000,  0.4004,  8.7500,  ..., -3.8281, -3.8281, -3.8281],
            #         [-4.5312,  0.1963,  8.3750,  ..., -4.1250, -4.1250, -4.1250]],

            #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
            #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
            #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
            #         ...,
            #         [-6.2812, -0.8398,  8.3750,  ..., -3.8281, -3.8281, -3.8281],
            #         [-6.4688, -0.9023,  8.2500,  ..., -3.9375, -3.9375, -3.9375],
            #         [-6.1562, -0.9141,  8.1875,  ..., -3.8906, -3.8906, -3.8906]],

            #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
            #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
            #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
            #         ...,
            #         [-5.1250,  0.2070,  8.6875,  ..., -4.0312, -4.0312, -4.0312],
            #         [-5.3125,  0.3887,  8.3750,  ..., -4.5938, -4.5938, -4.5938],
            #         [-5.2500,  0.1416,  7.9375,  ..., -4.6250, -4.6250, -4.6250]],

            #         ...,

            #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
            #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
            #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
            #         ...,
            #         [-4.6875, -0.0669,  8.6875,  ..., -3.4531, -3.4531, -3.4531],
            #         [-4.1875,  0.6914,  8.7500,  ..., -3.8438, -3.8438, -3.8438],
            #         [-4.1562,  0.5781,  8.4375,  ..., -4.0312, -4.0312, -4.0312]],

            #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
            #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
            #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
            #         ...,
            #         [-4.1875,  0.9219,  8.5625,  ..., -4.0312, -4.0312, -4.0312],
            #         [-4.6562,  0.5508,  8.3750,  ..., -4.3750, -4.3750, -4.3750],
            #         [-4.9688, -0.1797,  8.3125,  ..., -4.2500, -4.2500, -4.2500]],

            #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
            #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
            #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
            #         ...,
            #         [-5.6875, -0.1855,  8.8750,  ..., -3.7031, -3.7031, -3.7031],
            #         [-5.5938,  0.0204,  8.8125,  ..., -3.8750, -3.8750, -3.8750],
            #         [-5.0625,  0.2139,  8.5625,  ..., -3.9844, -3.9844, -3.9844]]],
            #     device='cuda:0', grad_fn=<ToCopyBackward0>)
            #     torch.float32
        logits = logits.float() #### logits.shape = torch.Size([14, 212, 128256])

        # logits
        # tensor([[[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
        #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
        #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
        #         ...,
        #         [-4.0938,  0.0334,  8.8750,  ..., -3.5000, -3.5000, -3.5000],
        #         [-4.0000,  0.4004,  8.7500,  ..., -3.8281, -3.8281, -3.8281],
        #         [-4.5312,  0.1963,  8.3750,  ..., -4.1250, -4.1250, -4.1250]],

        #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
        #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
        #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
        #         ...,
        #         [-6.2812, -0.8398,  8.3750,  ..., -3.8281, -3.8281, -3.8281],
        #         [-6.4688, -0.9023,  8.2500,  ..., -3.9375, -3.9375, -3.9375],
        #         [-6.1562, -0.9141,  8.1875,  ..., -3.8906, -3.8906, -3.8906]],

        #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
        #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
        #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
        #         ...,
        #         [-5.1250,  0.2070,  8.6875,  ..., -4.0312, -4.0312, -4.0312],
        #         [-5.3125,  0.3887,  8.3750,  ..., -4.5938, -4.5938, -4.5938],
        #         [-5.2500,  0.1416,  7.9375,  ..., -4.6250, -4.6250, -4.6250]],

        #         ...,

        #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
        #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
        #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
        #         ...,
        #         [-4.6875, -0.0669,  8.6875,  ..., -3.4531, -3.4531, -3.4531],
        #         [-4.1875,  0.6914,  8.7500,  ..., -3.8438, -3.8438, -3.8438],
        #         [-4.1562,  0.5781,  8.4375,  ..., -4.0312, -4.0312, -4.0312]],

        #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
        #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
        #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
        #         ...,
        #         [-4.1875,  0.9219,  8.5625,  ..., -4.0312, -4.0312, -4.0312],
        #         [-4.6562,  0.5508,  8.3750,  ..., -4.3750, -4.3750, -4.3750],
        #         [-4.9688, -0.1797,  8.3125,  ..., -4.2500, -4.2500, -4.2500]],

        #         [[ 7.0000,  8.8750, 13.0000,  ..., -4.5000, -4.5000, -4.5000],
        #         [ 1.9453,  1.8594,  1.9453,  ..., -8.1875, -8.1875, -8.1875],
        #         [ 6.3438,  2.6719,  3.1094,  ..., -7.8750, -7.8750, -7.8750],
        #         ...,
        #         [-5.6875, -0.1855,  8.8750,  ..., -3.7031, -3.7031, -3.7031],
        #         [-5.5938,  0.0204,  8.8125,  ..., -3.8750, -3.8750, -3.8750],
        #         [-5.0625,  0.2139,  8.5625,  ..., -3.9844, -3.9844, -3.9844]]],
        #     device='cuda:0', grad_fn=<ToCopyBackward0>)
        #     torch.float32
        loss_ce = None
        loss_rk = None
        ### split the logits torch.Size([21, 212, 128256])\input_ids torch.Size([21, 212])\attention_mask torch.Size([21, 212])\labels torch.Size([21, 212])
        logits_ce = logits[:self.train_batch_size, :, :] #### torch.Size([2, 212, 128256])
        logits_rk = logits[self.train_batch_size:, :, :] #### torch.Size([12, 212, 128256])
        # input_ids_ce = input_ids[:self.num_candidates, :]
        # input_ids_rk = input_ids[self.num_candidates:, :]
        # attention_mask_ce = attention_mask[:self.num_candidates, :]
        # attention_mask_rk = attention_mask[self.num_candidates:, :]
        labels_ce = labels[:self.train_batch_size, :] #### torch.Size([2, 212])
        labels_rk = labels[self.train_batch_size:, :] #### torch.Size([12, 212])

        # ### compute the cross entropy loss
        if labels_ce is not None:
            # Shift so that tokens < n predict n
            shift_logits_ce = logits_ce[..., :-1, :].contiguous() #### 2,211,128256
            shift_labels_ce = labels_ce[..., 1:].contiguous() #### 2,211
            # Flatten the tokens
            loss_ce_fct = CrossEntropyLoss()
            shift_logits_ce = shift_logits_ce.view(-1, self.config.vocab_size) #### torch.Size([422, 128256])
            shift_labels_ce = shift_labels_ce.view(-1) #### torch.Size([422])
            # Enable model parallelism
            shift_labels_ce = shift_labels_ce.to(shift_logits_ce.device) #### torch.Size([422])
            loss_ce = loss_ce_fct(shift_logits_ce, shift_labels_ce) #### torch.Size([]) tensor(0.4808, device='cuda:0', grad_fn=<NllLossBackward0>)

            # shift_labels_ce
            # tensor([  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         88766,    284,     34,    356,    356,    356,    284,     34,    356,
            #         26176,     17,  22249,     16,    507,    328,    284,  18697,     16,
            #         356,    284,     46,    284,  18697,     16,    356,    284,     46,
            #         452,    356,     31,     39,     16,    356,    356,  19741,     39,
            #             16,    356,    356,    356,    356,    284,     34,    356,    284,
            #             34,    356,  22249,     16,    507,    284,     34,  22249,     16,
            #         284,  18697,     16,  22249,     16,    674,  18697,     17,    284,
            #             34,    356,    284,     34,  22249,     17,  22249,     16,    284,
            #         18697,     16,    452,  22249,     17,  22249,     16,    674,  18697,
            #             17,      6, 128001,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,  88766,    284,     34,  26176,     17,
            #         22249,     16,    284,     34,    452,    356,    356,     31,     39,
            #             16,    356,    356,    356,    356,    452,  22249,     16,    284,
            #         18697,     16,    356,    284,     45,    356,    284,     45,    356,
            #         284,     34,  22249,     16,    284,  18697,     16,    507,    356,
            #         284,     34,    356,    284,     34,    356,    284,     34,  22249,
            #             16,    284,  18697,     16,  22249,     16,    284,  18697,     17,
            #         356,    284,     34,    356,    284,     34,    356,    284,     34,
            #         22249,     16,    284,  18697,     16,      6, 128001,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100],
            #     device='cuda:0')

        # compute the ranking loss 
        logits_rk_log_prob = F.log_softmax(logits_rk, dim=-1) #### torch.Size([12, 212, 128256])
        if labels_rk is not None:
            shift_logits_rk_log_prob = logits_rk_log_prob[..., :-1, :].contiguous() #### torch.Size([12, 211, 128256])
            shift_labels_rk = labels_rk[..., 1:].contiguous() #### torch.Size([12, 211])
            # 创建一个掩码来过滤掉 -100 的值
            # mask = shift_labels_rk != 128138
            mask = (shift_labels_rk != self.candidate_label_pad_token_id )
            # torch.Size([12, 211])
            # mask
            # tensor([[False, False, False,  ..., False, False, False],
            #         [False, False, False,  ..., False, False, False],
            #         [False, False, False,  ...,  True,  True, False],
            #         ...,
            #         [False, False, False,  ..., False, False, False],
            #         [False, False, False,  ..., False, False, False],
            #         [False, False, False,  ..., False, False, False]], device='cuda:0')

            penalty_length = mask.sum(dim=1) ### penalty_length = tensor([ 98,  96, 111,  88,  83,  91,  97, 103,  74, 105,  85,  82], device='cuda:0') torch.Size([12])

            # 使用掩码只选择有效的值
            # filtered_shift_labels_rk = shift_labels_rk[mask]
            # filtered_shift_logits_rk_log_prob = shift_logits_rk_log_prob[mask.unsqueeze(-1).expand_as(shift_logits_rk_log_prob)]

            # 使用 gather 方法获取下一个 token 的对数概率
            # next_token_log_prob = filtered_shift_logits_rk_log_prob.gather(2, filtered_shift_labels_rk.unsqueeze(-1)).squeeze(-1)
            next_token_log_prob = shift_logits_rk_log_prob.gather(2, shift_labels_rk.unsqueeze(-1)).squeeze(-1) ### -100 怎么办？能提取到位置吗，替换成了reserved 133，即128138 #### shift_labels_rk.unsqueeze(-1).shape = torch.Size([12, 211, 1]) ### torch.Size([12, 211])

            next_token_log_prob = next_token_log_prob * mask.float() ### torch.Size([12, 211])

            # mean_next_token_log_prob = next_token_log_prob.mean(dim=1) ### /(D+S) --> /S
            ### length penalty
            sum_penalty_next_token_log_prob = next_token_log_prob.sum(dim=1) / (penalty_length * penalty_length) ### torch.Size([12])

            candidates_scores = self.reshape_tensor(sum_penalty_next_token_log_prob, self.train_batch_size, self.num_candidates).to(self.device) ### biubiubiu ? torch.Size([2, 6])

            loss_rk_split = self.compute_ranking_loss(candidates_scores) ### torch.Size([2, 6, 6])
            # self.brio_logger.info(f"The ranking loss device is: {loss_rk.device}") ### 
            loss_rk = torch.sum(loss_rk_split)

        total_loss = (0.1 * loss_ce) + (10.0 * loss_rk)
        self.brio_logger.info(f"model device: {self.device}, cross_entropy_loss: {loss_ce}, ranking_loss: {loss_rk}, total loss: {total_loss},\nc_e loss device: {loss_ce.device}, r_k loss device: {loss_rk.device}, total loss device: {total_loss.device}") ### ranking_loss_sum = tensor(0.2904, device='cuda:0', grad_fn=<SumBackward0>)
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits_ce,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # if labels is not None: ### labels.shape = torch.Size([21, 212])
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()       ### torch.Size([3, 211, 128256])
        #     shift_labels = labels[..., 1:].contiguous()       ### torch.Size([3, 211])
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)       ### torch.Size([633, 128256])
        #     shift_labels = shift_labels.view(-1)   ### torch.Size([633])
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    ### biubiubiu 适应不满batchsize的情况
    def reshape_tensor(self, tensor, train_batch_size, num_candidates):
        # 获取tensor的实际长度
        actual_length = tensor.shape[0]
        # 计算实际的batch数量
        # actual_batchsize = actual_length // num_candidates ### 不行！如果是batchsize设置为1，则这里计算的actual_batchsize就是0了，全部出错！
        # 考虑到，数据集里actual_length永远是num_candidates的整数倍，因此可以改成
        # actual_batchsize = actual_length / num_candidates ### 也不行，因为返回的是float浮点数
        actual_batchsize = int(actual_length / num_candidates)
        # remaining_elements = actual_length % num_candidates
        # 如果剩余元素大于0，则需要额外处理一个batch
        # total_batchsize = actual_batchsize + (1 if remaining_elements > 0 else 0)

        # 初始化一个空的张量来存储结果
        reshaped_tensor = torch.empty(actual_batchsize, num_candidates, dtype=tensor.dtype)
        if actual_batchsize == train_batch_size:
            self.brio_logger.info(f"In normal mode, the empty reshaped_tensor is:\n{reshaped_tensor}")
            self.brio_logger.info(f"In normal mode, the shape of empty reshaped_tensor is:{reshaped_tensor.shape}")
            self.brio_logger.info(f"------------")
            self.brio_logger.info(f"In normal mode, the shape of the unreshaped input tensor is:\n{tensor} ")
            self.brio_logger.info(f"In normal mode, the shape of the unreshaped input tensor is: {tensor.shape}")
            self.brio_logger.info(f"In normal mode, the setting train_batch_size = {train_batch_size}")
            self.brio_logger.info(f"In normal mode, the setting num_candidates = {num_candidates}")
        else:
            self.brio_logger.info(f"In abnormal mode, this batch is smaller, there are only {actual_batchsize} here!")
            self.brio_logger.info(f"In abnormal mode, the tensor is:\n{tensor} ")
            self.brio_logger.info(f"In abnormal mode, the shape of the tensor is: {tensor.shape}")
            self.brio_logger.info(f"In abnormal mode, the setting train_batch_size = {train_batch_size}")
            self.brio_logger.info(f"In abnormal mode, the setting num_candidates = {num_candidates}")

        for i in range(actual_batchsize):
            reshaped_tensor[i] = tensor[i::actual_batchsize]
            # if i < actual_batchsize:
            #     reshaped_tensor[i] = tensor[i * num_candidates: (i + 1) * num_candidates]
            # else:
            #     self.brio_logger.info(f"ASSERT i will never come here!")
            #     reshaped_tensor[i, :remaining_elements] = tensor[i * num_candidates:]
            #     reshaped_tensor[i, remaining_elements:] = torch.tensor([float('nan')] * (num_candidates - remaining_elements), dtype=tensor.dtype)

        assert reshaped_tensor.shape[0] == actual_batchsize, "reshaped Tensor wrong."
        assert reshaped_tensor.shape[1] == num_candidates, "reshaped Tensor wrong again."
        self.brio_logger.info(f"The filled reshaped_tensor is:\n{reshaped_tensor}")

        return reshaped_tensor

    def compute_ranking_loss(self, reshaped_scores):
        # train_batch_size = self._train_batch_size
        # number_of_candidates = self.num_candidates
        # reshaped_scores = self.reshape_tensor(scores, self._train_batch_size, self.num_candidates).to(scores.device)
        # reshaped_scores = self.reshape_tensor(scores, self._train_batch_size, self.num_candidates).to(model.device) ### 没有model，不应该在这里reshape
        self.brio_logger.info(f"------------------------------------- computing ranking loss now ---------------------------------------------")
        # self.brio_logger.info(f"The model device is: {model.device}") ### 想啥呢，没有model
        self.brio_logger.info(f"When computing ranking loss, the device is: {reshaped_scores.device}")
        self.brio_logger.info(f"The reshaped scores are:\n{reshaped_scores}") ### 到这一步，都是多device都有的
        self.brio_logger.info(f"The shape of the reshaped socres are: {reshaped_scores.shape}")

        # ranking_loss = nn.MarginRankingLoss()
        ranking_loss = nn.MarginRankingLoss(size_average=False, reduce=False, reduction='none')
        self.brio_logger.info(f"The original ranking losses obj is:{ranking_loss}")
        
        # 利用广播技术
        # 原张量是一维时
        # x1 = reshaped_scores.unsqueeze(1)  # 转换为列向量
        # self.brio_logger.info(f"The liexiangliang of reshaped scores are: {x1}")
        # x2 = reshaped_scores.unsqueeze(0)  # 转换为行向量
        # self.brio_logger.info(f"The hangxiangliang of reshaped scores are: {x2}")
        # 原张量是二维时
        x1 = reshaped_scores.unsqueeze(2)  # 在第二维度上增加一个维度，变成列向量
        self.brio_logger.info(f"The liexiangliang of reshaped scores are:\n{x1}, device is: {x1.device}")
        x2 = reshaped_scores.unsqueeze(1)  # 在第一维度上增加一个维度，变成行向量
        self.brio_logger.info(f"The hangxiangliang of reshaped scores are:\n{x2}, device is: {x2.device}")

        # 创建标签张量，y 的值为 1，因为我们希望 x1 > x2
        y = torch.ones_like(x1 - x2)
        # 这样，通过广播，x1 - x2 的操作会自动扩展到 (L, L) 的形状，而 y 作为标签张量，每个元素对应着 x1 和 x2 对的比较关系。
        self.brio_logger.info(f"The broadcast y is:\n{y}, device is: {y.device}")

        loss_matrix = ranking_loss(x1, x2, y)
        self.brio_logger.info(f"The ranking loss of all batches are:\n{loss_matrix}, device is: {loss_matrix.device}")

        # 创建掩码 mask，上三角矩阵
        mask_scores = torch.triu(torch.ones_like(y), diagonal=1)
        self.brio_logger.info(f"The mask is:\n{mask_scores}, device is: {mask_scores.device}")

        # 过滤结果，x1的index小于 x2的index
        filtered_loss = loss_matrix * mask_scores
        self.brio_logger.info(f"The filtered ranking loss of all batches are:\n{filtered_loss}, device is: {filtered_loss.device}")

        return filtered_loss
    
    def _setup_brio_logger(self, brio_log_file_path):
        # 创建输出到文件的 logger
        file_handler = FileHandler(brio_log_file_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.brio_logger = logging.getLogger('BRIO_logger')
        self.brio_logger.addHandler(file_handler)
        self.brio_logger.setLevel(logging.DEBUG)
### biubiubiu 35 one forward


MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    # "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    # "llama": (AutoConfig, LlamaForCausalLM, AutoTokenizer), ### biubiubiu 35 特有
    "llama": (AutoConfig, LlamaForCausalLMBrio, AutoTokenizer), ### biubiubiu 35 特有 one forward
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


# ### brio candidate Class dataset
# class CandidateDataset(torch.utils.data.Dataset):
#     def __init__(self,
#                 tokenizer, finetune_path, max_len=100, is_sorted=True
#                 ):
#         self.tokenizer = tokenizer
#         self.finetune_path = finetune_path
#         self.data = pd.read_csv(self.finetune_path)
#         self.maxlen = max_len
#         self.sorted = is_sorted
#         self.inputs = self.data['input'].tolist()
#         self.candidate_inputs = self.data['candidates'].tolist()

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         input = self.inputs[index]
#         src = self.tokenizer.batch_encode_plus([input], max_length=self.maxlen, return_tensors="pt", padding="max_length", truncation=True)
#         src_input_ids = src["input_ids"]
#         src_input_ids = src_input_ids.squeeze(0)
            
#         candidate = self.candidate_inputs[index]
#         candidate = eval(candidate)
        
#         if self.sorted:
#             candidate = sorted(candidate, key=lambda x:x[1], reverse=True) ## 倒序
#         candidates = [x[0] for x in candidate]
#         cand = self.tokenizer.batch_encode_plus(candidates, max_length=self.maxlen, return_tensors="pt", padding="max_length", truncation=True)
#         candidate_ids = cand["input_ids"]
#         result = {
#             "src_input_ids": src_input_ids, 
#             "candidate_ids": candidate_ids,
#             }
#         return result

# ### 
# 定义自定义参数的数据类
@dataclass
class CustomArguments:
    # finetune_path: str = field(default=None, metadata={"help": "."})
    # custom_arg2: int = field(default=123, metadata={"help": "."})
    brio_log_file_path: str = field(default=None, metadata={"help": "."})
    brio_candidate_labels_pad_id: int = field(default=128138, metadata={"help": "."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # use_fast_tokenizer: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    # )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        # metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."}
        metadata={"help": "Enable shifted sparse attention (S^2-Attn) proposed by LongLoRA."}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "The alpha parameter to control the noise magnitude in NEFTune. value can be 5."}
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    MODEL_CLASSES.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."})
    # template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("You may set max_train_samples = -1 to run all samples in production.")

@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on inputs"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum model context length. suggest: 8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"}
    )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."}) ### 35 update

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")


# class CastOutputToFloat(torch.nn.Module):
#     """Cast the output of the model to float"""

#     def __init__(self, ori_linear: torch.nn.Linear) -> None:
#         super().__init__()
#         self.in_features = ori_linear.in_features
#         self.out_features = ori_linear.out_features
#         self.weight = ori_linear.weight
#         if ori_linear.bias is not None:
#             self.bias = ori_linear.bias
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, input):
#         return torch.nn.functional.linear(input, self.weight, self.bias).to(torch.float32)

#     def extra_repr(self) -> str:
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )


# # Copied from: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/extras/patches/llama_patch.py
# class LlamaShiftShortAttention(LlamaAttention):

#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_value: Optional[Tuple[torch.Tensor]] = None,
#             output_attentions: bool = False,
#             use_cache: bool = False,
#             **kwargs
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]

#         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#         if past_key_value is not None:  # reuse k, v, self_attention
#             key_states = torch.cat([past_key_value[0], key_states], dim=2)
#             value_states = torch.cat([past_key_value[1], value_states], dim=2)

#         past_key_value = (key_states, value_states) if use_cache else None

#         if getattr(self, "num_key_value_groups"):
#             key_states = repeat_kv(key_states, self.num_key_value_groups)
#             value_states = repeat_kv(value_states, self.num_key_value_groups)

#         if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
#             groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
#             assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
#             num_groups = q_len // groupsz

#             def shift(state: torch.Tensor) -> torch.Tensor:
#                 state = state.transpose(1, 2)  # output: (bsz, seq_len, n_heads, head_dim)
#                 state = torch.cat((
#                     state[:, :, :self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)
#                 ), dim=2)
#                 return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)

#             query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
#             if attention_mask is not None:
#                 attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attention_mask is not None:
#             attn_weights = attn_weights + attention_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_output = torch.matmul(attn_weights, value_states)  # (bsz, :, seq_len, :) or (bsz*n_group, :, groupsz, :)
#         attn_output = attn_output.transpose(1, 2).contiguous()

#         if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
#             groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
#             attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
#             attn_output = torch.cat((
#                 attn_output[:, :, :self.num_heads // 2],
#                 attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1)
#             ))

#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# class LlamaFlashAttention2(LlamaAttention):

#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_value: Optional[Tuple[torch.Tensor]] = None,
#             output_attentions: bool = False,
#             use_cache: bool = False,
#             **kwargs
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         # LlamaFlashAttention2 attention does not support output_attentions
#         output_attentions = False

#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]

#         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#         if past_key_value is not None:  # reuse k, v, self_attention
#             key_states = torch.cat([past_key_value[0], key_states], dim=2)
#             value_states = torch.cat([past_key_value[1], value_states], dim=2)

#         past_key_value = (key_states, value_states) if use_cache else None

#         # cast to half precision
#         input_dtype = query_states.dtype
#         if input_dtype == torch.float32:
#             logger.warning("The input hidden states seems to be silently casted in float32.")
#             query_states = query_states.to(self.config.torch_dtype)
#             key_states = key_states.to(self.config.torch_dtype)
#             value_states = value_states.to(self.config.torch_dtype)

#         if getattr(self, "num_key_value_groups", None):
#             key_states = repeat_kv(key_states, self.num_key_value_groups)
#             value_states = repeat_kv(value_states, self.num_key_value_groups)

#         query_states = query_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
#         key_states = key_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
#         value_states = value_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)

#         if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
#             groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
#             assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
#             num_groups = q_len // groupsz

#             def shift(state: torch.Tensor) -> torch.Tensor:
#                 state = torch.cat((
#                     state[:, :, :self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)
#                 ), dim=2)
#                 return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)

#             query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.reshape(bsz * num_groups, groupsz)

#         if attention_mask is not None:
#             logger.warning("Padded sequences are less efficient in FlashAttention.")
#             # -q_len: assumes left padding when q_len != kv_len
#             unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_states, attention_mask[:, -q_len:])
#             unpadded_k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_states, attention_mask)
#             unpadded_v, _, _, _ = unpad_input(value_states, attention_mask)
#             attn_output_unpad = flash_attn_varlen_func(
#                 unpadded_q,
#                 unpadded_k,
#                 unpadded_v,
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_k=cu_seqlens_k,
#                 max_seqlen_q=max_seqlen_q,
#                 max_seqlen_k=max_seqlen_k,
#                 dropout_p=0.0,
#                 softmax_scale=None,
#                 causal=True,
#             )
#             attn_output = pad_input(attn_output_unpad, indices_q, bsz, q_len)
#         else:
#             attn_output = flash_attn_func(
#                 query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True
#             )

#         if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
#             groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
#             attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
#             attn_output = torch.cat((
#                 attn_output[:, :, :self.num_heads // 2],
#                 attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1)
#             ))

#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean padding_mask. Fills in the past kv length for use in forward.
# def _prepare_decoder_attention_mask(
#         self,
#         attention_mask: torch.Tensor,
#         input_shape: torch.Tensor,
#         inputs_embeds: torch.Tensor,
#         past_key_values_length: int
# ) -> torch.Tensor:
#     if attention_mask is not None and torch.all(attention_mask):
#         return None  # This uses the faster call when training with full samples

#     return attention_mask


# @dataclass
# class Conversation:
#     """A class that manages prompt templates and keeps all conversation history."""

#     # The name of this template
#     name: str
#     # The system prompt
#     system_prompt: str
#     # All messages. format: list of [question, answer]
#     messages: Optional[List[Sequence[str]]]
#     # The roles of the speakers
#     roles: Optional[Sequence[str]]
#     # Conversation prompt
#     prompt: str
#     # Separator
#     sep: str
#     # Stop token, default is tokenizer.eos_token
#     stop_str: Optional[str] = "</s>"

#     def get_prompt(
#             self,
#             messages: Optional[List[Sequence[str]]] = None,
#             system_prompt: Optional[str] = ""
#     ) -> str:
#         """
#         Returns a string containing prompt without response.
#         """
#         return "".join(self._format_example(messages, system_prompt))

#     def get_dialog(
#             self,
#             messages: Optional[List[Sequence[str]]] = None,
#             system_prompt: Optional[str] = ""
#     ) -> List[str]:
#         """
#         Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
#         """
#         return self._format_example(messages, system_prompt)

#     def _format_example(
#             self,
#             messages: Optional[List[Sequence[str]]] = None,
#             system_prompt: Optional[str] = ""
#     ) -> List[str]:
#         system_prompt = system_prompt or self.system_prompt
#         system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt
#         messages = messages or self.messages
#         convs = []
#         for turn_idx, [user_query, bot_resp] in enumerate(messages):
#             if turn_idx == 0:
#                 convs.append(system_prompt + self.prompt.format(query=user_query))
#                 convs.append(bot_resp)
#             else:
#                 convs.append(self.sep + self.prompt.format(query=user_query))
#                 convs.append(bot_resp)
#         return convs

#     def append_message(self, query: str, answer: str):
#         """Append a new message."""
#         self.messages.append([query, answer])


# # A global registry for all conversation templates
# conv_templates: Dict[str, Conversation] = {}


# def register_conv_template(template: Conversation):
#     """Register a new conversation template."""
#     conv_templates[template.name] = template


# """Vicuna v1.1 template
# Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
#           https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
# """
# register_conv_template(
#     Conversation(
#         name="vicuna",
#         system_prompt="A chat between a curious user and an artificial intelligence assistant. "
#                       "The assistant gives helpful, detailed, and polite answers to the user's questions. "
#                       "Generate a molecule with target property.",
#         messages=[],
#         roles=("USER", "ASSISTANT"),
#         prompt="USER: {query} ASSISTANT:",
#         sep="</s>",
#     )
# )

# """Alpaca template"""
# register_conv_template(
#     Conversation(
#         name="alpaca",
#         system_prompt="Below is an instruction that describes a task. "
#                       "Write a response that appropriately completes the request.",
#         messages=[],
#         roles=("### Instruction", "### Response"),
#         prompt="### Instruction:\n{query}\n\n### Response:\n",
#         sep="\n\n",
#     )
# )

# """Baichuan template
# source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py#L31
# Support: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
# """
# register_conv_template(
#     Conversation(
#         name="baichuan",
#         system_prompt="",
#         messages=[],
#         roles=("<reserved_102>", "<reserved_103>"),
#         prompt="<reserved_102>{query}<reserved_103>",
#         sep="</s>",
#     )
# )

# """Baichuan2 template
# Support: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
#          https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
# """
# register_conv_template(
#     Conversation(
#         name="baichuan2",
#         system_prompt="",
#         messages=[],
#         roles=("<reserved_106>", "<reserved_107>"),
#         prompt="<reserved_106>{query}<reserved_107>",
#         sep="</s>",
#     )
# )

# """ziya template"""
# register_conv_template(
#     Conversation(
#         name="ziya",
#         system_prompt="",
#         messages=[],
#         roles=("<human>", "<bot>"),
#         prompt="<human>:{query}\n<bot>:",
#         sep="\n",
#     )
# )

# """Linly template"""
# register_conv_template(
#     Conversation(
#         name="linly",
#         system_prompt="",
#         messages=[],
#         roles=("User", "Bot"),
#         prompt="User: {query}\nBot: ",
#         sep="\n",
#     )
# )

# """ChatGLM1 template
# Support: https://huggingface.co/THUDM/chatglm-6b
# source: https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1307
# """
# register_conv_template(
#     Conversation(
#         name="chatglm",
#         system_prompt="",
#         messages=[],
#         roles=("问", "答"),
#         prompt="问：{query}\n答：",
#         sep="\n",
#     )
# )

# """ChatGLM2 template
# Support: https://huggingface.co/THUDM/chatglm2-6b
# source: https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
# """
# register_conv_template(
#     Conversation(
#         name="chatglm2",
#         system_prompt="",
#         messages=[],
#         roles=("问", "答"),
#         prompt="问：{query}\n\n答：",
#         sep="\n\n",
#     )
# )

# """ChatGLM3 template
# Support: https://huggingface.co/THUDM/chatglm3-6b
# source: https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py#L179
# """
# register_conv_template(
#     Conversation(
#         name="chatglm3",
#         system_prompt="",
#         messages=[],
#         roles=("<|user|>", "<|assistant|>"),
#         prompt="<|user|>\n{query}<|assistant|>",
#         sep="\n",
#         stop_str="<|user|>",
#     )
# )

# """Phoenix template"""
# register_conv_template(
#     Conversation(
#         name="phoenix",
#         system_prompt="A chat between a curious human and an artificial intelligence assistant. "
#                       "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
#         messages=[],
#         roles=("Human", "Assistant"),
#         prompt="Human: <s>{query}</s>Assistant: ",
#         sep="</s>",
#     )
# )

# """belle template
# Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
# """
# register_conv_template(
#     Conversation(
#         name="belle",
#         system_prompt="",
#         messages=[],
#         roles=("Human", "Belle"),
#         prompt="Human: {query}\n\nBelle: ",
#         sep="\n\n",
#     )
# )

# """aquila template
# Supports: https://huggingface.co/qhduan/aquilachat-7b
#           https://huggingface.co/BAAI/AquilaChat2-34B
# """
# register_conv_template(
#     Conversation(
#         name="aquila",
#         system_prompt="A chat between a curious human and an artificial intelligence assistant. "
#                       "The assistant gives helpful, detailed, and polite answers to the human's questions.",
#         messages=[],
#         roles=("Human", "Assistant"),
#         prompt="Human: {query}###Assistant:",
#         sep="###",
#     )
# )

# """intern template
# Supports: https://huggingface.co/internlm/internlm-chat-7b
#           https://huggingface.co/internlm/internlm-chat-20b
# """
# register_conv_template(
#     Conversation(
#         name="intern",
#         system_prompt="",
#         messages=[],
#         roles=("<|User|>", "<|Bot|>"),
#         prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
#         sep="<eoa>\n",
#         stop_str="<eoa>",
#     )
# )

# """StarChat template
# Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
#           https://huggingface.co/HuggingFaceH4/starchat-beta
# """
# register_conv_template(
#     Conversation(
#         name="starchat",
#         system_prompt="<system>\n",
#         messages=[],
#         roles=("<|user|>", "<|assistant|>"),
#         prompt="<|user|>\n{query}<|end|>\n<|assistant|>\n",
#         sep="<|end|>\n",
#         stop_str="<|end|>",
#     )
# )

# """llama2 template
# Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#           https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
#           https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
# reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
# """
# register_conv_template(
#     Conversation(
#         name="llama2",
#         system_prompt="<<SYS>>\nYou are a helpful, respectful and honest assistant. "
#                       "Always answer as helpfully as possible, while being safe. "
#                       "Your answers should not include any harmful, unethical, racist, sexist, "
#                       "toxic, dangerous, or illegal content. "
#                       "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
#                       "If a question does not make any sense, or is not factually coherent, "
#                       "explain why instead of answering something not correct. "
#                       "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
#         messages=[],
#         roles=("[INST]", "[/INST]"),
#         prompt="[INST] {query} [/INST]",
#         sep="</s>",
#     )
# )

# """llama2-zh template
# source: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
# Supports: https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
# """
# register_conv_template(
#     Conversation(
#         name="llama2-zh",
#         system_prompt="[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST]",
#         messages=[],
#         roles=("[INST]", "[/INST]"),
#         prompt="[INST] {query} [/INST]",
#         sep="</s>",
#     )
# )

# """mistral template
# Supports: https://huggingface.co/mistralai/Mistral-7B-v0.1
#           https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1
# """
# register_conv_template(
#     Conversation(
#         name="mistral",
#         system_prompt="<s>",
#         messages=[],
#         roles=("[INST]", "[/INST]"),
#         prompt="[INST] {query} [/INST]",
#         sep="</s>",
#     )
# )

# """XVERSE template
# Supports: https://huggingface.co/xverse/XVERSE-13B-Chat
# """
# register_conv_template(
#     Conversation(
#         name="xverse",
#         system_prompt="",
#         messages=[],
#         roles=("Human", "Assistant"),
#         prompt="Human: {query}\n\nAssistant: ",
#         sep="</s>",
#     )
# )

# """Qwen template
# Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
# chatml: https://xbot123.com/645a461b922f176d7cfdbc2d/
# """
# register_conv_template(
#     Conversation(
#         name="chatml",
#         system_prompt="You are a helpful assistant.",
#         messages=[],
#         roles=("user", "assistant"),
#         prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
#         sep="<|im_end|>\n",
#         stop_str="<|im_end|>",
#     )
# )

# def get_conv_template(name: str) -> Conversation:
#     """Get a conversation template."""
#     return conv_templates[name]

### biubiubiu
# class DataCollatorForSeq2SeqBrio(DataCollatorForSeq2Seq):
#     def __init__(self, num_candidates, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_candidates = num_candidates
class DataCollatorForSeq2SeqBrio(DataCollatorForSeq2Seq):
    # def __init__(self, num_candidates, brio_log_file_path, candidates_input_ids_pad_token_id, *args, **kwargs):
    def __init__(self, num_candidates, brio_logger, candidates_input_ids_pad_token_id, label_pad_token_id, candidate_label_pad_token_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_candidates = num_candidates
        # self.brio_logger = self._setup_brio_logger(brio_log_file_path)
        # 创建新的 logger
        # self._setup_brio_logger(brio_log_file_path)
        self.brio_logger = brio_logger
        self.brio_logger.info("now, i'm in datacollatorforseq2seqbrio.")
        self.candidates_input_ids_pad_token_id = candidates_input_ids_pad_token_id ### biubiubiu
        self.label_pad_token_id = label_pad_token_id
        self.candidate_label_pad_token_id = candidate_label_pad_token_id
    # tokenizer: PreTrainedTokenizerBase
    # num_candidates: int
    # model: Optional[Any] = None
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100 ### 父类DataCollatorForSeq2Seq里的init部分，self.label_pad_token_id = -100
    # candidates_input_ids_pad_token_id: int = 0 ### biubiubiu
    # return_tensors: str = "pt"

    # def _setup_brio_logger(self, brio_log_file_path):
    #     # 创建输出到文件的 logger
    #     file_handler = FileHandler(brio_log_file_path)
    #     file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     file_handler.setFormatter(file_formatter)
    #     brio_logger = logging.getLogger('BRIO_logger')
    #     brio_logger.addHandler(file_handler)
    #     brio_logger.setLevel(logging.DEBUG)
    #     return brio_logger

    def _setup_brio_logger(self, brio_log_file_path):
        # 创建输出到文件的 logger
        file_handler = FileHandler(brio_log_file_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.brio_logger = logging.getLogger('BRIO_logger')
        self.brio_logger.addHandler(file_handler)
        self.brio_logger.setLevel(logging.DEBUG)

    def __call__(self, features, return_tensors=None): ### 什么时候会进入call? sample = next(iter(trainer.get_train_dataloader())) 这里, ### 以及需要一个batch一个batch蹦出数据的时候会来
        self.brio_logger.info("now, i'm in datacollatorforseq2seqbrio's call function")
        if return_tensors is None:
            return_tensors = self.return_tensors  ### features 是一个batch的数据，list形式存放三个字典[{'input_ids': [...], 'attention_mask': [...], 'labels': [...], 'candidate_1_input_ids': [...], 'candidate_2_input_ids': [...], 'candidate_3_input_ids': [...], 'candidate_4_input_ids': [...], 'candidate_5_input_ids': [...], 'candidate_6_input_ids': [...]}, {'input_ids': [...], 'attention_mask': [...], 'labels': [...], 'candidate_1_input_ids': [...], 'candidate_2_input_ids': [...], 'candidate_3_input_ids': [...], 'candidate_4_input_ids': [...], 'candidate_5_input_ids': [...], 'candidate_6_input_ids': [...]}, {'input_ids': [...], 'attention_mask': [...], 'labels': [...], 'candidate_1_input_ids': [...], 'candidate_2_input_ids': [...], 'candidate_3_input_ids': [...], 'candidate_4_input_ids': [...], 'candidate_5_input_ids': [...], 'candidate_6_input_ids': [...]}]
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        # self.brio_logger.info("now, i'm in datacollatorforseq2seqbrio's call function")
        if labels is not None:### max_label_length需要是4的倍数
            self.brio_logger.info(f"Get in to process wendadui")
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"])) ### self.label_pad_token_id = -100 by default
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        ### biubiubiu one forward pad candidates的input_ids（0pad  self.candidates_input_ids_pad_token_id）和attention_mask（0pad  self.candidates_input_ids_pad_token_id）和labels（-100pad  self.label_pad_token_id = -100）
        for candidate_idx in range(1, self.num_candidates + 1): ### 每个 candidate_idx 单独处理其三个东西
            candidate_input_ids_key = f"candidate_{candidate_idx}_input_ids" ### 其实应该和上面一样用candidate_{candidate_idx}_labels这个键
            candidate_attention_mask_key = f"candidate_{candidate_idx}_attention_mask"
            candidate_labels_key = f"candidate_{candidate_idx}_labels"

            candidate_input_ids_list = [feature[candidate_input_ids_key] for feature in features] if candidate_input_ids_key in features[0].keys() else None
            if candidate_input_ids_list is not None:
                self.brio_logger.info(f"Get in to process candidates")
                max_candidate_input_ids_length = max(len(ids) for ids in candidate_input_ids_list)
                if self.pad_to_multiple_of is not None:
                    max_candidate_input_ids_length = (
                        (max_candidate_input_ids_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features: ### batch的for循环，### 看看此时的features里有什么键：在main的preprocess_function_train函数里添加好了list
                    reminder_input_ids = [self.candidates_input_ids_pad_token_id] * (max_candidate_input_ids_length - len(feature[candidate_input_ids_key]))
                    reminder_labels = [self.candidate_label_pad_token_id] * (max_candidate_input_ids_length - len(feature[candidate_labels_key]))
                    if isinstance(feature[candidate_input_ids_key], list):
                        feature[candidate_input_ids_key] = (
                            feature[candidate_input_ids_key] + reminder_input_ids if padding_side == "right" else reminder_input_ids + feature[candidate_input_ids_key]
                        ) ### here
                        feature[candidate_attention_mask_key] = (
                            feature[candidate_attention_mask_key] + reminder_input_ids if padding_side == 'right' else reminder_input_ids + feature[candidate_attention_mask_key]
                        )
                        feature[candidate_labels_key] = (
                            feature[candidate_labels_key] + reminder_labels if padding_side == 'right' else reminder_labels + feature[candidate_labels_key]
                        )
                    # elif padding_side == "right":
                    #     feature[candidate_input_ids_key] = np.concatenate([feature[candidate_input_ids_key], reminder_input_ids]).astype(np.int64)
                    # else:
                    #     feature[candidate_input_ids_key] = np.concatenate([reminder_input_ids, feature[candidate_input_ids_key]]).astype(np.int64)
### biubiubiu 在此之前，只对一个minibatch里的labels和candidate_*_input_ids/attention_mask/labels进行了padding，各自minibatch里最长的且是4的倍数
        self.brio_logger.info(f"I have processed the padding of the wendadui's label_ids and things three things about candidates, and i'm going to pad the wendadui's atten and inputs")
        features = self.tokenizer.pad( ### ***第五步*** 进入/data1/fcl/software/conda3/envs/matllmbrio/lib/python3.9/site-packages/transformers/tokenization_utils_base.py
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        ### 之后的features不再是先batch再键了，而是先键后batch，主要是        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis) ### here 这个函数
        ### /data/fcl/anaconda3/envs/matllm/lib/python3.9/site-packages/transformers/tokenization_utils_base.py   class BatchEncoding(UserDict):
        # features["input_ids"].size() = torch.Size([3, 184])
        # features["attention_mask"].size() = torch.Size([3, 184])
        # features["labels"].size() = torch.Size([3, 184])
        # features["candidate_1_input_ids"].size() = torch.Size([3, 204])
        # features["candidate_1_attention_mask"].size() = torch.Size([3, 204])
        # features["candidate_1_labels"].size() = torch.Size([3, 204])
        # features["candidate_5_input_ids"].size() = torch.Size([3, 208])
        # features["candidate_5_attention_mask"].size() = torch.Size([3, 208])
        # features["candidate_5_labels"].size() = torch.Size([3, 208])
### biubiubiu 在此之后，查看
        # prepare decoder_input_ids 判断完不进入
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features ### here
### biubiubiu

# class SavePeftModelTrainer(Trainer):
#     """
#     Trainer for lora models
#     """

#     def save_model(self, output_dir=None, _internal_call=False):
#         """Save the LoRA model."""
#         os.makedirs(output_dir, exist_ok=True)
#         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
#         self.model.save_pretrained(output_dir)

#######biubiubiu#######
# class SavePeftModelAndBrioTrainer(Trainer):
#     """
#     Trainer for lora models and for brio sft training
#     """
#     def __init__(self, brio_log_file_path, num_candidates, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_candidates = num_candidates
#         # 创建新的 logger
#         self._setup_brio_logger(brio_log_file_path)

#     def _setup_brio_logger(self, brio_log_file_path):
#         # 创建输出到文件的 logger
#         file_handler = FileHandler(brio_log_file_path)
#         file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(file_formatter)
#         self.brio_logger = logging.getLogger('BRIO_logger')
#         self.brio_logger.addHandler(file_handler)
#         self.brio_logger.setLevel(logging.DEBUG)
class SavePeftModelAndBrioTrainer(Trainer):
    """
    Trainer for lora models and for brio sft training
    """
    def __init__(self, num_candidates, brio_logger, input_ids_and_attention_mask_pad_id, labels_pad_id, candidate_labels_pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_candidates = num_candidates
        self.brio_logger = brio_logger
        self.input_ids_and_attention_mask_pad_id = input_ids_and_attention_mask_pad_id
        self.labels_pad_id = labels_pad_id
        self.candidate_labels_pad_id = candidate_labels_pad_id

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)

    # def _set_signature_columns_if_needed(self):
    #     if self._signature_columns is None:
    #         # Inspect model forward signature to keep only the arguments it accepts.
    #         signature = inspect.signature(self.model.forward)
    #         self._signature_columns = list(signature.parameters.keys())
    #         # Labels may be named label or label_ids, the default data collator handles that.
    #         self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
    #         ### 加上我自己需要的这个
    #         ### 动态添加候选人输入ID列表
    #         for j in range(1, self.num_candidates + 1):
    #             self._signature_columns.append(f'candidate_{j}_input_ids')   
    #         # self._signature_columns.append('candidates_ids') ###  签名列表不加上candidates_ids的话,会自动删除用不着的列#########################
    #         # self._signature_columns.append('candidates_input_ids') ###

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
### 确认candidates_ids有没有传进来 biubiubiu
        if 'candidate_1_input_ids' not in self.train_dataset[0]:
            raise ValueError("Training dataset does not contain 'candidates_input_ids' field. Please check your preprocessing function.")
        # else:
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['input_ids'][0])}, and {self.train_dataset['input_ids'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['input_ids'][1])}, and {self.train_dataset['input_ids'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['input_ids'][2])}, and {self.train_dataset['input_ids'][2]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['attention_mask'][0])}, and {self.train_dataset['attention_mask'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['attention_mask'][1])}, and {self.train_dataset['attention_mask'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['attention_mask'][2])}, and {self.train_dataset['attention_mask'][2]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['labels'][0])}, and {self.train_dataset['labels'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['labels'][1])}, and {self.train_dataset['labels'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['labels'][2])}, and {self.train_dataset['labels'][2]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['candidates_ids'][0])}, and {self.train_dataset['candidates_ids'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['candidates_ids'][1])}, and {self.train_dataset['candidates_ids'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example, {len(self.train_dataset['candidates_ids'][2])}, and {self.train_dataset['candidates_ids'][2]}, fclfclfcl")
        train_dataset = self.train_dataset
        # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here2, give you an example{train_dataset['candidates_ids'][0]}, fclfclfcl")
        # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example{self.train_dataset['candidates_ids'][0]}, fclfclfcl")
        # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here1, give you an example{self.train_dataset['candidates_ids'][0]}, fclfclfcl")
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training") ### 下面这个打印显示没有candidates_ids了，所以问题出在了这里，get in
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['input_ids'][0])}, and {train_dataset['input_ids'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['input_ids'][1])}, and {train_dataset['input_ids'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['input_ids'][2])}, and {train_dataset['input_ids'][2]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['attention_mask'][0])}, and {train_dataset['attention_mask'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['attention_mask'][1])}, and {train_dataset['attention_mask'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['attention_mask'][2])}, and {train_dataset['attention_mask'][2]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['labels'][0])}, and {train_dataset['labels'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['labels'][1])}, and {train_dataset['labels'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['labels'][2])}, and {train_dataset['labels'][2]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['candidates_ids'][0])}, and {train_dataset['candidates_ids'][0]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['candidates_ids'][1])}, and {train_dataset['candidates_ids'][1]}, fclfclfcl")
            # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here3, give you an example, {len(train_dataset['candidates_ids'][2])}, and {train_dataset['candidates_ids'][2]}, fclfclfcl")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size, ### 在trainer里，这个是全局的batchsize
            "collate_fn": data_collator,### 这个很重要吧
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers, ### 35的trainer特有
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor ### 35的trainer特有
        # brio_logger.info(f"i have been here in matllmbrio_trainer_py, come and got me!")
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params)) ### 可能这步还没走完,这里需要get in看看是走哪一步 不能直接运行 ***第负零点五步*** 进入DataLoader 转到/data1/fcl/software/conda3/envs/matllmbrio/lib/python3.9/site-packages/torch/utils/data/dataloader.py

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    # PeftMixedModel do not provide a `get_base_model` method
                    model_to_inspect = self.model.base_model.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            ### 加上我自己需要的这个
            # ### biubiubiu
            ### 动态添加候选人输入ID列表
            for j in range(1, self.num_candidates + 1):
                self._signature_columns.append(f'candidate_{j}_input_ids')
                self._signature_columns.append(f'candidate_{j}_attention_mask') ### biubiubiu one forward
                self._signature_columns.append(f'candidate_{j}_labels') ### biubiubiu one forward
            # ### biubiubiu         
            # self._signature_columns.append('candidates_ids') ###  签名列表不加上candidates_ids的话,会自动删除用不着的列#########################
            # # self._signature_columns.append('candidates_input_ids') ###
    
    def _inner_training_loop(## 训练start
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}, fclfclfcl")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader() ## 取Target_list中-100的部分索引idx，input_ids_list[idx]，作为采样数据，送入llama2的model生成候选序列（还是选择了先生成好candidates）
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
        
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        logger.debug(f"total train batch size is: {total_train_batch_size}, fclfclfcl")

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch) ### here
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    # " (torch.distributed.launch)."
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # delay_optimizer_creation = is_sagemaker_mp_enabled() or self.fsdp is not None or self.is_fsdp_enabled
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState()
        # self.state.is_hyper_param_search = trial is not None
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps ### here

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        # if resume_from_checkpoint is not None:
        #     if self.is_deepspeed_enabled:
        #         deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
        #     elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
        #         self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train! ## ### data_collator结束之后，慢慢就到这里了
        logger.info("***** Running training *****")
        # logger.info(f"  Num training examples = {num_examples:,}")
        # logger.info(f"  Num Epochs set= {num_train_epochs:,}")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")  ## 梯度累计
        logger.info(f"  Total optimization steps(number_of_examples / batchsize, fcl) = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            # self.brio_logger.info(f"epochs_trained is {epochs_trained}") ###biubiubiu
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                # if is_torch_less_than_1_11 or not is_random_sampler:
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)
##
        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            logger.info(f"steps_in_epoch is {steps_in_epoch}, fclfclfcl")
            self.brio_logger.info(f"the length of epoch_iterator is: {len(epoch_iterator)}, fclfclfcl")
            self.brio_logger.info(f"steps in epoch is: {steps_in_epoch}, fclfclfcl")
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True
###
            step = -1
            for step, inputs in enumerate(epoch_iterator):### 这里就会yield取出current_batch和准备好next_batch                
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                self.brio_logger.info(f"-------------------------Epoch: {epoch}/{num_train_epochs}, Step: {step}/{steps_in_epoch} in epoch iterator(total {len(epoch_iterator)})---------------------------")
                # brio_logger.info(f"Training dataset 'candidates_ids' has been sent here4, give you an example {inputs.data['candidates_ids'][1]}, fclfclfcl")

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    # tr_loss_step, ranking_loss_step = self.training_step(model, inputs) ### 这里才是真正的训练 返回的是不需要追踪构建计算图的loss, 在里面计算了lm loss和brio的loss
                    tr_loss_step = self.training_step(model, inputs) ### 这里才是真正的训练 返回的是不需要追踪构建计算图的loss, 在里面计算了lm loss和brio的loss biubiubiu
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    # if is_last_step_and_steps_less_than_grad_acc or (
                    #     version.parse(accelerate_version) <= version.parse("0.20.3")
                    # ):
                    #     self.accelerator.gradient_state._set_sync_gradients(True)
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            # self.optimizer.clip_master_grads(args.max_grad_norm)
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            # nn.utils.clip_grad_norm_(
                            #     amp.master_params(self.optimizer),
                            #     args.max_grad_norm,
                            # )
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            # self.accelerator.clip_grad_norm_(
                            #     model.parameters(),
                            #     args.max_grad_norm,
                            # )
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    # Optimizer step
                    self.optimizer.step()
                    self.brio_logger.info(f"yes, i have just optimized myself!")
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            # if is_torch_tpu_available():
            if is_torch_xla_available(): ### 35特有biubiubiu
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        # train_loss = self._total_loss_scalar / self.state.global_step
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:##
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # inputs.data["input_ids"].shape
        # torch.Size([3, 184])
        # inputs.data["attention_mask"].shape
        # torch.Size([3, 184])
        # inputs.data["labels"].shape
        # torch.Size([3, 184])
        # inputs.data["candidate_1_input_ids"].shape
        # torch.Size([3, 204])
        # inputs.data["candidate_2_input_ids"].shape
        # torch.Size([3, 212])
        # inputs.data["candidate_3_input_ids"].shape
        # torch.Size([3, 192])
        # inputs.data["candidate_4_input_ids"].shape
        # torch.Size([3, 204])
        # inputs.data["candidate_5_input_ids"].shape
        # torch.Size([3, 208])
        # inputs.data["candidate_6_input_ids"].shape
        # torch.Size([3, 188])

        # inputs["input_ids"].shape = torch.Size([3, 184])
        # inputs["attention_mask"].shape = torch.Size([3, 184])
        # inputs["labels"].shape = torch.Size([3, 184])
        # inputs["candidate_1_input_ids"].shape = torch.Size([3, 204])
        # inputs["candidate_1_attention_mask"].shape = torch.Size([3, 204])
        # inputs["candidate_1_labels"].shape = torch.Size([3, 204])
        # inputs["candidate_5_input_ids"].shape = torch.Size([3, 208])
        # inputs["candidate_5_attention_mask"].shape = torch.Size([3, 208])
        # inputs["candidate_5_labels"].shape = torch.Size([3, 208])

        ### biubiubiu one forward 将inputs 里的input_ids和candidates_*_拼接起来，并pad，并制作相关的attention_mask和labels
        merged_batchsize_inputs = self.merge_batchsize_input_ids_and_candidate_x_and_pad(inputs, self.num_candidates, self.input_ids_and_attention_mask_pad_id, self.labels_pad_id, self.candidate_labels_pad_id)

        # merged_batchsize_inputs["input_ids"].shape
        # torch.Size([21, 212])
        # merged_batchsize_inputs["attention_mask"].shape
        # torch.Size([21, 212])
        # merged_batchsize_inputs["labels"].shape
        # torch.Size([21, 212])

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            # loss = self.compute_loss(model, inputs) ### ***进入*** 需要改的！biubiubiu
            loss = self.compute_loss(model, merged_batchsize_inputs) ### ***进入*** 需要改的！biubiubiu

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
### biubiubiu
### 应该是在这里加上ranking_loss***重要***
### 开始用candidates做排序loss
        ### 注意，tensor inputs此时已经在gpu上了 # 查看candidates的shape,以及有没有做好padding(按batch里最大长度做padding)
        # candidates_scores = self.score_from_model(model, inputs.data) ### 已经mean了
        # self.brio_logger.info(f"model device: {model.device}, the size of candidates_scores tensor are: {candidates_scores.shape}, and the socres themselves are:\n{candidates_scores}, the device of the scores is: {candidates_scores.device}")
        # ranking_loss = self.compute_ranking_loss(candidates_scores)
        # self.brio_logger.info(f"The ranking loss device is: {ranking_loss.device}")
        # ranking_loss_sum = torch.sum(ranking_loss)
        # self.brio_logger.info(f"model device: {model.device}, cross_entropy_loss: {loss}, ranking_loss: {ranking_loss_sum}, c_e loss device: {loss.device}, r_k loss device: {ranking_loss_sum.device}") ### ranking_loss_sum = tensor(0.2904, device='cuda:0', grad_fn=<SumBackward0>)

        # ### 合并loss
        # loss += ranking_loss_sum ### biubiubiu 先看看原来的sft有没有问题，没问题
        # self.brio_logger.info(f"model device: {model.device}, total loss: {loss}, loss device: {loss.device}") ### loss = tensor(0.7922, device='cuda:0', grad_fn=<AddBackward0>)

        # if self.args.n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
        #     self.brio_logger.info(f"device_mean_loss: {loss}")
        #     ranking_loss_sum = ranking_loss_sum.mean() ### biubiubiu
        #     self.brio_logger.info(f"device_mean_ranking_loss: {ranking_loss_sum}")

        ### 不用判断apex，反正不用
        # if self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # # else:
        # #     self.accelerator.backward(loss) ### 这里用loss做了梯度回传
        # #     self.accelerator.backward(ranking_loss_sum) ### biubiubiu
        # # return loss.detach() / self.args.gradient_accumulation_steps, ranking_loss_sum.detach() / self.args.gradient_accumulation_steps
    
        # else:
        #     total_loss = loss + ranking_loss_sum
        #     self.accelerator.backward(total_loss)
        
        # total_loss = loss + ranking_loss_sum
        # self.accelerator.backward(total_loss)
        
        # return total_loss.detach() / self.args.gradient_accumulation_steps
### biubiubiu

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs: ### self.label_smoother = None
            labels = inputs.pop("labels")
        else:
            labels = None ### here
            
        ### biubiubiu
        self.brio_logger.info(f"shape of inputs is: input_ids: {inputs['input_ids'].shape}, attention_mask: {inputs['attention_mask'].shape}, labels: {inputs['labels'].shape}")
        self.brio_logger.info(f"size of inputs['input_ids'] is: {inputs['input_ids'].size()}, size of inputs['attention_mask'] is: {inputs['attention_mask'].size()}, size of inputs['labels'] is: {inputs['labels'].size()}")
        ### 送入模型的不要candidates ***很重要，因为llama2模型forward不接收candidates***
        # inputs_lm = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": inputs["labels"]} ### 其实这里能运行也是奇怪的，因为inputs的data里才是字典，但是我好像可以直接调用字典跳过data
        # outputs = model(**inputs_lm)## ***进入***
        # outputs = model(**{
        # "input_ids": inputs["input_ids"],
        # "attention_mask": inputs["attention_mask"],
        # "labels": inputs["labels"]})
        outputs = model(**inputs)

        self.brio_logger.info(f"shape of outputs.logits is: {outputs.logits.shape}")
        # inputs['input_ids'].size()=torch.Size([3, 128])
        # inputs['attention_mask'].size()=torch.Size([3, 128])
        # inputs['labels'].size()=torch.Size([3, 128])
        # inputs['candidates_ids'].size()=torch.Size([3, 796]) ### 历史洪流了属于是，这个拼到一起再解析，不合理
        ### biubibiu

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # if is_peft_available() and isinstance(model, PeftModel):
            #     model_name = unwrap_model(model.base_model)._get_name()
            # else:
            #     model_name = unwrap_model(model)._get_name()
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] ### here， 前面的一个也没走 tensor(0.5018, device='cuda:0', grad_fn=<NllLossBackward0>)
        return (loss, outputs) if return_outputs else loss
    
    # def pad_candidate_x_input_ids_2_the_same_length(self, **inputs):
    #     # self.brio_logger.info(f"in the pad_function the passed dataset is like: {inputs}")
    #     self.brio_logger.info(f"in the pad_function the passed dataset is like: {[inputs[f'candidate_{i+1}_input_ids'].shape for i in range(self.num_candidates)]}")
    #     max_length = max(inputs[f"candidate_{i+1}_input_ids"].shape[1] for i in range(self.num_candidates))
    #     for i in range(self.num_candidates):
    #         input_key = f"candidate_{i+1}_input_ids"
    #         current_length = inputs[input_key].shape[1]
    #         if current_length < max_length:
    #             padding = torch.zeros((inputs[input_key].shape[0], max_length - current_length), dtype=torch.long, device=inputs[input_key].device)
    #             inputs[input_key] = torch.cat([inputs[input_key], padding], dim=1)
    #     self.brio_logger.info(f"After padding, the shapes are: {[inputs[f'candidate_{i+1}_input_ids'].shape for i in range(self.num_candidates)]}")

# ### 再定义一个函数计算模型的得分
#     def score_from_model(self, model, inputs):
#         # debug_target_tokens_list = []###*
#         score_list = []
#         ### pad to the same length
#         # self.brio_logger.info(f"the passed dataset is like: {inputs}")
#         self.pad_candidate_x_input_ids_2_the_same_length(
#             **{f"candidate_{i+1}_input_ids": inputs[f"candidate_{i+1}_input_ids"] for i in range(self.num_candidates)}
#             )
    def merge_batchsize_input_ids_and_candidate_x_and_pad(self, inputs, num_candidates, input_ids_and_attention_mask_pad_id, labels_pad_id, candidate_labels_pad_id):
        # Find the longest length among inputs["input_ids"] and all candidate_x_input_ids
        max_input_len = inputs["input_ids"].size(1)
        max_candidate_len = max(inputs[f"candidate_{i}_input_ids"].size(1) for i in range(1, num_candidates + 1))
        MAX_LEN = max(max_input_len, max_candidate_len)

        # Pad inputs["input_ids"] and inputs["attention_mask"] to MAX_LEN
        padded_input_ids = F.pad(inputs["input_ids"], (0, MAX_LEN - inputs["input_ids"].size(1)), value=input_ids_and_attention_mask_pad_id) ### torch.Size([3, 212])
        padded_attention_mask = F.pad(inputs["attention_mask"], (0, MAX_LEN - inputs["attention_mask"].size(1)), value=input_ids_and_attention_mask_pad_id)
        # Pad inputs["labels"] to MAX_LEN
        padded_labels = F.pad(inputs["labels"], (0, MAX_LEN - inputs["labels"].size(1)), value=labels_pad_id)

        # padded_input_ids
        # tensor([[128000,     32,   6369,   1990,    264,  22999,   1217,    323,    459,
        #           21075,  11478,  18328,     13,    578,  18328,   6835,  11190,     11,
        #           11944,     11,    323,  48887,  11503,    311,    279,   1217,    596,
        #            4860,  58682,    264,  43030,    449,   2218,   6012,   4005,     82,
        #              29,   6584,     25,  20400,    264,  43030,    449,    279,   2768,
        #            6012,     25,    264,  32541,  22018,  73815,    315,  26166,  76646,
        #            2779,    434,   5573,    315,    220,     15,     13,   5313,     11,
        #             264,  52132,   2168,    367,  19863,   5573,    315,    482,     24,
        #              13,   1313,     11,    323,    264,  75928,  81372,  18607,    315,
        #             220,     18,     13,   1806,     13,  30379,    279,  43030,    374,
        #           44910,    304,   3254,  55331,  15785,     13,  36660,   3931,   2891,
        #              25,  88766,    284,     34,    356,    356,    356,    284,     34,
        #             356,  26176,     17,  22249,     16,    507,    328,    284,  18697,
        #              16,    356,    284,     46,    284,  18697,     16,    356,    284,
        #              46,    452,    356,     31,     39,     16,    356,    356,  19741,
        #              39,     16,    356,    356,    356,    356,    284,     34,    356,
        #             284,     34,    356,  22249,     16,    507,    284,     34,  22249,
        #              16,    284,  18697,     16,  22249,     16,    674,  18697,     17,
        #             284,     34,    356,    284,     34,  22249,     17,  22249,     16,
        #             284,  18697,     16,    452,  22249,     17,  22249,     16,    674,
        #           18697,     17,      6, 128001,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0],
        #         [128000,     32,   6369,   1990,    264,  22999,   1217,    323,    459,
        #           21075,  11478,  18328,     13,    578,  18328,   6835,  11190,     11,
        #           11944,     11,    323,  48887,  11503,    311,    279,   1217,    596,
        #            4860,  58682,    264,  43030,    449,   2218,   6012,   4005,     82,
        #              29,   6584,     25,  20400,    264,  43030,    449,    279,   2768,
        #            6012,     25,    264,  32541,  22018,  73815,    315,  26166,  76646,
        #            2779,    434,   5573,    315,    220,     15,     13,   3226,     11,
        #             264,  52132,   2168,    367,  19863,   5573,    315,    482,     24,
        #              13,   1114,     11,    323,    264,  75928,  81372,  18607,    315,
        #             220,     17,     13,   2495,     13,  30379,    279,  43030,    374,
        #           44910,    304,   3254,  55331,  15785,     13,  36660,   3931,   2891,
        #              25,  88766,    284,     34,  26176,     17,  22249,     16,    284,
        #              34,    452,    356,    356,     31,     39,     16,    356,    356,
        #             356,    356,    452,  22249,     16,    284,  18697,     16,    356,
        #             284,     45,    356,    284,     45,    356,    284,     34,  22249,
        #              16,    284,  18697,     16,    507,    356,    284,     34,    356,
        #             284,     34,    356,    284,     34,  22249,     16,    284,  18697,
        #              16,  22249,     16,    284,  18697,     17,    356,    284,     34,
        #             356,    284,     34,    356,    284,     34,  22249,     16,    284,
        #           18697,     16,      6, 128001, 128001, 128001, 128001, 128001, 128001,
        #          128001, 128001, 128001, 128001,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0],
        #         [128000,     32,   6369,   1990,    264,  22999,   1217,    323,    459,
        #           21075,  11478,  18328,     13,    578,  18328,   6835,  11190,     11,
        #           11944,     11,    323,  48887,  11503,    311,    279,   1217,    596,
        #            4860,  58682,    264,  43030,    449,   2218,   6012,   4005,     82,
        #              29,   6584,     25,  20400,    264,  43030,    449,    279,   2768,
        #            6012,     25,    264,  32541,  22018,  73815,    315,  26166,  76646,
        #            2779,    434,   5573,    315,    220,     15,     13,   3487,     11,
        #             264,  52132,   2168,    367,  19863,   5573,    315,    482,     24,
        #              13,   2437,     11,    323,    264,  75928,  81372,  18607,    315,
        #             220,     16,     13,   4161,     13,  30379,    279,  43030,    374,
        #           44910,    304,   3254,  55331,  15785,     13,  36660,   3931,   2891,
        #              25,      6,     45,    356,    284,  18697,     16,    356,    284,
        #              46,    356,    284,     34,    356,    284,     34,    356,  26176,
        #              17,  22249,     16,    393,    356,    452,  26176,     16,    674,
        #           18697,     17,    356,    356,    284,     34,    356,    284,     34,
        #             356,    284,     34,  22249,     16,    284,  18697,     16,    356,
        #             284,  18697,     16,    356,    284,     46,    452,    356,    284,
        #              34,    356,    284,     34,  26176,     16,    356,   2493,    356,
        #           26176,     16,    356,   2493,    284,     34,  22249,     16,  26176,
        #              17,    284,     34,  22249,     17,  22249,     16,    674,  18697,
        #              17,      6, 128001, 128001,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0,      0,      0,      0]], device='cuda:0')

        # Pad and concatenate candidate_x_input_ids, candidate_x_attention_mask, candidate_x_labels
        concatenated_candidate_input_ids = []
        concatenated_candidate_attention_mask = []
        concatenated_candidate_labels = []

        for i in range(1, num_candidates + 1):
            candidate_input_ids = F.pad(inputs[f"candidate_{i}_input_ids"], (0, MAX_LEN - inputs[f"candidate_{i}_input_ids"].size(1)), value=input_ids_and_attention_mask_pad_id)
            candidate_attention_mask = F.pad(inputs[f"candidate_{i}_attention_mask"], (0, MAX_LEN - inputs[f"candidate_{i}_attention_mask"].size(1)), value=input_ids_and_attention_mask_pad_id)
            candidate_labels = F.pad(inputs[f"candidate_{i}_labels"], (0, MAX_LEN - inputs[f"candidate_{i}_labels"].size(1)), value=candidate_labels_pad_id)

            concatenated_candidate_input_ids.append(candidate_input_ids)
            concatenated_candidate_attention_mask.append(candidate_attention_mask)
            concatenated_candidate_labels.append(candidate_labels)

        # concatenated_candidate_input_ids = torch.stack(concatenated_candidate_input_ids, dim=0) ### torch.Size([6, 3, 212])
        # concatenated_candidate_attention_mask = torch.stack(concatenated_candidate_attention_mask, dim=0)
        # concatenated_candidate_labels = torch.stack(concatenated_candidate_labels, dim=0)

        # concatenated_candidate_input_ids
        # tensor([[[128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0]],

        #         [[128000,     32,   6369,  ...,      6, 128001,      0], ### 得是4的倍数 __call__
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0]],

        #         [[128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0]],

        #         [[128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0]],

        #         [[128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0]],

        #         [[128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0]]],
        #     device='cuda:0')

        concatenated_candidate_input_ids = torch.cat(concatenated_candidate_input_ids, dim=0)  # 拼接而不是堆叠 torch.Size([18, 212])
        concatenated_candidate_attention_mask = torch.cat(concatenated_candidate_attention_mask, dim=0)
        concatenated_candidate_labels = torch.cat(concatenated_candidate_labels, dim=0)

        # concatenated_candidate_input_ids
        # tensor([[128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        
        #         [128000,     32,   6369,  ...,      6, 128001,      0],
        #         ...,
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0],
        #         [128000,     32,   6369,  ...,      0,      0,      0]],
        #     device='cuda:0')
        # concatenated_candidate_input_ids.shape
        # torch.Size([18, 212])

        # Concatenate inputs["input_ids"] and candidate_x_input_ids
        inputs_new = {
            # "input_ids": torch.cat([padded_input_ids.unsqueeze(0), concatenated_candidate_input_ids], dim=0),
            # "attention_mask": torch.cat([padded_attention_mask.unsqueeze(0), concatenated_candidate_attention_mask], dim=0),
            # "labels": torch.cat([padded_labels.unsqueeze(0), concatenated_candidate_labels], dim=0)
            "input_ids": torch.cat([padded_input_ids, concatenated_candidate_input_ids], dim=0),
            "attention_mask": torch.cat([padded_attention_mask, concatenated_candidate_attention_mask], dim=0),
            "labels": torch.cat([padded_labels, concatenated_candidate_labels], dim=0)
        }
        ### TODO: 可以删除原inputs吗？如果在cuda上的话！
        # # 删除原来的数据
        # padded_input_ids = None
        # concatenated_candidate_input_ids = None
        # padded_attention_mask = None
        # concatenated_candidate_attention_mask = None
        # padded_labels = None
        # concatenated_candidate_labels = None
        # # 清空缓存
        # torch.cuda.empty_cache()

        return inputs_new

    def pad_candidate_x_input_ids_2_the_same_length(self, inputs): # inputs是真正的完整输入数据的inputs.data
        self.brio_logger.info(f"in the pad_function the passed dataset is like: {[inputs[f'candidate_{i+1}_input_ids'].shape for i in range(self.num_candidates)]}")
        max_length = max(inputs[f"candidate_{i+1}_input_ids"].shape[1] for i in range(self.num_candidates))
        for i in range(self.num_candidates):
            input_key = f"candidate_{i+1}_input_ids"
            current_length = inputs[input_key].shape[1]
            if current_length < max_length:
                padding = torch.zeros((inputs[input_key].shape[0], max_length - current_length), dtype=torch.long, device=inputs[input_key].device)
                inputs[input_key] = torch.cat([inputs[input_key], padding], dim=1)
        self.brio_logger.info(f"After padding, the shapes are: {[inputs[f'candidate_{i+1}_input_ids'].shape for i in range(self.num_candidates)]}")

    # def extract_next_token_probs(self, log_probs, input_ids):
    #     """
    #     Extract the log probabilities of the next token in the sequence for each candidate.
        
    #     Args:
    #     log_probs (torch.Tensor): The log probabilities of shape (batch_size, sequence_length, vocab_size).
    #     input_ids (torch.Tensor): The input ids of shape (batch_size, sequence_length).

    #     Returns:
    #     torch.Tensor: The log probabilities of the next token in the sequence for each candidate.
    #     log_probs[:, :-1, :] 去除了最后一个时间步的对数概率，因为没有对应的下一个 token。
    #     input_ids[:, 1:] 去除了第一个时间步的输入，因为第一个时间步没有上一个 token。
    #     gather(2, next_input_ids.unsqueeze(-1)) 使用 next_input_ids 提取相应位置的对数概率。
    #     squeeze(-1) 移除最后一个维度，得到形状为 (batch_size, sequence_length - 1) 的张量。
    #     """
    #     # 移除最后一个时间步的log_probs，并与下一个时间步的input_ids对齐
    #     log_probs = log_probs[:, :-1, :]  # shape: (batch_size, sequence_length - 1, vocab_size)
    #     next_input_ids = input_ids[:, 1:]  # shape: (batch_size, sequence_length - 1)

    #     # 使用 gather 提取对应的对数概率
    #     next_token_log_probs = log_probs.gather(2, next_input_ids.unsqueeze(-1)).squeeze(-1)
        
    #     return next_token_log_probs

    def extract_next_token_probs(self, log_probs, input_ids):
        """
        Extract the log probabilities of the next token in the sequence for each candidate.
        
        Args:
        log_probs (torch.Tensor): The log probabilities of shape (batch_size, sequence_length, vocab_size).
        input_ids (torch.Tensor): The input ids of shape (batch_size, sequence_length).

        Returns:
        torch.Tensor: The log probabilities of the next token in the sequence for each candidate.
        """
        batch_size, seq_length, vocab_size = log_probs.shape
        
        # 去除最后一个时间步的 log_probs，并与下一个时间步的 input_ids 对齐
        log_probs = log_probs[:, :-1, :]  # shape: (batch_size, sequence_length - 1, vocab_size)
        next_input_ids = input_ids[:, 1:]  # shape: (batch_size, sequence_length - 1) ### 其实对于自回归，input_ids就是答案，就是labels
        
        # 创建掩码，遇到 0 时停止
        mask = (next_input_ids != 0)
        
        # 使用 gather 提取对应的对数概率
        next_token_log_probs = log_probs.gather(2, next_input_ids.unsqueeze(-1)).squeeze(-1)  # shape: (batch_size, sequence_length - 1)
        
        # 将掩码应用到对数概率
        next_token_log_probs = next_token_log_probs * mask.float()
        
        return next_token_log_probs

    def score_from_model(self, model, inputs):
        self.pad_candidate_x_input_ids_2_the_same_length(inputs)
        self.brio_logger.info(f"The padding processed inputs are:\n {[inputs[f'candidate_{i+1}_input_ids'] for i in range(self.num_candidates)]}")
        # Concatenate candidate input ids
        all_candidates_input_ids = torch.cat([inputs[f"candidate_{i+1}_input_ids"] for i in range(self.num_candidates)], dim=0) ### torch.Size([18, 136])
        self.brio_logger.info(f"The concated inputs are:\n{all_candidates_input_ids}")
        self.brio_logger.info(f"The shape of the concated inputs are: {all_candidates_input_ids.shape}") ### torch.Size([18, 136])
        # Forward pass through the model
        candidate_x_input_ids_output = model(all_candidates_input_ids) ### biubiubiu 这里面有没有loss呢？
        ### biubiubiu candidate_x_input_ids_output[0]是什么？CausalLMOutputWithPast(loss={'logits': tensor([[[-12.5625,  -7.0312,  -0.3086,  ...,  -6.8438,  -8.2500,  -7.4375], device='cuda:0', grad_fn=<ToCopyBackward0>)}, logits=tensor([[[-12.5625,  -7.0312,  -0.3086,  ...,  -6.8438,  -8.2500,  -7.4375],device='cuda:0', grad_fn=<ToCopyBackward0>), past_key_values=None, hidden_states=None, attentions=None)
        # candidate_x_input_ids_output = model(**{f"candidate_{i+1}_input_ids": inputs[f"candidate_{i+1}_input_ids"] for i in range(self.num_candidates)}) ### 这是错误做法
        candidate_x_input_ids_output_logits = candidate_x_input_ids_output.logits
        self.brio_logger.info(f"The candidate_x_input_ids_output_logits is:\n{candidate_x_input_ids_output_logits}")
        self.brio_logger.info(f"The shape of the model output is: {candidate_x_input_ids_output_logits.shape}") ### torch.Size([18, 136, 32000])
        candidate_x_input_ids_output_logits_log_probs = F.log_softmax(candidate_x_input_ids_output_logits, dim=-1)
        self.brio_logger.info(f"The log probabilities are:\n{candidate_x_input_ids_output_logits_log_probs}")
        self.brio_logger.info(f"The shape of the log probabilities is: {candidate_x_input_ids_output_logits_log_probs.shape}") ### torch.Size([18, 136, 32000])
        ### biubiubiu 不用提取condition部分，然后再从后面开始计算概率，因为对于同一批待排序的句子组，condition都是一样的，因此，相对顺序不受概率绝对值影响。
        ### 下面计算下一个token的log probability
        next_token_log_probs = self.extract_next_token_probs(candidate_x_input_ids_output_logits_log_probs, all_candidates_input_ids) ###  torch.Size([18, 135])
        self.brio_logger.info(f"The shape of the next token log probabilities is: {next_token_log_probs.shape}")
        mean_next_token_log_probs = next_token_log_probs.mean(dim=1) ### torch.Size([18])
        self.brio_logger.info(f"The mean scores are:\n{mean_next_token_log_probs}")
        self.brio_logger.info(f"The shape of the mean of the next token log probabilities is: {mean_next_token_log_probs.shape}")
        
        ### 更改一下顺序biubiubiu
        reshaped_scores = self.reshape_tensor(mean_next_token_log_probs, self._train_batch_size, self.num_candidates).to(model.device)
        
        return reshaped_scores

    # def reshape_tensor(self, tensor, train_batch_size, num_candidates):
    #     # 确保tensor的长度符合要求
    #     assert tensor.shape[0] == train_batch_size * num_candidates, "Tensor length does not match the expected shape."

    #     # 初始化一个空的张量来存储结果
    #     reshaped_tensor = torch.empty(train_batch_size, num_candidates, dtype=tensor.dtype)
    #     self.brio_logger.info(f"The reshaped_tensor is: {reshaped_tensor}")

    #     for i in range(train_batch_size):
    #         # 按照间隔提取元素
    #         reshaped_tensor[i] = tensor[i::train_batch_size]
    #     assert reshaped_tensor.shape[0] == train_batch_size, "rshaped Tensor wrong."
    #     assert reshaped_tensor.shape[1] == num_candidates, "reshaped Tensor wrong again."
    #     self.brio_logger.info(f"The filled reshaped_tensor is: {reshaped_tensor}")

    #     return reshaped_tensor
### biubiubiu 适应不满batchsize的情况
    def reshape_tensor(self, tensor, train_batch_size, num_candidates):
        # 获取tensor的实际长度
        actual_length = tensor.shape[0]

        # 计算实际的batch数量
        actual_batchsize = actual_length // num_candidates
        # remaining_elements = actual_length % num_candidates
        # 如果剩余元素大于0，则需要额外处理一个batch
        # total_batchsize = actual_batchsize + (1 if remaining_elements > 0 else 0)

        # 初始化一个空的张量来存储结果
        reshaped_tensor = torch.empty(actual_batchsize, num_candidates, dtype=tensor.dtype)
        if actual_batchsize == train_batch_size:
            self.brio_logger.info(f"The empty reshaped_tensor is:\n{reshaped_tensor}")
        else:
            self.brio_logger.info(f"This batchsize is smaller, there are only {actual_batchsize} here!")

        for i in range(actual_batchsize):
            reshaped_tensor[i] = tensor[i::actual_batchsize]
            # if i < actual_batchsize:
            #     reshaped_tensor[i] = tensor[i * num_candidates: (i + 1) * num_candidates]
            # else:
            #     self.brio_logger.info(f"ASSERT i will never come here!")
            #     reshaped_tensor[i, :remaining_elements] = tensor[i * num_candidates:]
            #     reshaped_tensor[i, remaining_elements:] = torch.tensor([float('nan')] * (num_candidates - remaining_elements), dtype=tensor.dtype)

        assert reshaped_tensor.shape[0] == actual_batchsize, "reshaped Tensor wrong."
        assert reshaped_tensor.shape[1] == num_candidates, "reshaped Tensor wrong again."
        self.brio_logger.info(f"The filled reshaped_tensor is:\n{reshaped_tensor}")

        return reshaped_tensor

    def compute_ranking_loss(self, reshaped_scores):
        # train_batch_size = self._train_batch_size
        # number_of_candidates = self.num_candidates
        # reshaped_scores = self.reshape_tensor(scores, self._train_batch_size, self.num_candidates).to(scores.device)
        # reshaped_scores = self.reshape_tensor(scores, self._train_batch_size, self.num_candidates).to(model.device) ### 没有model，不应该在这里reshape
        self.brio_logger.info(f"------------------------------------- computing ranking loss now ---------------------------------------------")
        # self.brio_logger.info(f"The model device is: {model.device}") ### 想啥呢，没有model
        self.brio_logger.info(f"When computing ranking loss, the device is: {reshaped_scores.device}")
        self.brio_logger.info(f"The reshaped scores are:\n{reshaped_scores}") ### 到这一步，都是多device都有的
        self.brio_logger.info(f"The shape of the reshaped socres are: {reshaped_scores.shape}")

        # ranking_loss = nn.MarginRankingLoss()
        ranking_loss = nn.MarginRankingLoss(size_average=False, reduce=False, reduction='none')
        self.brio_logger.info(f"The original ranking losses obj is:{ranking_loss}")
        
        # 利用广播技术
        # 原张量是一维时
        # x1 = reshaped_scores.unsqueeze(1)  # 转换为列向量
        # self.brio_logger.info(f"The liexiangliang of reshaped scores are: {x1}")
        # x2 = reshaped_scores.unsqueeze(0)  # 转换为行向量
        # self.brio_logger.info(f"The hangxiangliang of reshaped scores are: {x2}")
        # 原张量是二维时
        x1 = reshaped_scores.unsqueeze(2)  # 在第二维度上增加一个维度，变成列向量
        self.brio_logger.info(f"The liexiangliang of reshaped scores are:\n{x1}, device is: {x1.device}")
        x2 = reshaped_scores.unsqueeze(1)  # 在第一维度上增加一个维度，变成行向量
        self.brio_logger.info(f"The hangxiangliang of reshaped scores are:\n{x2}, device is: {x2.device}")

        # 创建标签张量，y 的值为 1，因为我们希望 x1 > x2
        y = torch.ones_like(x1 - x2)
        # 这样，通过广播，x1 - x2 的操作会自动扩展到 (L, L) 的形状，而 y 作为标签张量，每个元素对应着 x1 和 x2 对的比较关系。
        self.brio_logger.info(f"The broadcast y is:\n{y}, device is: {y.device}")

        loss_matrix = ranking_loss(x1, x2, y)
        self.brio_logger.info(f"The ranking loss of all batches are:\n{loss_matrix}, device is: {loss_matrix.device}")

        # 创建掩码 mask，上三角矩阵
        mask_scores = torch.triu(torch.ones_like(y), diagonal=1)
        self.brio_logger.info(f"The mask is:\n{mask_scores}, device is: {mask_scores.device}")

        # 过滤结果，x1的index小于 x2的index
        filtered_loss = loss_matrix * mask_scores
        self.brio_logger.info(f"The filtered ranking loss of all batches are:\n{filtered_loss}, device is: {filtered_loss.device}")

        return filtered_loss

def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

# Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def llama_torch_attn_forward(
        self: "LlamaAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    past_key_value = getattr(self, "past_key_value", past_key_value)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
        assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
        num_groups = q_len // groupsz

        def shift(state: torch.Tensor) -> torch.Tensor:
            state = state.transpose(1, 2)  # output: (bsz, seq_len, n_heads, head_dim)
            state = torch.cat(
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)  # (bsz, :, seq_len, :) or (bsz*n_group, :, groupsz, :)
    attn_output = attn_output.transpose(1, 2).contiguous()

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
        attn_output = torch.cat(
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1),
            )
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def llama_flash_attn_forward(
        self: "LlamaFlashAttention2",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
    key_states = key_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
    value_states = value_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)

    dropout_rate = self.attention_dropout if self.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning("The input hidden states seems to be silently casted in float32.")
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
        assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
        num_groups = q_len // groupsz

        def shift(state: torch.Tensor) -> torch.Tensor:
            state = torch.cat(
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

    attn_output: torch.Tensor = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
        attn_output = torch.cat(
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1),
            )
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def apply_llama_patch() -> None:
    LlamaAttention.forward = llama_torch_attn_forward
    LlamaFlashAttention2.forward = llama_flash_attn_forward

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments, CustomArguments)) ###
    # parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args, custom_args= parser.parse_args_into_dataclasses() ###
    # model_args, data_args, training_args, script_args= parser.parse_args_into_dataclasses()

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")
    ### biubiubiu
    logger.info(f"BRIO log path: {custom_args.brio_log_file_path}")
    logger.info(f"BRIO candidate labels pad id: {custom_args.brio_candidate_labels_pad_id}")
    ### biubiubiu one forward
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    # prompt_template = get_conv_template(data_args.template_name)
    prompt_template = get_conv_template(script_args.template_name)
    if tokenizer.eos_token_id is None:
        # tokenizer.eos_token = prompt_template.stop_str  # eos token is required for SFT
        # logger.info("Add eos token: {}".format(tokenizer.eos_token))
        tokenizer.eos_token = prompt_template.stop_str  # eos token is required
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        # logger.info("Add pad token: {}".format(tokenizer.pad_token))
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")
    ### IGNORE_INDEX用于teacher forcing任务的label smoothing
    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    IGNORE_INDEX_For_candidates = custom_args.brio_candidate_labels_pad_id ### biubiubiu one forward
    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        # raw_datasets = load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     cache_dir=model_args.cache_dir,
        # )
        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["validation"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[:{data_args.validation_split_percentage}%]",
        #         cache_dir=model_args.cache_dir,
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[{data_args.validation_split_percentage}%:]",
        #         cache_dir=model_args.cache_dir,
        #     )
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            # Split the shuffled train dataset into training and validation sets
            split = shuffled_train_dataset.train_test_split(
                test_size=data_args.validation_split_percentage / 100,
                seed=42
            )
            # Assign the split datasets back to raw_datasets
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        # Loading a dataset from local files.
        logger.info("We are Loading finetune dataset from a local files, fclfclfcl")
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            # raw_datasets["validation"] = load_dataset(
            #     'json',
            #     data_files=data_files,
            #     split=f"train[:{data_args.validation_split_percentage}%]",
            #     cache_dir=model_args.cache_dir,
            # )
            # raw_datasets["train"] = load_dataset(
            #     'json',
            #     data_files=data_files,
            #     split=f"train[{data_args.validation_split_percentage}%:]",
            #     cache_dir=model_args.cache_dir,
            # )
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train_dataset.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    logger.info(f"Raw datasets: {raw_datasets}") ### 字典

    # Preprocessing the datasets
    max_length = script_args.model_max_length

    def preprocess_function_train(examples): # 这里的examples是全部数据集
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        ###biubiubiu
        candidate_source_ids_lists = [] ### 其实就是下面的source_ids的一个合集
        ###biubiubiu
        roles = ["human", "gpt"]
### 处理数据集时不会涉及到batch，且是一条一条处理的（可以多线程同步处理）
        def get_dialog(examples):
            for i, source in enumerate(examples['conversations']):
                # logger.debug(f"we are processing the {i}-th example, fclfclfcl")
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])  ### 分别添加human和gpt的内容
                if len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                yield prompt_template.get_dialog(history_messages)  ### [["human","gpt"]]

        for dialog in get_dialog(examples): ## 99个examples
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0)) # 其实是为了多轮问答准备的
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                # input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn ### 最后加上了结束符
                # labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id] ### 输入prompt部分变成-100
                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                if script_args.train_on_inputs: ### 这个意思就是inputs也计算到loss里，inputs也纳入训练内容里
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)
            ### biubiubiu
            candidate_source_ids_lists.append(source_ids)
            ### biubiubiu

        # # 初始化独立的候选人输入id列表
        # num_candidates = len(examples['conversations'][0][0]['candidates_sorted'])
        # # 动态创建独立的候选人输入id列表
        # for j in range(num_candidates):
        #     exec(f'candidate_{j+1}_input_ids_list = []')
        
        # # 遍历每个对话
        # for i, source in enumerate(examples['conversations']):  # len(examples['conversations']) = 99
        #     # 遍历每个候选人
        #     for j, candidate in enumerate(source[0]['candidates_sorted']):  # len(source[0]['candidates_sorted']) = 6
        #         # 编码候选人文本
        #         candidate_ids = tokenizer.encode(text=candidate, add_special_tokens=False)
        #         candidate_ids = candidate_source_ids_lists[i] + candidate_ids + [tokenizer.eos_token_id]
        #         # 将编码后的候选人id加入相应的列表
        #         exec(f'candidate_{j+1}_input_ids_list.append(candidate_ids)')

        # return dict(    ### 最终得到的train_dataset的内容
        #     input_ids=input_ids_list,
        #     attention_mask=attention_mask_list,
        #     labels=targets_list,
        #     candidate_1_input_ids_list=candidate_1_input_ids_list,
        #     candidate_2_input_ids_list=candidate_2_input_ids_list,
        #     candidate_3_input_ids_list=candidate_3_input_ids_list,
        #     candidate_4_input_ids_list=candidate_4_input_ids_list,
        #     candidate_5_input_ids_list=candidate_5_input_ids_list,
        #     candidate_6_input_ids_list=candidate_6_input_ids_list
        # )

        # # 初始化独立的候选人输入id列表
        # num_candidates = len(examples['conversations'][0][0]['candidates_sorted'])
        # candidate_input_ids_lists = [[] for _ in range(num_candidates)]

        # # 遍历每个对话
        # for i, source in enumerate(examples['conversations']):  # len(examples['conversations']) = 99
        #     # 遍历每个候选人
        #     for j, candidate in enumerate(source[0]['candidates_sorted']):  # len(source[0]['candidates_sorted']) = 6
        #         # 编码候选人文本
        #         candidate_ids = tokenizer.encode(text=candidate, add_special_tokens=False)
        #         candidate_ids = candidate_source_ids_lists[i] + candidate_ids + [tokenizer.eos_token_id]
        #         # 将编码后的候选人id加入相应的列表
        #         candidate_input_ids_lists[j].append(candidate_ids)

        # return dict(    ### 最终得到的train_dataset的内容
        #     input_ids=input_ids_list,
        #     attention_mask=attention_mask_list,
        #     labels=targets_list,
        #     candidate_input_ids_lists=candidate_input_ids_lists
        # )
        ###biubiubiu###
        # 初始化独立的候选人输入id列表
        num_candidates = len(examples['conversations'][0][0]['candidates_sorted'])
        candidate_input_ids_lists = [[] for _ in range(num_candidates)]
        candidate_attention_mask_lists = [[] for _ in range(num_candidates)]
        candidate_labels_lists = [[] for _ in range(num_candidates)]

        # 遍历每个对话
        for i, source in enumerate(examples['conversations']):  # len(examples['conversations']) = 99
            # 遍历每个候选人
            for j, candidate in enumerate(source[0]['candidates_sorted']):  # len(source[0]['candidates_sorted']) = 6
                # 编码候选人文本
                candidate_target_ids = tokenizer.encode(text=candidate, add_special_tokens=False)
                candidate_input_ids = candidate_source_ids_lists[i] + candidate_target_ids + [tokenizer.eos_token_id]
                candidate_labels = [IGNORE_INDEX_For_candidates] * len(candidate_source_ids_lists[i]) + candidate_target_ids + [tokenizer.eos_token_id]
                # 将编码后的候选人id加入相应的列表
                candidate_input_ids_lists[j].append(candidate_input_ids)
                candidate_attention_mask_lists[j].append([1] * len(candidate_input_ids))
                candidate_labels_lists[j].append(candidate_labels)

        # 将嵌套列表拆分为独立的列表
        result = {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': targets_list
        }

        for j in range(num_candidates):
            result[f'candidate_{j+1}_input_ids'] = candidate_input_ids_lists[j]
            result[f'candidate_{j+1}_attention_mask'] = candidate_attention_mask_lists[j]
            result[f'candidate_{j+1}_labels'] = candidate_labels_lists[j]

        return result
        ###biubiubiu###

#     def preprocess_function_valid(examples):
#         """
#         Preprocessing the datasets.
#             part of code modified from https://github.com/lm-sys/FastChat
#         """
#         input_ids_list = []
#         attention_mask_list = []
#         targets_list = []
#         roles = ["human", "gpt"]    
# ### 处理数据集时不会涉及到batch，且是一条一条处理的（可以多线程同步处理）
#         def get_dialog(examples):
#             for i, source in enumerate(examples['conversations']):
#                 logger.debug(f"we are processing the {i}-th example, fclfclfcl")
#                 if len(source) < 2:
#                     continue
#                 data_role = source[0].get("from", "")
#                 if data_role not in roles or data_role != roles[0]:
#                     # Skip the first one if it is not from human
#                     source = source[1:]
#                 if len(source) < 2:
#                     continue
#                 messages = []
#                 for j, sentence in enumerate(source):
#                     data_role = sentence.get("from", "")
#                     if data_role not in roles:
#                         logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
#                         break
#                     if data_role == roles[j % 2]:
#                         messages.append(sentence["value"])  ### 分别添加human和gpt的内容
#                 if len(messages) % 2 != 0:
#                     continue
#                 # Convert the list to pairs of elements
#                 history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
#                 yield prompt_template.get_dialog(history_messages)  ### [["human","gpt"]]

#         for dialog in get_dialog(examples):
#             input_ids, labels = [], []

#             for i in range(len(dialog) // 2):
#                 source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
#                 target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

#                 total_len = len(source_ids) + len(target_ids)
#                 max_source_len = int(max_length * (len(source_ids) / total_len))
#                 max_target_len = int(max_length * (len(target_ids) / total_len))

#                 if len(source_ids) > max_source_len:
#                     source_ids = source_ids[:max_source_len]
#                 if len(target_ids) > max_target_len - 1:  # eos token
#                     target_ids = target_ids[:max_target_len - 1]
#                 if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
#                     source_ids = source_ids[1:]
#                 if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
#                     target_ids = target_ids[:-1]
#                 if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
#                     break

#                 input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn ### 最后加上了结束符
#                 labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id] ### 输入prompt部分变成-100

#             input_ids_list.append(input_ids)
#             attention_mask_list.append([1] * len(input_ids))
#             targets_list.append(labels)

#         return dict(    ### 最终得到的train_dataset的内容
#             input_ids=input_ids_list,
#             attention_mask=attention_mask_list,
#             labels=targets_list,
#         )

### eval set 用的
    def preprocess_function(examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            for i, source in enumerate(examples['conversations']):
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                yield prompt_template.get_dialog(history_messages)

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                if script_args.train_on_inputs:
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )



    def filter_empty_labels(example):
        """Remove empty labels dataset."""
        return not all(label == IGNORE_INDEX for label in example["labels"])

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        # train_dataset = raw_datasets['train']
        train_dataset = raw_datasets['train'].shuffle(seed=42) ### 35 update biubiubiu
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        with training_args.main_process_first(desc="Train dataset tokenization"):
            train_dataset = train_dataset.shuffle().map(    ### 在这里处理了数据集
                preprocess_function_train,### 这里必须要进入看看的
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            train_dataset = train_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers) ## 过滤掉 label 全是-100的example
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
            # logger.debug(f"Decode labels[0]: {tokenizer.decode(train_dataset[0]['labels'])}, fclfclfcl") ### 这里肯定错了，不能这么写,无法解码,为什么呢，因为IGNORE_INDEX是-100，不在词表列表里
                        # raise IndexError('piece id is out of range.')
                        # IndexError: piece id is out of range.
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id  # 拿shuffle后的trainingset数据的第零个，做示例打印出来，将labels中不等于-100的部分保留下来，然后，将=-100的部分，转换成pad_token_id，也就是unk_token_id，也就是0
                               for label in list(train_dataset[0]['labels'])]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
            # logger.debug(f"Decode Candidates_ids[0][0]: {tokenizer.decode(train_dataset[0]['candidates_ids'][0])}, fclfclfcl") ### 这个输出是空的哈哈哈，不知道为啥
            # logger.debug(f"Candidates_ids[0][0]: {train_dataset[0]['candidates_ids'][0]}, fclfclfcl")
            logger.debug(f"length of dataset, input_ids: {len(train_dataset[0]['input_ids'])}")
            logger.debug(f"length of dataset, input_ids: {len(train_dataset[1]['input_ids'])}")
            logger.debug(f"length of dataset, input_ids: {len(train_dataset[2]['input_ids'])}")
            logger.debug(f"length of dataset, attention_mask: {len(train_dataset[0]['attention_mask'])}")
            logger.debug(f"length of dataset, attention_mask: {len(train_dataset[1]['attention_mask'])}")
            logger.debug(f"length of dataset, attention_mask: {len(train_dataset[2]['attention_mask'])}")
            logger.debug(f"length of dataset, labels_ids: {len(train_dataset[0]['labels'])}")
            logger.debug(f"length of dataset, labels_ids: {len(train_dataset[1]['labels'])}")
            logger.debug(f"length of dataset, labels_ids: {len(train_dataset[2]['labels'])}")
            # logger.debug(f"length of dataset, candidates_ids: {len(train_dataset[0]['candidates_ids'])}")
            # logger.debug(f"length of dataset, candidates_ids: {len(train_dataset[1]['candidates_ids'])}")
            # logger.debug(f"length of dataset, candidates_ids: {len(train_dataset[2]['candidates_ids'])}")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_size = len(eval_dataset)
            logger.debug(f"Num eval_samples: {eval_size}")
            if eval_size > 500:
                logger.warning(f"Num eval_samples is large: {eval_size}, "
                               f"training slow, consider reduce it by `--max_eval_samples=50`")
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                # preprocess_function_valid, ###
                preprocess_function, ### 35 update
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))
    logger.debug("=================Now i'm going to load the model, are you ready?==========================")
    # Load model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp:
            logger.debug("================= i'm in ddp mode ==========================")
            model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
            training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size or 1
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            # logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        # config = config_class.from_pretrained(
        #     model_args.model_name_or_path,
        #     trust_remote_code=model_args.trust_remote_code,
        #     torch_dtype=torch_dtype,
        #     cache_dir=model_args.cache_dir
        # )
        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        config = config_class.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        # Set RoPE scaling
        if model_args.rope_scaling is not None:
            # if hasattr(config, "use_dynamic_ntk"):  # for Qwen models
            #     logger.warning("Qwen model does not support RoPE scaling in training.")
            # elif hasattr(config, "rope_scaling"):  # for LLaMA and Falcon models
            if hasattr(config, "rope_scaling"):
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and script_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(script_args.model_max_length / current_max_length))
                else:
                    logger.warning(f"The model_max_length({script_args.model_max_length}) is smaller than max "
                                   f"length({current_max_length}). Consider increase model_max_length.")
                    scaling_factor = 1.0

                setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
                logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    model_args.rope_scaling, scaling_factor
                ))
            else:
                logger.warning("Current model does not support RoPE scaling.")

        # Set FlashAttention-2
        # if model_args.flash_attn:
        #     if getattr(config, "model_type", None) == "llama":
        #         modeling_llama.LlamaAttention = LlamaFlashAttention2
        #         modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        #         logger.info("Using FlashAttention-2 for faster training and inference.")
        #     elif getattr(config, "model_type", None) == "qwen":
        #         logger.info("Qwen models automatically enable FlashAttention if installed.")
        #     else:
        #         logger.warning("Current model does not support FlashAttention-2.")
        # elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
        #     modeling_llama.LlamaAttention = LlamaShiftShortAttention
        #     logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
        #                    " is RTX4090, A100 or H100.")
        if model_args.flash_attn:
            if is_flash_attn_2_available:
                config_kwargs["use_flash_attention_2"] = True
                logger.info("Using FlashAttention-2 for faster training and inference.")
            else:
                logger.warning("FlashAttention-2 is not installed.")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                           " is RTX3090, RTX4090, A100 or H100.")

        # Set shift short attention (S^2-Attn)
        # Set shifted sparse attention (S^2-Attn)
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                apply_llama_patch()
                # logger.info("Using shift short attention with group_size_ratio=1/4.")
                logger.info("Using shifted sparse attention with group_size_ratio=1/4.")
            else:
                # logger.warning("Current model does not support shift short attention.")
                logger.warning("Current model does not support shifted sparse attention.")

        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        # load_in_8bit_skip_modules = None
        # if load_in_8bit or load_in_4bit:
        #     logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
        #     if script_args.modules_to_save is not None:
        #         load_in_8bit_skip_modules = script_args.modules_to_save.split(',')
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if script_args.qlora:
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch_dtype,
                    )
                else:
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )

        # model = model_class.from_pretrained(### 加载模型
        #     model_args.model_name_or_path,
        #     config=config,
        #     torch_dtype=torch_dtype,
        #     load_in_4bit=load_in_4bit,
        #     load_in_8bit=load_in_8bit,
        #     low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        #     device_map=model_args.device_map,
        #     trust_remote_code=model_args.trust_remote_code,
        #     quantization_config=BitsAndBytesConfig(
        #         load_in_4bit=load_in_4bit,
        #         load_in_8bit=load_in_8bit,
        #         load_in_8bit_skip_modules=load_in_8bit_skip_modules,
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch_dtype,
        #     ) if script_args.qlora else None,
        # )
        
        brio_log_file_path = custom_args.brio_log_file_path ### biubiubiu
        num_candidates = len(raw_datasets['train'][0]['conversations'][0]['candidates_sorted']) ### biubiubiu
        train_batch_size = training_args.train_batch_size ### biubiubiu
        model = model_class.from_pretrained(
            model_args.model_name_or_path, ### 这个位置参数要在关键字参数之前
            num_candidates=num_candidates, ### biubiubiu
            train_batch_size = train_batch_size, ### biubiubiu
            brio_log_file_path=brio_log_file_path, ### biubiubiu
            candidate_label_pad_token_id=IGNORE_INDEX_For_candidates, ### biubiubiu one forward
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=model_args.device_map,
            **config_kwargs,
        )

        # Fix ChatGLM2 and ChatGLM3 LM head
        if getattr(config, "model_type", None) == "chatglm":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # Set NEFTune trick for fine-tuning
        if model_args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):
                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = model_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                input_embed.forward = MethodType(noisy_forward, input_embed)
                logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
            else:
                logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

        # Patch Mixtral MOE model
        if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules  # type: ignore
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # type: ignore

            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)") ### 输出

        # Set fp32 forward hook for lm_head
        output_layer = getattr(model, "lm_head")
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_hook(fp32_forward_post_hook)

        # Load LoRA model
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model") ### 输出
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}") ### 输出
            logger.info(f"Peft lora_rank: {script_args.lora_rank}") ### 输出
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config) ### peft包装
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters() ### 输出
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    # Initialize our Trainer
    # if training_args.gradient_checkpointing:
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # data_collator = DataCollatorForSeq2Seq(### dataloader
    #     tokenizer=tokenizer,
    #     model=model,
    #     label_pad_token_id=IGNORE_INDEX,
    #     pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shift short attention
    # )

    ### biubiubiu
    # num_candidates = len(raw_datasets['train'][0]['conversations'][0]['candidates_sorted']) ### biubiubiu
    # brio_log_file_path = '/data1/fcl/workspace/2024_73/240517_73_sft_brio_logp_around2/continue_sft_training/brio_logs.txt'
    # brio_log_file_path = '/data1/fcl/workspace/2024_73/240329_73_debug_sft_brio/log/brio_logs_multi_devices.txt'
    # 获取当前时间并格式化为字符串
    # formatted_time = time.strftime("%Y%m%d_%H%M%S")
    # 构建包含时间戳的日志路径
    # brio_log_file_path = f'/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/brio_sft_debug/brio_logs_single_devices_240712.txt'
    # brio_log_file_path = custom_args.brio_log_file_path ### biubiubiu
    # brio_log_file_path = f'/data1/fcl/workspace/2024_73/240329_73_debug_sft_brio/log/brio_logs_multi_devices.txt'

    ### biubiubiu
    data_collator = DataCollatorForSeq2SeqBrio(
        tokenizer=tokenizer,
        # brio_log_file_path=brio_log_file_path, ### biubiubiu
        brio_logger=model.brio_logger, ### biubiubiu
        num_candidates=num_candidates, ### biubiubiu
        candidates_input_ids_pad_token_id=0, ### biubiubiu 用unk去pad
        model=model,
        label_pad_token_id=IGNORE_INDEX, ### biubiubiu原sft代码继承里有传入INGORE_INDEX,父类里也有init 默认 = -100
        candidate_label_pad_token_id=IGNORE_INDEX_For_candidates, ### biubiubiu one forward
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None, ### 所以说,pad后会有莫名奇妙的0出现, 因为是最大长度且一定得是4的倍数!
    ) ### innit会执行
    ### biubiubiu

    # Initialize our Trainer
    ### biubiubiu
    num_candidates = len(raw_datasets['train'][0]['conversations'][0]['candidates_sorted'])
    # brio_log_file_path = '/data1/fcl/workspace/2024_73/240517_73_sft_brio_logp_around2/continue_sft_training/brio_logs.txt'
    ### biubiubiu
    trainer = SavePeftModelAndBrioTrainer(
        model=model,
        # brio_log_file_path=brio_log_file_path, ### biubiubiu
        # brio_logger=data_collator.brio_logger, ### biubiubiu
        brio_logger=model.brio_logger, ### biubiubiu
        num_candidates=num_candidates, ### biubiubiu
        input_ids_and_attention_mask_pad_id=0, ### biubiubiu
        labels_pad_id=IGNORE_INDEX, ### biubiubiu -100 给正常的input用的
        candidate_labels_pad_id=IGNORE_INDEX_For_candidates, ### biubiubiu -100
        args=training_args, ### training_args就是我在sh文件里传入的那些，在trainer里是args，然后trainer里可以提取到args.train_batch_size ### biubiubiu
        train_dataset=train_dataset if training_args.do_train else None, ### 在这里把训练集传入了
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        if trainer.is_world_process_zero(): ### 只要一个打印就行了

            # logger.debug("Getting data loader...")
            # dataloader = trainer.get_train_dataloader()
            # logger.debug("Data loader obtained.")

            # logger.debug("Creating iterator...")
            # data_iter = iter(dataloader)
            # logger.debug("Iterator created.")

            # logger.debug("Getting next batch...")
            # sample = next(data_iter)
            # logger.debug("Next batch obtained.")

            sample = next(iter(trainer.get_train_dataloader())) ## 第一次进入get_train_dataloader()这里进去调整删除columns的代码,    第二次回到这里会进入iter里 ### *** 第负一步 ***, 转到上面的函数定义
            logger.debug(f"i have been here in supervised_finetuning_training_py! come and got me!")
            logger.debug(f"Train dataloader example: {sample}")
            # logger.debug(f"Detail input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}")
            logger.debug(f"Decode candidate_1_input_ids[0]:\n{sample['candidate_1_input_ids'][0]}") ### 出错了，修复了，解决方法是：等于说现在已经有了candidate_ids(解释器要选matllmbrio)
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}") ### 这一步之后，-100也变成0了
        checkpoint = None
        if training_args.resume_from_checkpoint is not None: ### 不进入
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint) ## 开始训练

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
