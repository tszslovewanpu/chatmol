# # -*- coding: utf-8 -*-
# # Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Fine-tuning the library models for causal language modeling (GPT, LLaMA, Bloom, ...) on a json file or a dataset.

# part of code is modified from https://github.com/shibing624/textgen
# """

# import math
# import os
# from dataclasses import dataclass, field
# from glob import glob
# from types import MethodType
# from typing import Literal, Optional, Tuple

# import torch
# import torch.nn as nn
# from datasets import load_dataset
# from loguru import logger
# from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
# from transformers import (
#     AutoConfig,
#     BloomForCausalLM,
#     AutoModel,
#     AutoModelForCausalLM,
#     LlamaForCausalLM,
#     BloomTokenizerFast,
#     AutoTokenizer,
#     HfArgumentParser,
#     Trainer,
#     Seq2SeqTrainingArguments,
#     set_seed,
#     BitsAndBytesConfig,
#     DataCollatorForSeq2Seq,
# )
# from transformers.models.llama.modeling_llama import (
#     LlamaAttention,
#     apply_rotary_pos_emb,
#     repeat_kv,
#     LlamaFlashAttention2,
#     Cache
# )
# from transformers.trainer import TRAINING_ARGS_NAME
# from transformers.trainer_pt_utils import LabelSmoother
# from transformers.utils.versions import require_version

# try:
#     from transformers.integrations import is_deepspeed_zero3_enabled
# except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
#     from transformers.deepspeed import is_deepspeed_zero3_enabled

# is_flash_attn_2_available = False
# try:
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import pad_input, unpad_input

#     is_flash_attn_2_available = True
# except ImportError:
#     is_flash_attn_2_available = False

# from template import get_conv_template




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


from template import get_conv_template

### biubiubiu datacollator
### biubiubiu 摘录自35的/data/fcl/anaconda3/envs/matllm/lib/python3.9/site-packages/transformers/data/data_collator.py
### 因为下面用到了
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

class DataCollatorForSeq2SeqNumerical(DataCollatorForSeq2Seq):
    # def __init__(self, num_candidates, brio_log_file_path, candidates_input_ids_pad_token_id, *args, **kwargs):
    # def __init__(self, num_candidates, brio_logger, candidates_input_ids_pad_token_id, label_pad_token_id, candidate_label_pad_token_id, *args, **kwargs):
    # def __init__(self, numerical_logger, *args, **kwargs):
    def __init__(self, numerical_log_file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.num_candidates = num_candidates
        # self.brio_logger = self._setup_brio_logger(brio_log_file_path)
        # 创建新的 logger
        # self._setup_brio_logger(brio_log_file_path)
        # self.numerical_logger = numerical_logger
        self._setup_numerical_logger(numerical_log_file_path)
        self.numerical_logger_datacollator.info("now, i'm in datacollatorforseq2seqbrio.")
        # self.candidates_input_ids_pad_token_id = candidates_input_ids_pad_token_id ### biubiubiu
        # self.label_pad_token_id = label_pad_token_id
        # self.candidate_label_pad_token_id = candidate_label_pad_token_id
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

    # def _setup_brio_logger(self, brio_log_file_path):
    #     # 创建输出到文件的 logger
    #     file_handler = FileHandler(brio_log_file_path)
    #     file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     file_handler.setFormatter(file_formatter)
    #     self.brio_logger = logging.getLogger('BRIO_logger')
    #     self.brio_logger.addHandler(file_handler)
    #     self.brio_logger.setLevel(logging.DEBUG)

    def _setup_numerical_logger(self, numerical_log_file_path):
        file_handler = FileHandler(numerical_log_file_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.numerical_logger_datacollator = logging.getLogger('numerical_logger_datacollator')
        self.numerical_logger_datacollator.addHandler(file_handler)
        self.numerical_logger_datacollator.setLevel(logging.DEBUG)
        self.numerical_logger_datacollator.info(f"numerical_logger_datacollator has been setup.")


    def __call__(self, features, return_tensors=None):
        self.numerical_logger_datacollator.info("now, i'm in datacollatorforseq2seqNumerical's call function, it pad labels first and pad the rest align with labels")
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
### biubiubiu







class SavePeftModelAndNumericalTrainer(Trainer):
    """
    Trainer for lora models and for brio sft training
    """
    # def __init__(self, num_candidates, brio_logger, input_ids_and_attention_mask_pad_id, labels_pad_id, candidate_labels_pad_id, *args, **kwargs):
    # def __init__(self, number_of_numbers, numerical_logger, *args, **kwargs):
    def __init__(self, number_of_numbers, numerical_log_file_path, *args, **kwargs):
        super().__init__(*args, **kwargs) ### 这里将train_batchsize传给了Trainer父类
        self.number_of_numbers = number_of_numbers
        # self.numerical_logger = numerical_logger
        self._setup_numerical_logger(numerical_log_file_path)
        # self.input_ids_and_attention_mask_pad_id = input_ids_and_attention_mask_pad_id
        # self.labels_pad_id = labels_pad_id
        # self.candidate_labels_pad_id = candidate_labels_pad_id

    def _setup_numerical_logger(self, numerical_log_file_path):
        file_handler = FileHandler(numerical_log_file_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.numerical_logger_trainer = logging.getLogger('numerical_logger_trainer')
        self.numerical_logger_trainer.addHandler(file_handler)
        self.numerical_logger_trainer.setLevel(logging.DEBUG)
        self.numerical_logger_trainer.info(f"numerical_logger_trainer has been setup.")

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)

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
            # for j in range(1, self.num_candidates + 1):
            for j in range(1, self.number_of_numbers + 1):
                self._signature_columns.append(f'number_{j}')
                self._signature_columns.append(f'number_{j}_number_start_idx') ### biubiubiu
                self._signature_columns.append(f'number_{j}_number_end_idx') ### biubiubiu

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
### 确认有没有传进来 biubiubiu
        if 'number_1_number_start_idx' not in self.train_dataset[0]:
            raise ValueError("Training dataset does not contain 'number_1_start_number_idx. Please check your preprocessing function.")
        if 'number_1' not in self.train_dataset[0]:
            raise ValueError("Training dataset does not contain 'number_1. Please check your preprocessing function.")
        if 'number_1_number_end_idx' not in self.train_dataset[0]:
            raise ValueError("Training dataset does not contain 'number_1_end_number_idx. Please check your preprocessing function.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training") ### 下面这个打印显示没有candidates_ids了，所以问题出在了这里，get in
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
        self.numerical_logger_trainer.info(f"shape of inputs is: input_ids: {inputs['input_ids'].shape}, attention_mask: {inputs['attention_mask'].shape}, labels: {inputs['labels'].shape}")
        for i in range(self.number_of_numbers):
            self.numerical_logger_trainer.info(f"number{i+1} and its shape is: {inputs[f'number_{i+1}']}, {inputs[f'number_{i+1}'].shape}, number{i+1}_start_idx and its shape is: {inputs[f'number_{i+1}_number_start_idx']}, {inputs[f'number_{i+1}_number_start_idx'].shape}, number{i+1}_end_idx and its shape is: {inputs[f'number_{i+1}_number_end_idx']}, {inputs[f'number_{i+1}_number_end_idx'].shape}")
        # self.brio_logger.info(f"size of inputs['input_ids'] is: {inputs['input_ids'].size()}, size of inputs['attention_mask'] is: {inputs['attention_mask'].size()}, size of inputs['labels'] is: {inputs['labels'].size()}")
        ### 送入模型的不要candidates ***很重要，因为llama2模型forward不接收candidates***
        # inputs_lm = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": inputs["labels"]} ### 其实这里能运行也是奇怪的，因为inputs的data里才是字典，但是我好像可以直接调用字典跳过data
        # outputs = model(**inputs_lm)## ***进入***
        # outputs = model(**{
        # "input_ids": inputs["input_ids"],
        # "attention_mask": inputs["attention_mask"],
        # "labels": inputs["labels"]})

### biubiubiu可视化一下
# {'input_ids': tensor([[128000,     32,   6369,   17,      6, 128001, 128001], [128000,     32,   6369,   435,      6,      128001, 128001]], device='cuda:0'), 
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1,  1, 1, 0, 0, 0, 0, 0, 0]], device='cuda:0'), 
# 'labels': tensor([[  -100,   -100,  6, 128001,   -100], [  -100,   -100,   -100,  -100]], device='cuda:0'), 
# 'number_1': tensor([-2.2400, -2.1600], device='cuda:0'), 
# 'number_1_number_start_idx': tensor([43, 43], device='cuda:0'), 
# 'number_1_number_end_idx': tensor([46, 46], device='cuda:0')}
# 后面三个的shape都是torch.Size([2])
        ### biubiubiu numerical 这里将inputs调整一下，将所有的number融合到一个里头
        numbers = []
        numbers_start_idx = []
        numbers_end_idx = []
        
        for i in range(self.number_of_numbers):
            self.numerical_logger_trainer.info(f"number{i+1} and its shape is: {inputs[f'number_{i+1}']}, {inputs[f'number_{i+1}'].shape}, number{i+1}_start_idx and its shape is: {inputs[f'number_{i+1}_number_start_idx']}, {inputs[f'number_{i+1}_number_start_idx'].shape}, number{i+1}_end_idx and its shape is: {inputs[f'number_{i+1}_number_end_idx']}, {inputs[f'number_{i+1}_number_end_idx'].shape}")
            
            numbers.append(inputs.pop(f'number_{i+1}').unsqueeze(1))
            numbers_start_idx.append(inputs.pop(f'number_{i+1}_number_start_idx').unsqueeze(1))
            numbers_end_idx.append(inputs.pop(f'number_{i+1}_number_end_idx').unsqueeze(1))
        
        # inputs.pop(f'number_{i+1}_number_end_idx')  -->  tensor([46, 46], device='cuda:0')

        # inputs.pop(f'number_{i+1}_number_start_idx').unsqueeze(1) -->  
        # tensor([[43],
        #         [43]], device='cuda:0')

        inputs['numbers'] = torch.cat(numbers, dim=1)
        inputs['numbers_start_idx'] = torch.cat(numbers_start_idx, dim=1)
        inputs['numbers_end_idx'] = torch.cat(numbers_end_idx, dim=1)

### biubiubiu 处理完后可视化一下
# {'input_ids': tensor([[128000,  128001, 128001],[128000, 128001]],device='cuda:0'), 
# 'attention_mask': tensor([[ 0],[1, 1, 1,  0]], device='cuda:0'), 
# 'labels': tensor([[  -100,   128001,   -100], [  -100,  -100]], device='cuda:0'), 
# 'numbers': tensor([[-2.2400], [-2.1600]], device='cuda:0'), 
# 'numbers_start_idx': tensor([[43], [43]], device='cuda:0'), 
# 'numbers_end_idx': tensor([[46], [46]], device='cuda:0')}
# inputs['numbers'].shape  -->  torch.Size([2, 1])

        outputs = model(**inputs)

        self.numerical_logger_trainer.info(f"shape of outputs.logits is: {outputs.logits.shape}")
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





### 应该是没有重写，但是我需要记录里面的过程
    def _inner_training_loop(
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
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

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
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
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
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

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
                self.state.save_steps = args.save_steps

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

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
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
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

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

            step = -1
            for step, inputs in enumerate(epoch_iterator): ### 一步就可以有2次卧槽    def __call__(self, features, return_tensors=None):进入两次
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
                    tr_loss_step = self.training_step(model, inputs)

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
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
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
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

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
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
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






### numerical
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
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
class LlamaModelNumerical(LlamaModel):
    def __init__(self, config):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        ### numerical
        numbers: Optional[torch.Tensor] = None, ### biubiubiu
        numbers_start_idx: Optional[torch.Tensor] = None, ### biubiubiu
        numbers_end_idx: Optional[torch.Tensor] = None, ### biubiubiu
        ### numerical
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # input_ids.shape    torch.Size([2, 124])
        # input_ids.device    device(type='cuda', index=0)
        # inputs_embeds.shape    torch.Size([2, 124, 4096])
        # inputs_embeds.device    device(type='cuda', index=0)

### numerical ******
        # 首先拷贝一个和inputs_embeds形状一样的全零张量enhance_embeds（维度为torch.Size([2, 124, 4096])）
        zero_tensor = torch.zeros_like(inputs_embeds)

        def numerical_embed(numbers, numbers_start_idx, numbers_end_idx, zero_tensor):
            batch_size, seq_len, hidden_dim = zero_tensor.shape
            enhance_embeds = zero_tensor.clone()
            for i in range(batch_size):
                for j in range(len(numbers[i])):
                    start_idx = numbers_start_idx[i][j].item()
                    end_idx = numbers_end_idx[i][j].item()
                    number = numbers[i][j].item()
                    slice_tensor = enhance_embeds[i, start_idx:end_idx + 1, :]
                    # embedding
                    div_term = 10000 ** (torch.arange(0, hidden_dim, 2) / hidden_dim)
                    sin_vals = torch.sin(number / div_term)
                    cos_vals = torch.cos(number / div_term)
                    
                    for token in range(slice_tensor.shape[0]):# seq_len维度，每个数字的token，都加上数字整体的编码
                        slice_tensor[token, 0::2] = sin_vals
                        slice_tensor[token, 1::2] = cos_vals
                    
                    enhance_embeds[i, start_idx:end_idx + 1, :] = slice_tensor # 好像是多余的
            return enhance_embeds

# number
# 0.4300000071525574
# start_idx
# 59
# end_idx
# 62
# enhance_embeds[i, start_idx:end_idx + 1, :]
# tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16)
# enhance_embeds[i, start_idx:end_idx + 1, :]
# tensor([[4.1602e-01, 9.1016e-01, 4.1602e-01,  ..., 1.0000e+00, 4.3154e-05,
#          1.0000e+00],
#         [4.1602e-01, 9.1016e-01, 4.1602e-01,  ..., 1.0000e+00, 4.3154e-05,
#          1.0000e+00],
#         [4.1602e-01, 9.1016e-01, 4.1602e-01,  ..., 1.0000e+00, 4.3154e-05,
#          1.0000e+00],
#         [4.1602e-01, 9.1016e-01, 4.1602e-01,  ..., 1.0000e+00, 4.3154e-05,
#          1.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

# number
# -10.109999656677246
# start_idx
# 71
# end_idx
# 74
# slice_tensor
# tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16)
# enhance_embeds[i, start_idx:end_idx + 1, :]
# tensor([[ 0.6328, -0.7734,  0.5977,  ...,  1.0000, -0.0010,  1.0000],
#         [ 0.6328, -0.7734,  0.5977,  ...,  1.0000, -0.0010,  1.0000],
#         [ 0.6328, -0.7734,  0.5977,  ...,  1.0000, -0.0010,  1.0000],
#         [ 0.6328, -0.7734,  0.5977,  ...,  1.0000, -0.0010,  1.0000]],
#        device='cuda:0', dtype=torch.bfloat16)


# enhance_embeds[i, 59:75, :]
# tensor([[ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#           4.3154e-05,  1.0000e+00],
#         [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#           4.3154e-05,  1.0000e+00],
#         [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#           4.3154e-05,  1.0000e+00],
#         ...,
#         [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#          -1.0147e-03,  1.0000e+00],
#         [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#          -1.0147e-03,  1.0000e+00],
#         [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#          -1.0147e-03,  1.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

# 第一个batch处理完
# enhance_embeds[:, 59:75, :]
# tensor([[[ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          ...,
#          [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#           -1.0147e-03,  1.0000e+00],
#          [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#           -1.0147e-03,  1.0000e+00],
#          [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#           -1.0147e-03,  1.0000e+00]],

#         [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          ...,
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00]]], device='cuda:0', dtype=torch.bfloat16)

# enhance_embeds[:, 59:86, :]
# tensor([[[ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          ...,
#          [ 6.0547e-01, -7.9688e-01,  6.1719e-01,  ...,  1.0000e+00,
#            2.4986e-04,  1.0000e+00],
#          [ 6.0547e-01, -7.9688e-01,  6.1719e-01,  ...,  1.0000e+00,
#            2.4986e-04,  1.0000e+00],
#          [ 6.0547e-01, -7.9688e-01,  6.1719e-01,  ...,  1.0000e+00,
#            2.4986e-04,  1.0000e+00]],

#         [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          ...,
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00]]], device='cuda:0', dtype=torch.bfloat16)

# 所有batch处理完
# enhance_embeds[:, 59:75, :]
# tensor([[[ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          ...,
#          [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#           -1.0147e-03,  1.0000e+00],
#          [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#           -1.0147e-03,  1.0000e+00],
#          [ 6.3281e-01, -7.7344e-01,  5.9766e-01,  ...,  1.0000e+00,
#           -1.0147e-03,  1.0000e+00]],

#         [[ 2.1777e-01,  9.7656e-01,  2.1680e-01,  ...,  1.0000e+00,
#            2.2054e-05,  1.0000e+00],
#          [ 2.1777e-01,  9.7656e-01,  2.1680e-01,  ...,  1.0000e+00,
#            2.2054e-05,  1.0000e+00],
#          [ 2.1777e-01,  9.7656e-01,  2.1680e-01,  ...,  1.0000e+00,
#            2.2054e-05,  1.0000e+00],
#          ...,
#          [ 8.7891e-01, -4.7461e-01,  8.5547e-01,  ...,  1.0000e+00,
#           -1.0529e-03,  1.0000e+00],
#          [ 8.7891e-01, -4.7461e-01,  8.5547e-01,  ...,  1.0000e+00,
#           -1.0529e-03,  1.0000e+00],
#          [ 8.7891e-01, -4.7461e-01,  8.5547e-01,  ...,  1.0000e+00,
#           -1.0529e-03,  1.0000e+00]]], device='cuda:0', dtype=torch.bfloat16)

# enhance_embeds[:, 59:86, :]
# tensor([[[ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          [ 4.1602e-01,  9.1016e-01,  4.1602e-01,  ...,  1.0000e+00,
#            4.3154e-05,  1.0000e+00],
#          ...,
#          [ 6.0547e-01, -7.9688e-01,  6.1719e-01,  ...,  1.0000e+00,
#            2.4986e-04,  1.0000e+00],
#          [ 6.0547e-01, -7.9688e-01,  6.1719e-01,  ...,  1.0000e+00,
#            2.4986e-04,  1.0000e+00],
#          [ 6.0547e-01, -7.9688e-01,  6.1719e-01,  ...,  1.0000e+00,
#            2.4986e-04,  1.0000e+00]],

#         [[ 2.1777e-01,  9.7656e-01,  2.1680e-01,  ...,  1.0000e+00,
#            2.2054e-05,  1.0000e+00],
#          [ 2.1777e-01,  9.7656e-01,  2.1680e-01,  ...,  1.0000e+00,
#            2.2054e-05,  1.0000e+00],
#          [ 2.1777e-01,  9.7656e-01,  2.1680e-01,  ...,  1.0000e+00,
#            2.2054e-05,  1.0000e+00],
#          ...,
#          [ 7.1875e-01, -6.9531e-01,  7.2656e-01,  ...,  1.0000e+00,
#            2.3460e-04,  1.0000e+00],
#          [ 7.1875e-01, -6.9531e-01,  7.2656e-01,  ...,  1.0000e+00,
#            2.3460e-04,  1.0000e+00],
#          [ 7.1875e-01, -6.9531e-01,  7.2656e-01,  ...,  1.0000e+00,
#            2.3460e-04,  1.0000e+00]]], device='cuda:0', dtype=torch.bfloat16)


        # 将全零张量和numbers_start_idx（维度为torch.Size([2, 1])）和numbers_end_idx（维度为torch.Size([2, 1])）和numbers（维度为torch.Size([2, 1])）一起送入函数numerical_embed()
### multi biubiubiu
# numbers
# tensor([[  0.4300, -10.1100,   2.4900],
#         [  0.2200, -10.5000,   2.3400]], device='cuda:0')
# numbers_start_idx
# tensor([[59, 71, 82],
#         [59, 71, 82]], device='cuda:0')
# numbers_end_idx
# tensor([[62, 74, 85],
#         [62, 74, 85]], device='cuda:0')
### multi biubiubiu
        enhance_embeds = numerical_embed(numbers, numbers_start_idx, numbers_end_idx, zero_tensor)

        # 函数内部，将传入的这个全零张量切片enhance_embeds[:,nummbers_start_idx:numbers_end_idx+1,:]
        # 这个切片出来的部分维度为torch.Size([2, 4, 4096])，我们针对这部分做处理
        # 首先根据第一维度batch，取numbers的对应该batch的值number和切片张量的对应该batch的部分，再根据切片张量的最后一个维度（隐藏层维度d）计算我想要的embedding值
        # 计算公式如下，可以根据number和隐藏层位置，得到隐藏层维度不同位置的embedding值
# enhance_embeds[:,numbers_start_idx[0]-1 : numbers_end_idx[0]+2,:]
# tensor([[[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [-8.8281e-01, -4.7070e-01, -8.8672e-01,  ...,  1.0000e+00,
#           -2.0695e-04,  1.0000e+00],
#          [-8.8281e-01, -4.7070e-01, -8.8672e-01,  ...,  1.0000e+00,
#           -2.0695e-04,  1.0000e+00],
#          [-8.8281e-01, -4.7070e-01, -8.8672e-01,  ...,  1.0000e+00,
#           -2.0695e-04,  1.0000e+00],
#          [-8.8281e-01, -4.7070e-01, -8.8672e-01,  ...,  1.0000e+00,
#           -2.0695e-04,  1.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00]],

#         [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [-8.9453e-01, -4.4336e-01, -9.0234e-01,  ...,  1.0000e+00,
#           -2.0409e-04,  1.0000e+00],
#          [-8.9453e-01, -4.4336e-01, -9.0234e-01,  ...,  1.0000e+00,
#           -2.0409e-04,  1.0000e+00],
#          [-8.9453e-01, -4.4336e-01, -9.0234e-01,  ...,  1.0000e+00,
#           -2.0409e-04,  1.0000e+00],
#          [-8.9453e-01, -4.4336e-01, -9.0234e-01,  ...,  1.0000e+00,
#           -2.0409e-04,  1.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00]]], device='cuda:0', dtype=torch.bfloat16)
        inputs_embeds = inputs_embeds + enhance_embeds
### numerical ******

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
### numerical


### numerical enhancement
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
class LlamaForCausalLMNumerical(LlamaForCausalLM):
    def __init__(self, config, train_batch_size, numerical_log_file_path):
        ######
        # super().__init__(config) # 不要直接用，因为父类LlamaForCausalLM(LlamaPreTrainedModel)的init里会创建一个LlamaModel的实例，这个实例我不希望他是由LlamaModel创建的，而是用我自定义（继承自LlamaModel）的子类创建
        ######

        ######
        # 这是父类class LlamaForCausalLM(LlamaPreTrainedModel):的init函数
                    # class LlamaForCausalLM(LlamaPreTrainedModel):
                    #     _tied_weights_keys = ["lm_head.weight"]

                    #     def __init__(self, config):
                    #         super().__init__(config)
                    #         self.model = LlamaModel(config) ### 居然不先来这里
                    #         self.vocab_size = config.vocab_size
                    #         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

                    #         # Initialize weights and apply final processing
                    #         self.post_init()
        ######

        ######
        # 我将手动写下来父类class LlamaForCausalLM(LlamaPreTrainedModel)的init函数里的内容
        # 首先是爷爷类
        from transformers.models.llama.modeling_llama import LlamaPreTrainedModel # 先导入爷爷类
        LlamaPreTrainedModel.__init__(self, config) # 因此，手动初始化父类的父类
        # 最关键的
        self.model = LlamaModelNumerical(config) ### 最关键的，一切的起因，是要这一行重写
        # 剩下的不变
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        ######

        ######
        # 下面写我自己class LlamaForCausalLMNumerical(LlamaForCausalLM):的init函数里需要添加的东西
        self.train_batch_size = train_batch_size
        self._setup_numerical_logger(numerical_log_file_path)
        ######
    
    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # def forward( ### step1 先来到了这里进行forward的配置
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     r"""
    #     Args:
    #         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    #             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    #             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    #     Returns:

    #     Example:

    #     ```python
    #     >>> from transformers import AutoTokenizer, LlamaForCausalLM

    #     >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    #     >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    #     >>> prompt = "Hey, are you conscious? Can you talk to me?"
    #     >>> inputs = tokenizer(prompt, return_tensors="pt")

    #     >>> # Generate
    #     >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    #     >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    #     ```"""
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions ### None 然后是 后面self的值，False
    #     output_hidden_states = ( ### None 然后是，最终选了这个self.config.output_hidden_states=False
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict ### None，然后是，最终选了这个self.config.use_return_dict=True
    #     # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #     ) ### outputs.last_hidden_state.shape = torch.Size([21, 212, 4096])

    #     hidden_states = outputs[0] ### here torch.Size([3, 184, 4096]) ### one forward hidden_states.shape = torch.Size([21, 212, 4096]) 
    #     if self.config.pretraining_tp > 1:
    #         lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
    #         logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
    #         logits = torch.cat(logits, dim=-1)
    #     else:
    #         logits = self.lm_head(hidden_states) ### here， torch.Size([3, 184, 128256])    self.lm_head = Linear(in_features=4096, out_features=128256, bias=False) ### logits.shape = torch.Size([21, 212, 128256]) one forward 

    #     logits = logits.float() #### logits.shape = torch.Size([14, 212, 128256])

    #     loss_ce = None
    #     loss_rk = None
    #     ### split the logits torch.Size([21, 212, 128256])\input_ids torch.Size([21, 212])\attention_mask torch.Size([21, 212])\labels torch.Size([21, 212])
    #     logits_ce = logits[:self.train_batch_size, :, :] #### torch.Size([2, 212, 128256])
    #     logits_rk = logits[self.train_batch_size:, :, :] #### torch.Size([12, 212, 128256])
    #     # input_ids_ce = input_ids[:self.num_candidates, :]
    #     # input_ids_rk = input_ids[self.num_candidates:, :]
    #     # attention_mask_ce = attention_mask[:self.num_candidates, :]
    #     # attention_mask_rk = attention_mask[self.num_candidates:, :]
    #     labels_ce = labels[:self.train_batch_size, :] #### torch.Size([2, 212])
    #     labels_rk = labels[self.train_batch_size:, :] #### torch.Size([12, 212])

    #     # ### compute the cross entropy loss
    #     if labels_ce is not None:
    #         # Shift so that tokens < n predict n
    #         shift_logits_ce = logits_ce[..., :-1, :].contiguous() #### 2,211,128256
    #         shift_labels_ce = labels_ce[..., 1:].contiguous() #### 2,211
    #         # Flatten the tokens
    #         loss_ce_fct = CrossEntropyLoss()
    #         shift_logits_ce = shift_logits_ce.view(-1, self.config.vocab_size) #### torch.Size([422, 128256])
    #         shift_labels_ce = shift_labels_ce.view(-1) #### torch.Size([422])
    #         # Enable model parallelism
    #         shift_labels_ce = shift_labels_ce.to(shift_logits_ce.device) #### torch.Size([422])
    #         loss_ce = loss_ce_fct(shift_logits_ce, shift_labels_ce) #### torch.Size([]) tensor(0.4808, device='cuda:0', grad_fn=<NllLossBackward0>)

    #     # compute the ranking loss 
    #     logits_rk_log_prob = F.log_softmax(logits_rk, dim=-1) #### torch.Size([12, 212, 128256])
    #     if labels_rk is not None:
    #         shift_logits_rk_log_prob = logits_rk_log_prob[..., :-1, :].contiguous() #### torch.Size([12, 211, 128256])
    #         shift_labels_rk = labels_rk[..., 1:].contiguous() #### torch.Size([12, 211])
    #         # 创建一个掩码来过滤掉 -100 的值
    #         # mask = shift_labels_rk != 128138
    #         mask = (shift_labels_rk != self.candidate_label_pad_token_id )
    #         # torch.Size([12, 211])
    #         # mask
    #         # tensor([[False, False, False,  ..., False, False, False],
    #         #         [False, False, False,  ..., False, False, False],
    #         #         [False, False, False,  ...,  True,  True, False],
    #         #         ...,
    #         #         [False, False, False,  ..., False, False, False],
    #         #         [False, False, False,  ..., False, False, False],
    #         #         [False, False, False,  ..., False, False, False]], device='cuda:0')

    #         penalty_length = mask.sum(dim=1) ### penalty_length = tensor([ 98,  96, 111,  88,  83,  91,  97, 103,  74, 105,  85,  82], device='cuda:0') torch.Size([12])

    #         # 使用掩码只选择有效的值
    #         # filtered_shift_labels_rk = shift_labels_rk[mask]
    #         # filtered_shift_logits_rk_log_prob = shift_logits_rk_log_prob[mask.unsqueeze(-1).expand_as(shift_logits_rk_log_prob)]

    #         # 使用 gather 方法获取下一个 token 的对数概率
    #         # next_token_log_prob = filtered_shift_logits_rk_log_prob.gather(2, filtered_shift_labels_rk.unsqueeze(-1)).squeeze(-1)
    #         next_token_log_prob = shift_logits_rk_log_prob.gather(2, shift_labels_rk.unsqueeze(-1)).squeeze(-1) ### -100 怎么办？能提取到位置吗，替换成了reserved 133，即128138 #### shift_labels_rk.unsqueeze(-1).shape = torch.Size([12, 211, 1]) ### torch.Size([12, 211])

    #         next_token_log_prob = next_token_log_prob * mask.float() ### torch.Size([12, 211])

    #         # mean_next_token_log_prob = next_token_log_prob.mean(dim=1) ### /(D+S) --> /S
    #         ### length penalty
    #         sum_penalty_next_token_log_prob = next_token_log_prob.sum(dim=1) / (penalty_length * penalty_length) ### torch.Size([12])

    #         candidates_scores = self.reshape_tensor(sum_penalty_next_token_log_prob, self.train_batch_size, self.num_candidates).to(self.device) ### biubiubiu ? torch.Size([2, 6])

    #         loss_rk_split = self.compute_ranking_loss(candidates_scores) ### torch.Size([2, 6, 6])
    #         # self.numerical_logger.info(f"The ranking loss device is: {loss_rk.device}") ### 
    #         loss_rk = torch.sum(loss_rk_split)

    #     total_loss = (0.1 * loss_ce) + (10.0 * loss_rk)
    #     self.numerical_logger.info(f"model device: {self.device}, cross_entropy_loss: {loss_ce}, ranking_loss: {loss_rk}, total loss: {total_loss},\nc_e loss device: {loss_ce.device}, r_k loss device: {loss_rk.device}, total loss device: {total_loss.device}") ### ranking_loss_sum = tensor(0.2904, device='cuda:0', grad_fn=<SumBackward0>)
    #     return CausalLMOutputWithPast(
    #         loss=total_loss,
    #         logits=logits_ce,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )
    
    def _setup_numerical_logger(self, numerical_log_file_path):
        # 创建输出到文件的 logger
        file_handler = FileHandler(numerical_log_file_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.numerical_logger_model = logging.getLogger('numerical_logger_model')
        self.numerical_logger_model.addHandler(file_handler)
        self.numerical_logger_model.setLevel(logging.DEBUG)
        self.numerical_logger_model.info(f"numerical_logger_model has been setup.")

    def forward( ### step1 先来到了这里进行forward的配置
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        numbers: Optional[torch.Tensor] = None, ### biubiubiu
        numbers_start_idx: Optional[torch.Tensor] = None, ### biubiubiu
        numbers_end_idx: Optional[torch.Tensor] = None, ### biubiubiu
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
        outputs = self.model( ### 进入       ******   self.model = LlamaModelNumerical(config) ### 最关键的，一切的起因，是要这一行重写
            numbers=numbers, ### biubiubiu
            numbers_start_idx=numbers_start_idx, ### biubiubiu
            numbers_end_idx=numbers_end_idx, ### biubiubiu
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
        )

        hidden_states = outputs[0] ### here torch.Size([3, 184, 4096]) ### one forward hidden_states.shape = torch.Size([21, 212, 4096]) 
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states) ### here torch.Size([3, 184, 128256])    self.lm_head = Linear(in_features=4096, out_features=128256, bias=False) ### logits.shape = torch.Size([21, 212, 128256]) one forward
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

### numerical enhancement







































































MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    # "llama": (AutoConfig, LlamaForCausalLM, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLMNumerical, AutoTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}

@dataclass
class CustomArguments:
    # finetune_path: str = field(default=None, metadata={"help": "."})
    # custom_arg2: int = field(default=123, metadata={"help": "."})
    numerical_log_file_path: str = field(default=None, metadata={"help": "."})
    number_of_numbers: int = field(default=None, metadata={"help": "."})
    # cut_off_keywords: list = field(default=None, metadata={"help": "."})
    # cut_off_keywords: List[str] = field(default=None, metadata={"help": "."})
    # cut_off_keywords_before: str = field(default=None, metadata={"help": "."})
    # cut_off_keywords_after: str = field(default=None, metadata={"help": "."})
    # brio_candidate_labels_pad_id: int = field(default=128138, metadata={"help": "."})

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
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


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


### numerical biubiubiu
# import logging

def setup_main_numerical_logger(numerical_log_file_path):
    # main_numerical_logger = logging.getLogger('numerical_logger')
    # main_numerical_logger.setLevel(logging.DEBUG)
    # file_handler = logging.FileHandler(numerical_log_file_path)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # main_numerical_logger.addHandler(file_handler)
    # return main_numerical_logger

    file_handler = FileHandler(numerical_log_file_path)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    numerical_logger_main = logging.getLogger('numerical_logger_main')
    numerical_logger_main.addHandler(file_handler)
    numerical_logger_main.setLevel(logging.DEBUG)
    numerical_logger_main.info(f"numerical_logger has been setup.")
    return numerical_logger_main
### numerical biubiubiu





def main():
    # parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
    # model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments, CustomArguments)) ###
    model_args, data_args, training_args, script_args, custom_args= parser.parse_args_into_dataclasses() ###

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")
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
    prompt_template = get_conv_template(script_args.template_name)
    if tokenizer.eos_token_id is None:
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
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
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
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train_dataset.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_length = script_args.model_max_length


    ### biubiubiu
    ### setup the numerical logger used in main function
    numerical_logger_main = setup_main_numerical_logger(custom_args.numerical_log_file_path)
    ### biubiubiu

    # def preprocess_function(examples, keywords_before, keywords_after, main_numerical_logger):
    def preprocess_function(examples, numerical_logger_main, number_of_numbers):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        # number_list = []
        # number_idx_list = []
        returned_number_lists = []
        returned_start_index_lists = []
        returned_end_index_lists = []
        for _ in range(number_of_numbers):
            returned_number_lists.append([])
            returned_start_index_lists.append([])
            returned_end_index_lists.append([])
        roles = ["human", "gpt"]
        ### cutoff biubiubiu
        # main_numerical_logger.info(f"have a look at keywords:{keywords_before}")
        # main_numerical_logger.info(f"have a look at keywords:{keywords_after}")

        # # before_numbers_ids = tokenizer(keywords_before, return_tensors="pt", additional_special_tokens=False)
        # before_numbers_ids = tokenizer(keywords_before, return_tensors="pt", add_special_tokens=False)

        # main_numerical_logger.info(f"ids before number is:{before_numbers_ids}")

        # after_numbers_ids = tokenizer(keywords_after, return_tensors="pt")
        # main_numerical_logger.info(f"ids after number is:{after_numbers_ids}")
        
        # def extract_number_and_idx(input_ids, before_numbers_ids, after_numbers_ids):
        def extract_number_and_idx(input_ids):
            import re
            # # 将 input_ids, before_numbers_ids 和 after_numbers_ids 转换为列表
            # input_ids_list = input_ids.tolist()
            # before_numbers_list = before_numbers_ids.tolist()
            # after_numbers_list = after_numbers_ids.tolist()
            
            # 在 input_ids 中找到 before_numbers_ids 的起始索引
            # start_idx = -1
            # for i in range(len(input_ids) - len(before_numbers_ids) + 1):
            #     if input_ids[i:i + len(before_numbers_ids)] == before_numbers_ids:
            #         start_idx = i + len(before_numbers_ids)
            #         break
            
            # # 在 input_ids 中找到 after_numbers_ids 的起始索引
            # end_idx = -1
            # for i in range(start_idx, len(input_ids) - len(after_numbers_ids) + 1):
            #     if input_ids[i:i + len(after_numbers_ids)] == after_numbers_ids:
            #         end_idx = i
            #         break
            
            # # 提取夹杂在 before_numbers_ids 和 after_numbers_ids 中间的 ids
            # if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            #     ids_between = input_ids[start_idx:end_idx]
            # else:
            #     ids_between = []

            # return ids_between, start_idx, end_idx


        ### biubiubiu 处理单一数字情况
            # decoded_text = tokenizer.decode(input_ids)
            # number_pattern = re.compile(r'-?\d+\.\d+')  # 匹配浮点数模式
            # matches = list(number_pattern.finditer(decoded_text))
            # first_match = matches[0] if matches else None
            # if first_match:
            #     # print(f"First match: {first_match.group()} at position {first_match.start()}-{first_match.end()}")

            #     # num_start = first_match.start()
            #     # num_end = first_match.end()
            #     # number_token_ids = tokenizer.encode(decoded_text[num_start:num_end], add_special_tokens=False)
            #     # number_token_ids_2 = tokenizer.encode("float(first_match.group())", add_special_tokens=False)
            #     new_string = ' ' + first_match.group()
            #     number_token_ids = tokenizer.encode(new_string, add_special_tokens=False)
                
            #     def find_sublist_indexes(lst, sublst):
            #         for i in range(len(lst) - len(sublst) + 1):
            #             if lst[i:i+len(sublst)] == sublst:
            #                 main_numerical_logger.info("yes, i find it")
            #                 return i, i + len(sublst) - 1
            #         main_numerical_logger.info("no, i didn't find it")
            #         return None, None
            #     start_idx, end_idx = find_sublist_indexes(input_ids, number_token_ids)
            # return first_match.group(), start_idx, end_idx
        ### biubiubiu 处理单一数字情况


        ### biubiubiu 处理多数字情况
            decoded_text = tokenizer.decode(input_ids)
            number_pattern = re.compile(r'-?\d+\.\d+')
            matches = list(number_pattern.finditer(decoded_text))

            def find_sublist_indexes(lst, sublst):
                for i in range(len(lst) - len(sublst) + 1):
                    if lst[i:i+len(sublst)] == sublst:
                        numerical_logger_main.info("yes, i find it")
                        return i, i + len(sublst) - 1
                numerical_logger_main.info("no, i didn't find it")
                return None, None

            results = []

            for match in matches:
                new_string = ' ' + match.group()
                number_token_ids = tokenizer.encode(new_string, add_special_tokens=False)
                start_idx, end_idx = find_sublist_indexes(input_ids, number_token_ids)
                # results.append((match.group(), (start_idx, end_idx)))
                results.append((float(match.group()), start_idx, end_idx))
            return results
        ### biubiubiu 处理多数字情况

### cutoff biubiubiu

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
            
            numerical_logger_main.info(f"The constructed input ids is:{input_ids}")
            numerical_logger_main.info(f"Now extract the numbers and get the index of numbers")
            # number_ids, idx_start, idx_end = extract_number_and_idx(input_ids, before_numbers_ids, after_numbers_ids)
            # number = tokenizer.decode(number_ids)
            number_and_idx_tuple_list = extract_number_and_idx(input_ids)

            # # Ensure that the lists are long enough 转而在上面处理了
            # while len(returned_number_lists) < len(number_and_idx_tuple_list):
            #     returned_number_lists.append([])
            #     returned_index_lists.append([])
            
            # 这样每次循环都清空了
            # returned_number_lists = [[] for _ in range(len(number_and_idx_tuple_list))]
            # returned_index_lists = [[] for _ in range(len(number_and_idx_tuple_list))]

            # for i, (number, (idx_start, idx_end)) in enumerate(number_and_idx_tuple_list):
            #     main_numerical_logger.info(f"The {i}-th number, the extracted number is: {number} and the idx of number in the input_ids is from {idx_start} to {idx_end}")
            #     returned_number_lists[i].append(number)
            #     returned_index_lists[i].append((idx_start, idx_end))

            # Append data to the appropriate lists
            # for i, (number, (idx_start, idx_end)) in enumerate(number_and_idx_tuple_list):
            for i, (number, idx_start, idx_end) in enumerate(number_and_idx_tuple_list):
                numerical_logger_main.info(f"The {i}-th number, the extracted number is: {number} and the idx of number in the input_ids is from {idx_start} to {idx_end}")
                returned_number_lists[i].append(number)
                returned_start_index_lists[i].append(idx_start)
                returned_end_index_lists[i].append(idx_end)
# returned_number_lists [['-2.40', '-2.10', '-2.01', '-2.41', '-2.38', '-2.36', '-2.08', '-2.10', '-2.31', '-2.44', '-2.42', '-2.02', '-2.26', '-2.38', '-2.46', '-2.22', '-2.09', '-2.47', '-2.43', ...]]
# returned_index_lists [[(...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), (...), ...]]
# len(returned_index_lists[0]) = 629
# len(returned_number_lists[0]) = 629

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        result = {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': targets_list
        }

        # num_numbers = len(number_and_idx_tuple_list) ###  这里是依赖最后一个返回值的，万一最后一轮没有提取出三个数字，就完了
        # for j in range(num_numbers):
        for j in range(number_of_numbers):
            result[f'number_{j+1}'] = returned_number_lists[j]
            result[f'number_{j+1}_number_start_idx'] = returned_start_index_lists[j]
            result[f'number_{j+1}_number_end_idx'] = returned_end_index_lists[j]

        return result

        # return dict(
        #     input_ids=input_ids_list,
        #     attention_mask=attention_mask_list,
        #     labels=targets_list,            
        # )

    def filter_empty_labels(example):
        """Remove empty labels dataset."""
        return not all(label == IGNORE_INDEX for label in example["labels"])

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        with training_args.main_process_first(desc="Train dataset tokenization"):
            train_dataset = train_dataset.shuffle().map(
                # preprocess_function,
                # lambda examples: preprocess_function(examples, custom_args.cut_off_keywords_before, custom_args.cut_off_keywords_after, main_numerical_logger),
                lambda examples: preprocess_function(examples, numerical_logger_main, custom_args.number_of_numbers),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            train_dataset = train_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
# train_dataset['number_1_number_end_idx']
# [46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, ...]
# train_dataset['number_1']
# [-2.35, -2.39, -2.2, -2.25, -2.29, -2.03, -2.23, -2.01, -2.04, -2.17, -2.1, -2.23, -2.46, -2.0, -2.04, -2.23, -2.18, -2.03, -2.23, -2.08, ...]
# train_dataset['input_ids']
# [[128000, 32, 6369, 1990, 264, 22999, 1217, 323, 459, 21075, 11478, 18328, 13, 578, 18328, 6835, 11190, 11, 11944, ...],... , [128000, 32, 6369, 1990, 264, 22999, 1217, 323, 459, 21075, 11478, 18328, 13, 578, 18328, 6835, 11190, 11, 11944, ...], ...]

            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                               for label in list(train_dataset[0]['labels'])]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

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
                # preprocess_function,
                ### enhancement
                lambda examples: preprocess_function(examples, numerical_logger_main, custom_args.number_of_numbers),
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
            model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
            training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size or 1
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        config = config_class.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        # Set RoPE scaling
        if model_args.rope_scaling is not None:
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
        if model_args.flash_attn:
            if is_flash_attn_2_available:
                config_kwargs["use_flash_attention_2"] = True
                logger.info("Using FlashAttention-2 for faster training and inference.")
            else:
                logger.warning("FlashAttention-2 is not installed.")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                           " is RTX3090, RTX4090, A100 or H100.")

        # Set shifted sparse attention (S^2-Attn)
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                apply_llama_patch()
                logger.info("Using shifted sparse attention with group_size_ratio=1/4.")
            else:
                logger.warning("Current model does not support shifted sparse attention.")

        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
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

        # model = model_class.from_pretrained(
        #     model_args.model_name_or_path,
        #     config=config,
        #     torch_dtype=torch_dtype,
        #     low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        #     device_map=model_args.device_map,
        #     **config_kwargs,
        # )

        # num_candidates = len(raw_datasets['train'][0]['conversations'][0]['candidates_sorted']) ### biubiubiu
        numerical_log_file_path = custom_args.numerical_log_file_path ### biubiubiu
        train_batch_size = training_args.train_batch_size ### biubiubiu
        model = model_class.from_pretrained(
            model_args.model_name_or_path, ### 这个位置参数要在关键字参数之前
            # num_candidates=num_candidates, ### biubiubiu
            train_batch_size = train_batch_size, ### biubiubiu
            numerical_log_file_path=numerical_log_file_path, ### biubiubiu
            # candidate_label_pad_token_id=IGNORE_INDEX_For_candidates, ### biubiubiu one forward
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=model_args.device_map,
            **config_kwargs,
        )

# ### biubiubiu
#         model.numerical_logger.info(f"have a look at trainingset examples:\n{train_dataset[0]['input_ids']}")
#         model.numerical_logger.info(f"the ids before the number is: {before_numbers_ids}")
#         model.numerical_logger.info(f"the ids after the number is: {after_numbers_ids}")
        
#         number_example_0 = extract_number_and_get_idx(train_dataset[0]["input_ids"], before_numbers_ids, after_numbers_ids)
# ### biubiubiu





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
        logger.info("Fine-tuning method: LoRA(PEFT)")

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
            logger.info("Init new peft model")
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    # Initialize our Trainer
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

    data_collator = DataCollatorForSeq2SeqNumerical( ### biubiubiu 默认是将labels对齐到最大和4倍数长度，BRIO里是将candidates的三个输入都对齐，numerical好像不需要, /data/fcl/anaconda3/envs/matllm/lib/python3.9/site-packages/transformers/data/data_collator.py
        tokenizer=tokenizer,
        # numerical_logger=model.numerical_logger, ### biubiubiu
        numerical_log_file_path=numerical_log_file_path, ### biubiubiu
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shifted sparse attention
    )
    # Initialize our Trainer
    trainer = SavePeftModelAndNumericalTrainer(
        model=model,
        # numerical_logger=model.numerical_logger, ### biubiubiu
        numerical_log_file_path=numerical_log_file_path, ### biubiubiu
        number_of_numbers = custom_args.number_of_numbers, ### biubiubiu
        args=training_args, ### 这里传入了我们启动训练命令里的参数，有train_batchsize
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        if trainer.is_world_process_zero():
            sample = next(iter(trainer.get_train_dataloader())) ### get one batch data
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(f"input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

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


# {'input_ids': tensor([[128000,     32,   6369,   1990,    264,  22999,   1217,    323,    459,
#           21075,  11478,  18328,     13,    578,  18328,   6835,  11190,     11,
#           11944,     11,    323,  48887,  11503,    311,    279,   1217,    596,
#            4860,   4005,     82,     29,   6584,     25,  20400,    264,   3284,
#           43030,    449,    264,   1515,     47,    907,    315,    482,     17,
#              13,   1187,     13,  30379,    430,    279,  43030,    596,  13340,
#            8638,    323,  10548,    449,    459,  39571,  48983,    320,   1861,
#           36660,   3931,   2891,     25,  77761,    356,  35931,     17,     10,
#              16,    356,  26176,     16,    356,    356,  26176,     16,    356,
#             356,    356,    284,  18697,     16,    356,    284,     46,    452,
#             356,    356,  19741,     39,     16,    356,    356,    356,     31,
#              39,     16,  26176,     16,    284,  18697,     16,    356,    284,
#           18697,     16,    356,    284,     46,    507,     12,     16,    507,
#           22249,     16,  26176,     17,      6, 128001, 128001],
#         [128000,     32,   6369,   1990,    264,  22999,   1217,    323,    459,
#           21075,  11478,  18328,     13,    578,  18328,   6835,  11190,     11,
#           11944,     11,    323,  48887,  11503,    311,    279,   1217,    596,
#            4860,   4005,     82,     29,   6584,     25,  20400,    264,   3284,
#           43030,    449,    264,   1515,     47,    907,    315,    482,     17,
#              13,    845,     13,  30379,    430,    279,  43030,    596,  13340,
#            8638,    323,  10548,    449,    459,  39571,  48983,    320,   1861,
#           36660,   3931,   2891,     25,  77761,    356,  19741,  26176,     16,
#             284,     45,    452,    356,    284,  18697,     16,    356,    284,
#              46,    452,    356,    356,  26176,     16,    356,    452,    284,
#              46,  26176,     16,    284,  18697,     16,    356,    284,  18697,
#              16,    356,    284,     46,    507,     12,     16,    356,  26176,
#              16,    356,    435,  26176,     16,    356,    435,    435,      6,
#          128001, 128001, 128001, 128001, 128001, 128001, 128001]],
#        device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#          0, 0, 0, 0]], device='cuda:0'), 'labels': tensor([[  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,  77761,    356,  35931,     17,     10,
#              16,    356,  26176,     16,    356,    356,  26176,     16,    356,
#             356,    356,    284,  18697,     16,    356,    284,     46,    452,
#             356,    356,  19741,     39,     16,    356,    356,    356,     31,
#              39,     16,  26176,     16,    284,  18697,     16,    356,    284,
#           18697,     16,    356,    284,     46,    507,     12,     16,    507,
#           22249,     16,  26176,     17,      6, 128001,   -100],
#         [  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#            -100,   -100,   -100,   -10