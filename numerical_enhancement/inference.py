# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import json
import os
from threading import Thread

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
    BitsAndBytesConfig,
)

from template import get_conv_template



































# ### numerical generation
# from transformers.generation.utils import GenerationMixin
# from transformers.utils import PushToHubMixin
# from transformers.modeling_utils import PreTrainedModel, ModuleUtilsMixin
# from transformers.integrations import PeftAdapterMixin

# # from transformers.generation.logits_process import (
# #     EncoderNoRepeatNGramLogitsProcessor,
# #     EncoderRepetitionPenaltyLogitsProcessor,
# #     EpsilonLogitsWarper,
# #     EtaLogitsWarper,
# #     ExponentialDecayLengthPenalty,
# #     ForcedBOSTokenLogitsProcessor,
# #     ForcedEOSTokenLogitsProcessor,
# #     ForceTokensLogitsProcessor,
# #     HammingDiversityLogitsProcessor,
# #     InfNanRemoveLogitsProcessor,
# #     LogitNormalization,
# #     LogitsProcessorList,
# #     MinLengthLogitsProcessor,
# #     MinNewTokensLengthLogitsProcessor,
# #     NoBadWordsLogitsProcessor,
# #     NoRepeatNGramLogitsProcessor,
# #     PrefixConstrainedLogitsProcessor,
# #     RepetitionPenaltyLogitsProcessor,
# #     SequenceBiasLogitsProcessor,
# #     SuppressTokensAtBeginLogitsProcessor,
# #     SuppressTokensLogitsProcessor,
# #     TemperatureLogitsWarper,
# #     TopKLogitsWarper,
# #     TopPLogitsWarper,
# #     TypicalLogitsWarper,
# #     UnbatchedClassifierFreeGuidanceLogitsProcessor,
# # )
# # from transformers.generation.stopping_criteria import (
# #     MaxLengthCriteria,
# #     MaxTimeCriteria,
# #     StoppingCriteria,
# #     StoppingCriteriaList,
# #     validate_stopping_criteria,
# # )
# # if TYPE_CHECKING:
# #     # from ..modeling_utils import PreTrainedModel
# #     from transformers.generation.streamers import BaseStreamer

# # NEED_SETUP_CACHE_CLASSES_MAPPING = {
# #     "static": StaticCache,
# # }

# # from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
# # import inspect
# # logger = logging.get_logger(__name__)
# # from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
# # from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint


# import copy
# import inspect
# import warnings
# from dataclasses import dataclass
# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# import torch
# import torch.distributed as dist
# from torch import nn

# from transformers.cache_utils import Cache, DynamicCache, StaticCache
# from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
# from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
# from transformers.models.auto import (
#     MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
#     MODEL_FOR_CAUSAL_LM_MAPPING,
#     MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
#     MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
#     MODEL_FOR_VISION_2_SEQ_MAPPING,
# )
# from transformers.utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
# from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
# from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
# from transformers.generation.candidate_generator import (
#     AssistedCandidateGenerator,
#     CandidateGenerator,
#     PromptLookupCandidateGenerator,
#     _crop_past_key_values,
#     _prepare_attention_mask,
#     _prepare_token_type_ids,
# )
# from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
# from transformers.generation.logits_process import (
#     EncoderNoRepeatNGramLogitsProcessor,
#     EncoderRepetitionPenaltyLogitsProcessor,
#     EpsilonLogitsWarper,
#     EtaLogitsWarper,
#     ExponentialDecayLengthPenalty,
#     ForcedBOSTokenLogitsProcessor,
#     ForcedEOSTokenLogitsProcessor,
#     ForceTokensLogitsProcessor,
#     HammingDiversityLogitsProcessor,
#     InfNanRemoveLogitsProcessor,
#     LogitNormalization,
#     LogitsProcessorList,
#     MinLengthLogitsProcessor,
#     MinNewTokensLengthLogitsProcessor,
#     NoBadWordsLogitsProcessor,
#     NoRepeatNGramLogitsProcessor,
#     PrefixConstrainedLogitsProcessor,
#     RepetitionPenaltyLogitsProcessor,
#     SequenceBiasLogitsProcessor,
#     SuppressTokensAtBeginLogitsProcessor,
#     SuppressTokensLogitsProcessor,
#     TemperatureLogitsWarper,
#     TopKLogitsWarper,
#     TopPLogitsWarper,
#     TypicalLogitsWarper,
#     UnbatchedClassifierFreeGuidanceLogitsProcessor,
# )
# from transformers.generation.stopping_criteria import (
#     MaxLengthCriteria,
#     MaxTimeCriteria,
#     StoppingCriteria,
#     StoppingCriteriaList,
#     validate_stopping_criteria,
# )


# if TYPE_CHECKING:
#     from transformers.modeling_utils import PreTrainedModel
#     from transformers.generation.streamers import BaseStreamer

# logger = logging.get_logger(__name__)

# if is_accelerate_available():
#     from accelerate.hooks import AlignDevicesHook, add_hook_to_module

# NEED_SETUP_CACHE_CLASSES_MAPPING = {
#     "static": StaticCache,
# }


# @dataclass
# class GenerateDecoderOnlyOutput(ModelOutput):
#     """
#     Outputs of decoder-only generation models, when using non-beam methods.

#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
#         logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
#             Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
#         attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
#         past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
#             Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
#             tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
#             `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
#             encoder_sequence_length, embed_size_per_head)`.
#     """

#     sequences: torch.LongTensor = None
#     scores: Optional[Tuple[torch.FloatTensor]] = None
#     logits: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
#     hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
#     past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None

# GenerateOutput = Union[GenerateDecoderOnlyOutput]

# class GenerationMixinNE(GenerationMixin):
#     # def __init__(self):
#     #     super().__init__()
#     #     # 在这里可以添加任何自定义初始化逻辑

#     # 重写，覆盖需要的方法
#     @torch.no_grad()
#     def generate(
#         self,
#         inputs: Optional[torch.Tensor] = None,
# #####################
#         numbers: Optional[torch.Tensor] = None, ### biubiubiu
#         numbers_start_idx: Optional[torch.Tensor] = None, ### biubiubiu
#         numbers_end_idx: Optional[torch.Tensor] = None, ### biubiubiu
# #####################
#         generation_config: Optional[GenerationConfig] = None,
#         logits_processor: Optional[LogitsProcessorList] = None,
#         stopping_criteria: Optional[StoppingCriteriaList] = None,
#         prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
#         synced_gpus: Optional[bool] = None,
#         assistant_model: Optional["PreTrainedModel"] = None,
#         streamer: Optional["BaseStreamer"] = None,
#         negative_prompt_ids: Optional[torch.Tensor] = None,
#         negative_prompt_attention_mask: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#         r"""

#         Generates sequences of token ids for models with a language modeling head.

#         <Tip warning={true}>

#         Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
#         model's default generation configuration. You can override any `generation_config` by passing the corresponding
#         parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

#         For an overview of generation strategies and code examples, check out the [following
#         guide](transformers./generation_strategies).

#         </Tip>

#         Parameters:
#             inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
#                 The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
#                 method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
#                 should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
#                 `input_ids`, `input_values`, `input_features`, or `pixel_values`.
#             generation_config (`~generation.GenerationConfig`, *optional*):
#                 The generation configuration to be used as base parametrization for the generation call. `**kwargs`
#                 passed to generate matching the attributes of `generation_config` will override them. If
#                 `generation_config` is not provided, the default will be used, which has the following loading
#                 priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
#                 configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
#                 default values, whose documentation should be checked to parameterize generation.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 Custom logits processors that complement the default logits processors built from arguments and
#                 generation config. If a logit processor is passed that is already created with the arguments or a
#                 generation config an error is thrown. This feature is intended for advanced users.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 Custom stopping criteria that complements the default stopping criteria built from arguments and a
#                 generation config. If a stopping criteria is passed that is already created with the arguments or a
#                 generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
#                 sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
#                 intended for advanced users.
#             prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
#                 If provided, this function constraints the beam search to allowed tokens only at each step. If not
#                 provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
#                 `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
#                 on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
#                 for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
#                 Retrieval](https://arxiv.org/abs/2010.00904).
#             synced_gpus (`bool`, *optional*):
#                 Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
#                 `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
#                 generating before other GPUs. Otherwise it'll be set to `False`.
#             assistant_model (`PreTrainedModel`, *optional*):
#                 An assistant model that can be used to accelerate generation. The assistant model must have the exact
#                 same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
#                 is much faster than running generation with the model you're calling generate from. As such, the
#                 assistant model should be much smaller.
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 The negative prompt needed for some processors such as CFG. The batch size must match the input batch
#                 size. This is an experimental feature, subject to breaking API changes in future versions.
#             negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Attention_mask for `negative_prompt_ids`.
#             kwargs (`Dict[str, Any]`, *optional*):
#                 Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
#                 forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
#                 specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

#         Return:
#             [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
#             or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

#                 If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
#                 [`~utils.ModelOutput`] types are:

#                     - [`~generation.GenerateDecoderOnlyOutput`],
#                     - [`~generation.GenerateBeamDecoderOnlyOutput`]

#                 If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
#                 [`~utils.ModelOutput`] types are:

#                     - [`~generation.GenerateEncoderDecoderOutput`],
#                     - [`~generation.GenerateBeamEncoderDecoderOutput`]
#         """
#         # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
#         self._validate_model_class()
#         generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
#         self._validate_model_kwargs(model_kwargs.copy())

#         # 2. Set generation parameters if not already defined
#         if synced_gpus is None:
#             if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
#                 synced_gpus = True
#             else:
#                 synced_gpus = False
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

#         if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
#             if model_kwargs.get("attention_mask", None) is None:
#                 logger.warning(
#                     "The attention mask and the pad token id were not set. As a consequence, you may observe "
#                     "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
#                 )
#             eos_token_id = generation_config.eos_token_id
#             if isinstance(eos_token_id, list):
#                 eos_token_id = eos_token_id[0]
#             logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
#             generation_config.pad_token_id = eos_token_id

#         # 3. Define model inputs
#         # inputs_tensor has to be defined
#         # model_input_name is defined if model-specific keyword input is passed
#         # otherwise model_input_name is None
#         # all model-specific keyword inputs are removed from `model_kwargs`
#         inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
#             inputs, generation_config.bos_token_id, model_kwargs
#         )
#         batch_size = inputs_tensor.shape[0]

#         # 4. Define other model kwargs
#         model_kwargs["output_attentions"] = generation_config.output_attentions
#         model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
#         # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
#         # generating the first new token or not, and we only want to use the embeddings for the first new token)
#         if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
#             model_kwargs["use_cache"] = True
#         else:
#             model_kwargs["use_cache"] = generation_config.use_cache

#         accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
#         requires_attention_mask = "encoder_outputs" not in model_kwargs

#         if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
#             model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
#                 inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
#             )

#         # decoder-only models should use left-padding for generation
#         if not self.config.is_encoder_decoder:
#             # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
#             # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
#             if (
#                 generation_config.pad_token_id is not None
#                 and len(inputs_tensor.shape) == 2
#                 and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
#             ):
#                 logger.warning(
#                     "A decoder-only architecture is being used, but right-padding was detected! For correct "
#                     "generation results, please set `padding_side='left'` when initializing the tokenizer."
#                 )

#         if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
#             # if model is encoder decoder encoder_outputs are created
#             # and added to `model_kwargs`
#             model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
#                 inputs_tensor, model_kwargs, model_input_name
#             )

#         # 5. Prepare `input_ids` which will be used for auto-regressive generation
#         if self.config.is_encoder_decoder:
#             input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
#                 batch_size=batch_size,
#                 model_input_name=model_input_name,
#                 model_kwargs=model_kwargs,
#                 decoder_start_token_id=generation_config.decoder_start_token_id,
#                 bos_token_id=generation_config.bos_token_id,
#                 device=inputs_tensor.device,
#             )
#         else:
#             input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

#         if streamer is not None:
#             streamer.put(input_ids.cpu())

#         # 6. Prepare `max_length` depending on other stopping criteria.
#         input_ids_length = input_ids.shape[-1]
#         has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
#         if generation_config.max_new_tokens is not None:
#             if not has_default_max_length and generation_config.max_length is not None:
#                 logger.warning(
#                     f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
#                     f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
#                     "Please refer to the documentation for more information. "
#                     "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
#                 )
#             generation_config.max_length = generation_config.max_new_tokens + input_ids_length

#         # otherwise the total length [inputs-embeds-len + new-tokens-len] will go beyond indicated `max_length``
#         elif (
#             model_input_name == "inputs_embeds"
#             and inputs_tensor.shape[:-1] != input_ids.shape
#             and not self.config.is_encoder_decoder
#         ):
#             generation_config.max_length -= inputs_tensor.shape[1]
#             generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

#         if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
#             if generation_config.cache_implementation == "static":
#                 if model_kwargs.get("past_key_values", False) is not False:
#                     raise ValueError(
#                         "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
#                     )
#                 cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
#                 if not callable(getattr(self, "_setup_cache", None)):
#                     raise ValueError(
#                         "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
#                         " Make sure it has a `_setup_cache` function."
#                     )
#                 self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

#         self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

#         # 7. determine generation mode
#         generation_mode = generation_config.get_generation_mode(assistant_model)

#         if streamer is not None and (generation_config.num_beams > 1):
#             raise ValueError(
#                 "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
#             )

#         if self.device.type != input_ids.device.type:
#             warnings.warn(
#                 "You are calling .generate() with the `input_ids` being on a device type different"
#                 f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
#                 f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
#                 " Please make sure that you have put `input_ids` to the"
#                 f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
#                 " running `.generate()`.",
#                 UserWarning,
#             )

#         # 8. prepare distribution pre_processing samplers
#         prepared_logits_processor = self._get_logits_processor(
#             generation_config=generation_config,
#             input_ids_seq_length=input_ids_length,
#             encoder_input_ids=inputs_tensor,
#             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#             logits_processor=logits_processor,
#             model_kwargs=model_kwargs,
#             negative_prompt_ids=negative_prompt_ids,
#             negative_prompt_attention_mask=negative_prompt_attention_mask,
#         )

#         # 9. prepare stopping criteria
#         prepared_stopping_criteria = self._get_stopping_criteria(
#             generation_config=generation_config, stopping_criteria=stopping_criteria
#         )
#         # 10. go into different generation modes
#         if generation_mode == GenerationMode.ASSISTED_GENERATION:
#             if generation_config.num_return_sequences > 1:
#                 raise ValueError(
#                     "num_return_sequences has to be 1 when doing assisted generate, "
#                     f"but is {generation_config.num_return_sequences}."
#                 )
#             if batch_size > 1:
#                 raise ValueError("assisted generate is only supported for batch_size = 1")
#             if not model_kwargs["use_cache"]:
#                 raise ValueError("assisted generate requires `use_cache=True`")

#             # 11. Get the candidate generator, given the parameterization
#             candidate_generator = self._get_candidate_generator(
#                 generation_config=generation_config,
#                 input_ids=input_ids,
#                 inputs_tensor=inputs_tensor,
#                 assistant_model=assistant_model,
#                 logits_processor=logits_processor,
#                 model_kwargs=model_kwargs,
#             )

#             # 12. run assisted generate
#             result = self.assisted_decoding(
#                 input_ids,
#                 candidate_generator=candidate_generator,
#                 do_sample=generation_config.do_sample,
#                 logits_processor=prepared_logits_processor,
#                 logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 streamer=streamer,
#                 **model_kwargs,
#             )
#         if generation_mode == GenerationMode.GREEDY_SEARCH:
#             # 11. run greedy search
#             result = self._greedy_search(
#                 input_ids,
#                 logits_processor=prepared_logits_processor,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 streamer=streamer,
#                 **model_kwargs,
#             )

#         elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
#             if not model_kwargs["use_cache"]:
#                 raise ValueError("Contrastive search requires `use_cache=True`")

#             result = self._contrastive_search(
#                 input_ids,
#                 top_k=generation_config.top_k,
#                 penalty_alpha=generation_config.penalty_alpha,
#                 logits_processor=prepared_logits_processor,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 streamer=streamer,
#                 sequential=generation_config.low_memory,
#                 **model_kwargs,
#             )

#         elif generation_mode == GenerationMode.SAMPLE:
#             # 11. prepare logits warper
#             logits_warper = self._get_logits_warper(generation_config)

#             # 12. expand input_ids with `num_return_sequences` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_return_sequences,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )

#             # 13. run sample
#             result = self._sample(
#                 input_ids,
#                 logits_processor=prepared_logits_processor,
#                 logits_warper=logits_warper,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 streamer=streamer,
#                 **model_kwargs,
#             )

#         elif generation_mode == GenerationMode.BEAM_SEARCH:
#             # 11. prepare beam search scorer
#             beam_scorer = BeamSearchScorer(
#                 batch_size=batch_size,
#                 num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                 max_length=generation_config.max_length,
#             )
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )
#             # 13. run beam search
#             result = self._beam_search(
#                 input_ids,
#                 beam_scorer,
#                 logits_processor=prepared_logits_processor,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 sequential=generation_config.low_memory,
#                 **model_kwargs,
#             )

#         elif generation_mode == GenerationMode.BEAM_SAMPLE:
#             # 11. prepare logits warper
#             logits_warper = self._get_logits_warper(generation_config)

#             # 12. prepare beam search scorer
#             beam_scorer = BeamSearchScorer(
#                 batch_size=batch_size,
#                 num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                 max_length=generation_config.max_length,
#             )

#             # 13. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )

#             # 14. run beam sample
#             result = self._beam_sample(
#                 input_ids,
#                 beam_scorer,
#                 logits_processor=prepared_logits_processor,
#                 logits_warper=logits_warper,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 **model_kwargs,
#             )

#         elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
#             # 11. prepare beam search scorer
#             beam_scorer = BeamSearchScorer(
#                 batch_size=batch_size,
#                 num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                 num_beam_groups=generation_config.num_beam_groups,
#                 max_length=generation_config.max_length,
#             )
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )
#             # 13. run beam search
#             result = self._group_beam_search(
#                 input_ids,
#                 beam_scorer,
#                 logits_processor=prepared_logits_processor,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 **model_kwargs,
#             )

#         elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
#             final_constraints = []
#             if generation_config.constraints is not None:
#                 final_constraints = generation_config.constraints

#             if generation_config.force_words_ids is not None:

#                 def typeerror():
#                     raise ValueError(
#                         "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
#                         f"of positive integers, but is {generation_config.force_words_ids}."
#                     )

#                 if (
#                     not isinstance(generation_config.force_words_ids, list)
#                     or len(generation_config.force_words_ids) == 0
#                 ):
#                     typeerror()

#                 for word_ids in generation_config.force_words_ids:
#                     if isinstance(word_ids[0], list):
#                         if not isinstance(word_ids, list) or len(word_ids) == 0:
#                             typeerror()
#                         if any(not isinstance(token_ids, list) for token_ids in word_ids):
#                             typeerror()
#                         if any(
#                             any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
#                             for token_ids in word_ids
#                         ):
#                             typeerror()

#                         constraint = DisjunctiveConstraint(word_ids)
#                     else:
#                         if not isinstance(word_ids, list) or len(word_ids) == 0:
#                             typeerror()
#                         if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
#                             typeerror()

#                         constraint = PhrasalConstraint(word_ids)
#                     final_constraints.append(constraint)

#             # 11. prepare beam search scorer
#             constrained_beam_scorer = ConstrainedBeamSearchScorer(
#                 constraints=final_constraints,
#                 batch_size=batch_size,
#                 num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                 max_length=generation_config.max_length,
#             )
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )
#             # 13. run beam search
#             result = self._constrained_beam_search(
#                 input_ids,
#                 constrained_beam_scorer=constrained_beam_scorer,
#                 logits_processor=prepared_logits_processor,
#                 stopping_criteria=prepared_stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 output_logits=generation_config.output_logits,
#                 return_dict_in_generate=generation_config.return_dict_in_generate,
#                 synced_gpus=synced_gpus,
#                 **model_kwargs,
#             )

#         if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
#             if not callable(getattr(self, "_reset_cache", None)):
#                 raise ValueError(
#                     "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
#                     " Make sure this model implements a `_reset_cache` function."
#                 )
#             self._reset_cache()

#         return result

# class PreTrainedModelNE(PreTrainedModel):
#     def __init__(self, *args, **kwargs):
#         pass
#         # super().__init__(*args, **kwargs)
    
#     @classmethod
#     def can_generate(cls) -> bool:
#         """
#         Returns whether this model can generate sequences with `.generate()`.

#         Returns:
#             `bool`: Whether this model can generate sequences with `.generate()`.
#         """
#         # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
#         # Alternativelly, the model can also have a custom `generate` function.
#         if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
#             return False
#         return True

# class LlamaPreTrainedModelNE(PreTrainedModelNE):
#     def __init__(self, config):
#         super().__init__(config) # ?


# class LlamaModelNE(LlamaPreTrainedModelNE):
#     def __init__(self, config):
#         super().__init__(config)

# class LlamaForCausalLMNE(LlamaPreTrainedModelNE):
#     def __init__(self, config):
#         super().__init__(config)
# ### numerical generation












### numerical
import re
import torch.nn as nn
import logging
from logging import FileHandler
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from packaging import version
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

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
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # input_ids.shape    torch.Size([2, 124])
        # input_ids.device    device(type='cuda', index=0)
        # inputs_embeds.shape    torch.Size([2, 124, 4096])
        # inputs_embeds.device    device(type='cuda', index=0)

### numerical ******
# numbers
# tensor([[-2.5000],
#         [-2.5000]])
# numbers_start_idx
# tensor([[64],
#         [64]])
# numbers_end_idx
# tensor([[67],
#         [67]])
# input_ids
# tensor([[128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001,
#          128001, 128000,     32,   6369,   1990,    264,  22999,   1217,    323,
#             459,  21075,  11478,  18328,     13,    578,  18328,   6835,  11190,
#              11,  11944,     11,    323,  48887,  11503,    311,    279,   1217,
#             596,   4860,   4005,     82,     29,   6584,     25,  20400,    264,
#            3284,  43030,  17502,  15790,   6754,     67,   6907,  30899,    713,
#             342,  14736,  37990,  70391,    449,    264,   1515,     47,    907,
#             315,    482,     17,     13,   1135,     13,  30379,    430,    279,
#           43030,    596,  13340,   8638,    323,  10548,    449,    459,  39571,
#           48983,    320,   1861,  36660,   3931,   2891,     25],
#         [128000,     32,   6369,   1990,    264,  22999,   1217,    323,    459,
#           21075,  11478,  18328,     13,    578,  18328,   6835,  11190,     11,
#           11944,     11,    323,  48887,  11503,    311,    279,   1217,    596,
#            4860,   4005,     82,     29,   6584,     25,  20400,    264,   3284,
#            6754,  42908,   3067,    342,  14736,    264,  20221,  12703,    274,
#             784,   6907,    274,    784,    784,  26877,  35478,   2781,  91117,
#             737,  21066,     72,  43030,    449,    264,   1515,     47,    907,
#             315,    482,     17,     13,   1135,     13,  30379,    430,    279,
#           43030,    596,  13340,   8638,    323,  10548,    449,    459,  39571,
#           48983,    320,   1861,  36660,   3931,   2891,     25]],
#        device='cuda:0')
        # 首先拷贝一个和inputs_embeds形状一样的全零张量enhance_embeds（维度为torch.Size([2, 124, 4096])）
        zero_tensor = torch.zeros_like(inputs_embeds)

        # def numerical_embed(numbers, numbers_start_idx, numbers_end_idx, zero_tensor):
        #     batch_size, seq_len, hidden_dim = zero_tensor.shape
        #     enhance_embeds = zero_tensor.clone()
        #     for i in range(batch_size):
        #         start_idx = numbers_start_idx[i].item()
        #         end_idx = numbers_end_idx[i].item()
        #         number = numbers[i].item()
        #         slice_tensor = enhance_embeds[i, start_idx:end_idx + 1, :]
        #         # embedding
        #         div_term = 10000 ** (torch.arange(0, hidden_dim, 2) / hidden_dim)
        #         sin_vals = torch.sin(number / div_term)
        #         cos_vals = torch.cos(number / div_term)
                
        #         for j in range(slice_tensor.shape[0]):
        #             slice_tensor[j, 0::2] = sin_vals
        #             slice_tensor[j, 1::2] = cos_vals
                
        #         enhance_embeds[i, start_idx:end_idx + 1, :] = slice_tensor
        #     return enhance_embeds

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

        # 将全零张量和numbers_start_idx（维度为torch.Size([2, 1])）和numbers_end_idx（维度为torch.Size([2, 1])）和numbers（维度为torch.Size([2, 1])）一起送入函数numerical_embed()
        enhance_embeds = numerical_embed(numbers, numbers_start_idx, numbers_end_idx, zero_tensor)

        # 函数内部，将传入的这个全零张量切片enhance_embeds[:,nummbers_start_idx:numbers_end_idx+1,:]
        # 这个切片出来的部分维度为torch.Size([2, 4, 4096])，我们针对这部分做处理
        # 首先根据第一维度batch，取numbers的对应该batch的值number和切片张量的对应该batch的部分，再根据切片张量的最后一个维度（隐藏层维度d）计算我想要的embedding值
        # 计算公式如下，可以根据number和隐藏层位置，得到隐藏层维度不同位置的embedding值
# enhance_embeds[:,numbers_start_idx[0]-1:numbers_end_idx[0]+2,:]
# tensor([[[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00]],

#         [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [-5.9863e-01, -8.0127e-01, -6.0742e-01,  ...,  1.0000e+00,
#           -2.5105e-04,  1.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
#            0.0000e+00,  0.0000e+00]]], device='cuda:0', dtype=torch.float16)
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
    # def __init__(self, config, train_batch_size, numerical_log_file_path):
    def __init__(self, config):
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
        # 换爷爷了
        # LlamaPreTrainedModelNE.__init__(self, config) # 因此，手动初始化父类的父类
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
        # self.train_batch_size = train_batch_size
        # self._setup_numerical_logger(numerical_log_file_path)
        ######

    # def _setup_numerical_logger(self, numerical_log_file_path):
    #     # 创建输出到文件的 logger
    #     file_handler = FileHandler(numerical_log_file_path)
    #     file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     file_handler.setFormatter(file_formatter)
    #     self.numerical_logger_model = logging.getLogger('numerical_logger_model')
    #     self.numerical_logger_model.addHandler(file_handler)
    #     self.numerical_logger_model.setLevel(logging.DEBUG)
    #     self.numerical_logger_model.info(f"numerical_logger_model has been setup.")

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
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLMNumerical, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        do_print=True,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        context_len=2048,
        stop_str="</s>",
):
    """Generate answer from prompt with GPT and stream the output"""
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        stop = False
        pos = new_text.find(stop_str)
        if pos != -1:
            new_text = new_text[:pos]
            stop = True
        generated_text += new_text
        if do_print:
            print(new_text, end="", flush=True)
        if stop:
            break
    if do_print:
        print()
    return generated_text

def extract_number_and_idx(tokenizer, input_ids):
    decoded_text = tokenizer.decode(input_ids)
    number_pattern = re.compile(r'-?\d+\.\d+')
    matches = list(number_pattern.finditer(decoded_text))

    def find_sublist_indexes(lst, sublst):
        for i in range(len(lst) - len(sublst) + 1):
            if lst[i:i+len(sublst)] == sublst:
                print("yes, i find it")
                return i, i + len(sublst) - 1
        print("no, i didn't find it")
        return None, None

    results = []

    for match in matches:
        new_string = ' ' + match.group()
        number_token_ids = tokenizer.encode(new_string, add_special_tokens=False)
        start_idx, end_idx = find_sublist_indexes(input_ids, number_token_ids)
        # results.append((match.group(), (start_idx, end_idx)))
        results.append((float(match.group()), start_idx, end_idx))
    # return torch.tensor(float(match.group())), torch.tensor(start_idx), torch.tensor(end_idx)
    # return float(match.group()), start_idx, end_idx
    return results



@torch.inference_mode()
def batch_generate_answer(
        sentences,
        model,
        tokenizer,
        prompt_template,
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
        number_of_numbers=1,
):
    """Generate answer from prompt with GPT, batch mode"""
    generated_texts = []
### numerical
    returned_number_lists = []
    returned_start_index_lists = []
    returned_end_index_lists = []
    for _ in range(number_of_numbers):
        returned_number_lists.append([])
        returned_start_index_lists.append([])
        returned_end_index_lists.append([])
### numerical
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
    )
    prompts = [prompt_template.get_prompt(messages=[[s, '']]) for s in sentences]
    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True) ### 补齐了，因此位置一致是一样的哈哈哈
    input_ids = inputs_tokens['input_ids'].to(device)
# input_ids.shape
# torch.Size([2, 67])
# input_ids.device
# device(type='cuda', index=0)

### numerical
    for i in range(len(input_ids)): # 逐batch
    # 该函数目标 得到三个 torch.Size([2, 1])
        number_and_idx_tuple_list = extract_number_and_idx(tokenizer, input_ids[i].tolist())
        if len(number_and_idx_tuple_list) == number_of_numbers:
            for i, (number, idx_start, idx_end) in enumerate(number_and_idx_tuple_list):
                print(f"The {i}-th number, the extracted number is: {number} and the idx of number in the input_ids is from {idx_start} to {idx_end}")
                returned_number_lists[i].append(number)
                returned_start_index_lists[i].append(idx_start)
                returned_end_index_lists[i].append(idx_end)
# returned_number_lists
# [[-2.5, -2.5]]
# returned_start_index_lists
# [[64, 64]]
# returned_end_index_lists
# [[67, 67]]
    for j in range(number_of_numbers):
        inputs_tokens[f'number_{j+1}'] = torch.tensor(returned_number_lists[j])
        inputs_tokens[f'number_{j+1}_number_start_idx'] = torch.tensor(returned_start_index_lists[j])
        inputs_tokens[f'number_{j+1}_number_end_idx'] = torch.tensor(returned_end_index_lists[j])
# inputs_tokens['number_1']
# tensor([-2.5000, -2.5000])
# inputs_tokens['number_1_number_start_idx']
# tensor([64, 64])
# inputs_tokens['number_1_number_end_idx']
# tensor([67, 67])
# {'input_ids': tensor([[128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001,
#          128001, 128000,     32,   6369,   1990,    264,  22999,   1217,    323,
#             459,  21075,  11478,  18328,     13,    578,  18328,   6835,  11190,
#              11,  11944,     11,    323,  48887,  11503,    311,    279,   1217,
#             596,   4860,   4005,     82,     29,   6584,     25,  20400,    264,
#            3284,  43030,  17502,  15790,   6754,     67,   6907,  30899,    713,
#             342,  14736,  37990,  70391,    449,    264,   1515,     47,    907,
#             315,    482,     17,     13,   1135,     13,  30379,    430,    279,
#           43030,    596,  13340,   8638,    323,  10548,    449,    459,  39571,
#           48983,    320,   1861,  36660,   3931,   2891,     25],
#         [128000,     32,   6369,   1990,    264,  22999,   1217,    323,    459,
#           21075,  11478,  18328,     13,    578,  18328,   6835,  11190,     11,
#           11944,     11,    323,  48887,  11503,    311,    279,   1217,    596,
#            4860,   4005,     82,     29,   6584,     25,  20400,    264,   3284,
#            6754,  42908,   3067,    342,  14736,    264,  20221,  12703,    274,
#             784,   6907,    274,    784,    784,  26877,  35478,   2781,  91117,
#             737,  21066,     72,  43030,    449,    264,   1515,     47,    907,
#             315,    482,     17,     13,   1135,     13,  30379,    430,    279,
#           43030,    596,  13340,   8638,    323,  10548,    449,    459,  39571,
#           48983,    320,   1861,  36660,   3931,   2891,     25]]), 'attention_mask': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'number_1': tensor([-2.5000, -2.5000]), 'number_1_number_start_idx': tensor([64, 64]), 'number_1_number_end_idx': tensor([67, 67])}
# inputs_tokens['number_1_number_start_idx'].shape
# torch.Size([2])
# inputs_tokens['number_1_number_end_idx'].shape
# torch.Size([2])
# inputs_tokens['number_1'].shape
# torch.Size([2])
# *************************
    numbers = []
    numbers_start_idx = []
    numbers_end_idx = []
    for i in range(number_of_numbers):            
        numbers.append(inputs_tokens.pop(f'number_{i+1}').unsqueeze(1))
        numbers_start_idx.append(inputs_tokens.pop(f'number_{i+1}_number_start_idx').unsqueeze(1))
        numbers_end_idx.append(inputs_tokens.pop(f'number_{i+1}_number_end_idx').unsqueeze(1))
    # inputs.pop(f'number_{i+1}_number_end_idx')  -->  tensor([46, 46], device='cuda:0')
    # inputs.pop(f'number_{i+1}_number_start_idx').unsqueeze(1) -->  
    # tensor([[43],
    #         [43]], device='cuda:0')
    inputs_tokens['numbers'] = torch.cat(numbers, dim=1)
    inputs_tokens['numbers_start_idx'] = torch.cat(numbers_start_idx, dim=1)
    inputs_tokens['numbers_end_idx'] = torch.cat(numbers_end_idx, dim=1)
# inputs_tokens['numbers'].shape
# torch.Size([2, 1])
# inputs_tokens['numbers_start_idx'].shape
# torch.Size([2, 1])
# inputs_tokens['numbers_end_idx'].shape
# torch.Size([2, 1])
# TODO: 多个数字可视化
# *********************************        
### numerical

    outputs = model.generate(input_ids=input_ids, numbers=inputs_tokens['numbers'], numbers_start_idx=inputs_tokens['numbers_start_idx'], numbers_end_idx=inputs_tokens['numbers_end_idx'], **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=True)
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

    return generated_texts


def main():
    parser = argparse.ArgumentParser()
    ### biubiubiu
    parser.add_argument('--number_of_numbers', default=1, type=int)
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (default multi-turn)")
    parser.add_argument('--single_tune', action='store_true', help='Whether to use single-tune model')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit')
    args = parser.parse_args()
    print(args)
    load_type = torch.float16
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    ###add_begin###
    tokenizer.pad_token = tokenizer.eos_token
    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_type,
        "low_cpu_mem_usage": True,
        "device_map": 'auto',
    }
    if args.load_in_8bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_type,
        )
#####################################
    base_model = model_class.from_pretrained(args.base_model, **config_kwargs) ### stepp # fcl step # fcl
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model ### here
    model.eval()
    print(tokenizer)
    # test data
    if args.data_file is None:
        examples = ["介绍下北京", "乙肝和丙肝的区别？"]
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)

    # Chat
    prompt_template = get_conv_template(args.template_name)
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    if args.interactive:
        print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
        history = []
        while True:
            try:
                query = input(f"{prompt_template.roles[0]}: ")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please try again.")
                continue
            except Exception:
                raise
            if query == "":
                print("Please input text, try again.")
                continue
            if query.strip() == "exit":
                print("exit...")
                break
            if query.strip() == "clear":
                history = []
                print("history cleared.")
                continue

            print(f"{prompt_template.roles[1]}: ", end="", flush=True)
            if args.single_tune:
                history = []

            history.append([query, ''])
            prompt = prompt_template.get_prompt(messages=history)
            response = stream_generate_answer(
                model,
                tokenizer,
                prompt,
                model.device,
                do_print=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            if history:
                history[-1][-1] = response.strip()
    else:
        print("Start inference.")
        counts = 0
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        eval_batch_size = args.eval_batch_size
        for batch in tqdm(
                [
                    examples[i: i + eval_batch_size]
                    for i in range(0, len(examples), eval_batch_size)
                ],
                desc="Generating outputs",
        ):
            responses = batch_generate_answer(
                batch,
                model,
                tokenizer,
                prompt_template,
                model.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
                number_of_numbers=args.number_of_numbers,
            )
            results = []
            for example, response in zip(batch, responses):
                print(f"===")
                print(f"Input: {example}")
                print(f"Output: {response}\n")
                results.append({"Input": example, "Output": response})
                counts += 1
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for entry in results:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
        print(f'save to {args.output_file}, size: {counts}')


if __name__ == '__main__':
    main()
