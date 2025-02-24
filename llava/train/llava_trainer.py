import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler
from torch.utils.data import ConcatDataset as TorchConcatDataset
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
import torch.distributed as dist
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from llava.utils import rank0_print

from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

if is_peft_available():
    from peft import PeftModel
def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # Label smoothing
    #     if self.args.label_smoothing_factor != 0:
    #         self.label_smoother = LabelSmootherPerSample(epsilon=self.args.label_smoothing_factor)
    #     else:
    #         self.label_smoother = None

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            if isinstance(self.train_dataset, TorchConcatDataset):
                lengths = []
                for sub_dataset in self.train_dataset.datasets:
                    lengths.extend(sub_dataset.modality_lengths)
            else:
                lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if any(module_keyword in name for module_keyword in lr_mapper)
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [
                        name for name, _ in opt_model.named_parameters() if module_keyword in name
                    ]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (n in decay_parameters and n in module_parameters and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in module_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False) or (
            hasattr(self.args, "mm_tunable_parts")
            and (
                len(self.args.mm_tunable_parts.split(",")) == 1
                and (
                    "mm_mlp_adapter" in self.args.mm_tunable_parts
                    or "mm_vision_resampler" in self.args.mm_tunable_parts
                )
            )
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     rank = dist.get_rank() if dist.is_initialized() else 0
    #     file_name = os.path.join(
    #         '/cpfs01/user/zhaoxiangyu/code_new/LLaVA/debug/llavanext/last_batches', f"last_2batch_rank_{rank}.pt"
    #     )
    #     da = torch.load(file_name)
    #     if not getattr(self, "lastbatch_flag", False):
    #         self.lastbatch_flag = True
    #         inputs = da[0]
    #         super().training_step(model, inputs)
    #     else:
    #         inputs = da[1]
    #         super().training_step(model, inputs)
    #         rank0_print("-------------------------Training step finished!!!!!!----------------------------")

    # save the error checkpoint
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     try:
    #         return super().compute_loss(model, inputs, return_outputs)
    #     except Exception as e:
    #         # 获取当前进程的 Rank
    #         rank = dist.get_rank() if dist.is_initialized() else 0
    #         print(f"Error occurred on rank {rank}: {e}")

    #         # 仅保存当前报错 Rank 的数据
    #         torch.save(inputs, f"./error_data_rank_{rank}.pt")
    #         print(f"Error data saved for rank {rank} at ./error_data_rank_{rank}.pt")
    #         if dist.is_initialized():
    #             dist.barrier()
    #         raise e

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
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
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            per_sample_loss = outputs['per_sample_losses'] if 'per_sample_losses' in outputs else None
            # rank0_print(f"per_sample_loss: {per_sample_loss}")

        # compute loss per type
        with torch.no_grad():
            if "data_type" in inputs and per_sample_loss is not None:
                data_types = inputs["data_type"]
                if len(data_types) != per_sample_loss.size(0):
                    raise ValueError("Mismatch between batch size and number of data types.")

                # Dictionary to store loss sums and counts per data type
                data_type_loss_sums = {}
                data_type_counts = {}

                # Iterate over batch samples
                for i, data_type in enumerate(data_types):
                    # data_type = data_type.item()  # Convert tensor to Python scalar
                    if data_type not in data_type_loss_sums:
                        data_type_loss_sums[data_type] = 0.0
                        data_type_counts[data_type] = 0

                    # Accumulate loss and count for each data type
                    data_type_loss_sums[data_type] += per_sample_loss[i].item()
                    data_type_counts[data_type] += 1

                # rank0_print(f"data_type_loss_sums: {data_type_loss_sums}")

                # Aggregate results across GPUs using dist
                # for data_type in data_type_loss_sums.keys():
                #     tensor_sum = torch.tensor(data_type_loss_sums[data_type], device=loss.device)
                #     tensor_count = torch.tensor(data_type_counts[data_type], device=loss.device)

                #     all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
                #     all_reduce(tensor_count, op=dist.ReduceOp.SUM)

                #     # Update with reduced values
                #     data_type_loss_sums[data_type] = tensor_sum.item()
                #     data_type_counts[data_type] = tensor_count.item()
                # rank0_print(f"data_type_loss_sums after all_reduce: {data_type_loss_sums}")

                # Prepare local dictionaries for synchronization
                local_data_type_loss_sums = {key: float(value) for key, value in data_type_loss_sums.items()}
                local_data_type_counts = {key: int(value) for key, value in data_type_counts.items()}

                # Gather all dictionaries from all GPUs
                world_size = dist.get_world_size()
                local_data_type_loss_sums_list = [None for _ in range(world_size)]
                local_data_type_counts_list = [None for _ in range(world_size)]

                dist.all_gather_object(local_data_type_loss_sums_list, local_data_type_loss_sums)
                dist.all_gather_object(local_data_type_counts_list, local_data_type_counts)

                # Merge dictionaries from all GPUs
                if dist.get_rank() == 0:
                    global_data_type_loss_sums = {}
                    global_data_type_counts = {}

                    for gpu_data_loss_sums, gpu_data_counts in zip(local_data_type_loss_sums_list, local_data_type_counts_list):
                        for data_type, loss_sum in gpu_data_loss_sums.items():
                            if data_type not in global_data_type_loss_sums:
                                global_data_type_loss_sums[data_type] = 0.0
                            global_data_type_loss_sums[data_type] += loss_sum

                        for data_type, count in gpu_data_counts.items():
                            if data_type not in global_data_type_counts:
                                global_data_type_counts[data_type] = 0
                            global_data_type_counts[data_type] += count

                    # Log average loss per data type at the end of gradient accumulation
                    if self.state.global_step % self.args.gradient_accumulation_steps == 0:
                        for data_type, loss_sum in global_data_type_loss_sums.items():
                            avg_loss = loss_sum / global_data_type_counts[data_type]
                            log_key = f"{data_type}_loss"
                            self.log(
                                {
                                    log_key: avg_loss,
                                    f"{data_type}_count": global_data_type_counts[data_type],
                                }
                            )
        return (loss, outputs) if return_outputs else loss

# from dataclasses import dataclass

# @dataclass
# class LabelSmootherPerSample:
#     """
#     Adds label-smoothing on a pre-computed output from a Transformers model.

#     Args:
#         epsilon (`float`, *optional*, defaults to 0.1):
#             The label smoothing factor.
#         ignore_index (`int`, *optional*, defaults to -100):
#             The index in the labels to ignore when computing the loss.
#     """

#     epsilon: float = 0.1
#     ignore_index: int = -100

#     def __call__(self, model_output, labels, shift_labels=False):
#         logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
#         if shift_labels:
#             logits = logits[..., :-1, :].contiguous()
#             labels = labels[..., 1:].contiguous()

#         log_probs = -nn.functional.log_softmax(logits, dim=-1)
#         if labels.dim() == log_probs.dim() - 1:
#             labels = labels.unsqueeze(-1)

#         padding_mask = labels.eq(self.ignore_index)
#         # Replace ignore_index labels with 0 to avoid gather errors
#         labels = torch.clamp(labels, min=0)

#         # Compute negative log-likelihood (NLL) loss for each token
#         nll_loss = log_probs.gather(dim=-1, index=labels).squeeze(-1)  # Shape: [batch_size, seq_len]
#         smoothed_loss = log_probs.sum(dim=-1, dtype=torch.float32)  # Shape: [batch_size, seq_len]

#         # Mask padding positions
#         nll_loss.masked_fill_(padding_mask.squeeze(-1), 0.0)
#         smoothed_loss.masked_fill_(padding_mask.squeeze(-1), 0.0)

#         # Compute per-sample loss by summing over sequence length
#         per_sample_nll_loss = nll_loss.sum(dim=-1)  # Shape: [batch_size]
#         per_sample_smoothed_loss = smoothed_loss.sum(dim=-1)  # Shape: [batch_size]

#         # Count the number of active elements for each sample
#         active_elements_per_sample = (~padding_mask).sum(dim=-1).squeeze(-1).clamp(min=1)

#         # Average losses for each sample
#         per_sample_nll_loss /= active_elements_per_sample
#         per_sample_smoothed_loss /= active_elements_per_sample

#         # Overall loss (mean across all samples)
#         nll_loss = per_sample_nll_loss.mean()
#         smoothed_loss = per_sample_smoothed_loss.mean()

#         # Combine losses using epsilon
#         loss = (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

#         # Return overall loss and per-sample losses
#         return loss, per_sample_nll_loss

from trl.trainer import DPOTrainer

class LLaVADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            # lengths = self.train_dataset.modality_lengths
            if isinstance(self.train_dataset, TorchConcatDataset):
                lengths = []
                for sub_dataset in self.train_dataset.datasets:
                    lengths.extend(sub_dataset.modality_lengths)
            else:
                lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts")
            and (
                len(self.args.mm_tunable_parts.split(",")) == 1
                and (
                    "mm_mlp_adapter" in self.args.mm_tunable_parts
                    or "mm_vision_resampler" in self.args.mm_tunable_parts
                )
            )
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            # super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)
            # print(type(model))
            # from transformers.modeling_utils import unwrap_model
            # print(type(unwrap_model(model)))
            # print(unwrap_model(model).config)
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)
