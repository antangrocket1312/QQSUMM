# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import errno
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import transformers
from transformers import BitsAndBytesConfig

import src.fid
from src import dist_utils
from src.atlas import Atlas
from src.retrievers import Contriever, DualEncoderRetriever, UntiedDualEncoderRetriever
from src.util import cast_to_precision, set_dropout, set_optim
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

Number = Union[float, int]

logger = logging.getLogger(__name__)


def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    return checkpoint_path


def create_checkpoint_directories(opt):
    checkpoint_path = get_checkpoint_path(opt)
    os.makedirs(checkpoint_path, exist_ok=True)
    if opt.save_index_path:
        os.makedirs(opt.save_index_path, exist_ok=True)
    dist_utils.barrier()
    return checkpoint_path, opt.save_index_path


def load_retriever(opt, opt_checkpoint=None):
    if opt.use_file_passages:
        return None, None
    print("retriever_model_path: ", opt.retriever_model_path)
    contriever_encoder = Contriever.from_pretrained(opt.retriever_model_path)
    retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(opt.retriever_model_path)

    # once you have done query side training you cannot go back to a parameter-tied retriever
    if opt_checkpoint is not None:
        retriever_is_untied = opt_checkpoint.query_side_retriever_training or opt.query_side_retriever_training
    else:
        retriever_is_untied = opt.query_side_retriever_training

    if retriever_is_untied:
        retriever = UntiedDualEncoderRetriever(opt, contriever_encoder)
    else:
        retriever = DualEncoderRetriever(opt, contriever_encoder)

    return retriever, retriever_tokenizer


def _convert_state_dict_from_dual_encoder_retriever(state_dict):
    """handles when we want to load an UntiedDualEncoderRetriever from a DualEncoderRetriever state dict"""
    new_state_dict = {}
    for k, tensor in state_dict.items():
        if k.startswith("retriever"):
            new_state_dict[k.replace("retriever.contriever", "retriever.passage_contriever")] = tensor
            new_state_dict[k.replace("retriever.contriever", "retriever.query_contriever")] = tensor
        else:
            new_state_dict[k] = tensor
    return new_state_dict


def load_reader(opt, model_args, data_args, training_args, lora_args):
    reader = None
    if not opt.retrieve_only:
        device_map = None
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if lora_args.q_lora:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None

        compute_dtype = (
            torch.float16
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )

        if opt.fc_only:
            model_name_or_path = model_args.model_name_or_path
        else:
            model_name_or_path = opt.model_path + "/../../" + "checkpoint_reader"

        print(model_name_or_path)
        # print(model_args.model_name_or_path)
        reader = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            # model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if lora_args.q_lora
            else None,
        )

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            # target_modules=lora_args.lora_target_modules,
            target_modules=[
                "q_proj",
                "v_proj"
            ],
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            reader = prepare_model_for_kbit_training(
                reader, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            if not ddp and torch.cuda.device_count() > 1:
                # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
                reader.is_parallelizable = True
                reader.model_parallel = True

        reader = get_peft_model(reader, lora_config)
        reader.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            reader.enable_input_require_grads()

        # if opt.lclm:
        #     reader.create_crossattention_storage_lm()
        #
        # if opt.lclm or opt.compute_crossattention_stats or "eval" in opt.gold_score_mode or "std" in opt.gold_score_mode:
        #     reader.overwrite_forward_crossattention()
        #     reader.create_crossattention_storage()

    # reader_tokenizer = transformers.AutoTokenizer.from_pretrained(opt.reader_model_type, padding_side="right")
    reader_tokenizer = transformers.AutoTokenizer.from_pretrained(
        opt.reader_model_type,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # reader_tokenizer.pad_token = reader_tokenizer.unk_token  # FOR VICUNA
    # reader_tokenizer.pad_token = "</s>"  # FOR MISTRAL
    reader_tokenizer.pad_token = reader_tokenizer.unk_token # FOR MISTRAL
    reader_tokenizer.add_prefix_space = False  # FOR MISTRAL
    return reader, reader_tokenizer


def _set_reader_encoder_cfg(model, opt):
    if model.reader is not None:
        cfg = model.reader.encoder.config
        cfg.n_context = opt.n_context
        cfg.bsz = opt.per_gpu_batch_size


def _cast_atlas_to_precision(atlas_model, precision):
    # if atlas_model.reader is not None:
    #     atlas_model.reader = cast_to_precision(atlas_model.reader, precision)
    if atlas_model.retriever is not None and precision == "bf16":
        atlas_model.retriever = cast_to_precision(atlas_model.retriever, precision)


def _cast_and_set_attrs_and_send_to_device(model, opt):
    # _set_reader_encoder_cfg(model, opt)
    set_dropout(model, opt.dropout)
    _cast_atlas_to_precision(model, opt.precision)
    model = model.to(opt.device)
    return model


def _load_atlas_model_state(opt, opt_checkpoint, model, model_dict):
    model_dict = {
        k.replace("retriever.module", "retriever").replace("reader.module", "reader"): v for k, v in model_dict.items()
    }
    if opt.query_side_retriever_training and not opt_checkpoint.query_side_retriever_training:
        model_dict = _convert_state_dict_from_dual_encoder_retriever(model_dict)

    print("NO READER STATE")
    model_dict = {k: v for k, v in model_dict.items() if not k.startswith("reader")}
    # if opt.fc_only:  # dont load reader if in backbone training
    #     print("NO READER STATE")
    #     model_dict = {k: v for k, v in model_dict.items() if not k.startswith("reader")}

    # if opt.retrieve_only:  # dont load reader if in retrieve only mode
    #     model_dict = {k: v for k, v in model_dict.items() if not k.startswith("reader")}
    #
    # if opt.use_file_passages:  # dont load retriever if in use_file_passages mode
    #     model_dict = {k: v for k, v in model_dict.items() if not k.startswith("retriever")}

    model.load_state_dict(model_dict)
    # model = _cast_and_set_attrs_and_send_to_device(model, opt)
    return model


def load_atlas_model(dir_path, opt, model_args, data_args, training_args, lora_args, reset_params=False, eval_only=False):
    epoch_path = os.path.realpath(dir_path)
    save_path = os.path.join(epoch_path, "model.pth.tar")
    logger.info(f"Loading {epoch_path}")
    logger.info(f"loading checkpoint {save_path}")
    checkpoint = torch.load(save_path, map_location="cpu")
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    model_dict = checkpoint["model"]

    retriever, retriever_tokenizer = load_retriever(opt, opt_checkpoint)

    if opt.fc_only:
        reader, reader_tokenizer = load_reader(opt, model_args, data_args, training_args, lora_args)
        print("READER LOAD DONE")
        print("OKOKOK")
        model = Atlas(opt, None, retriever, reader_tokenizer, retriever_tokenizer)
        # model = Atlas(opt, reader, retriever, reader_tokenizer, retriever_tokenizer)
        # model = _load_atlas_model_state(opt, opt_checkpoint, model, model_dict)
        model.add_reader(reader)
    else:
        print("RELOAD")
        # model_reader_dict = checkpoint["model_reader"]
        # model.reader.save_pretrained(os.path.join(dir_path, "checkpoint_reader"), from_pt=True)

        # OPTION 1 (FOR INFERENCE)
        # compute_dtype = (
        #     torch.float16
        #     if training_args.fp16
        #     else (torch.bfloat16 if training_args.bf16 else torch.float32)
        # )
        #
        # reader_tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     opt.reader_model_type,
        #     cache_dir=training_args.cache_dir,
        #     model_max_length=training_args.model_max_length,
        #     padding_side="right",
        #     use_fast=False,
        # )
        # reader_tokenizer.pad_token = reader_tokenizer.unk_token
        # print(opt.model_path + "/../../" + "checkpoint_reader")
        # reader = AutoPeftModelForCausalLM.from_pretrained(
        #     # os.path.join(dir_path, "../../checkpoint_reader"),
        #     opt.model_path + "/../../" + "checkpoint_reader",
        #     cache_dir=training_args.cache_dir,
        #     # device_map=device_map,
        #     quantization_config=BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=compute_dtype,
        #     )
        #     if lora_args.q_lora
        #     else None,
        #     # device_map='auto',
        #     # trust_remote_code=False,
        #     # revision="main",
        #     # attn_implementation="flash_attention_2"
        # )

        # OPTION 2 (FOR TRAINING)
        reader, reader_tokenizer = load_reader(opt, model_args, data_args, training_args, lora_args)

        model = Atlas(opt, None, retriever, reader_tokenizer, retriever_tokenizer)
        model = _load_atlas_model_state(opt, opt_checkpoint, model, model_dict)
        # reader.load_state_dict(model_reader_dict)
        model.add_reader(reader)
        print("SUCCESS")

    model = _cast_and_set_attrs_and_send_to_device(model, opt)

    if eval_only:
        return model, None, None, None, None, opt_checkpoint, step

    if not reset_params:
        optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt, model)

    return (
        model,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        opt_checkpoint,
        step,
    )


def init_atlas_model(opt, eval_only):
    reader, reader_tokenizer = load_reader(opt)
    retriever, retriever_tokenizer = load_retriever(opt)

    model = Atlas(opt, reader, retriever, reader_tokenizer, retriever_tokenizer)
    model = _cast_and_set_attrs_and_send_to_device(model, opt)

    if eval_only:
        return model, None, None, None, None, opt, 0

    optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt, model)
    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, 0


def load_or_initialize_atlas_model(opt, model_args, data_args, training_args, lora_args, eval_only=False):
    """
    Either initializes a Atlas from t5 and contriever or loads one from disk.

    if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} doesn't exist, it will init a Atlas

    or, if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} does exist, it will load the Atlas at opt.checkpoint_dir/opt.name/latest

    or, if opt.model_path is not "none" it will load the saved Atlas in opt.model_path
    """
    checkpoint_path = get_checkpoint_path(opt)
    latest_checkpoint_path = os.path.join(checkpoint_path, "checkpoint", "latest")

    if opt.model_path == "none":
        if not os.path.exists(latest_checkpoint_path):  # Fresh run:
            return init_atlas_model(opt, eval_only)
        else:  # Resume  run
            load_path, reset_params = latest_checkpoint_path, False
    else:  # fresh finetune run, initialized from old model
        load_path, reset_params = opt.model_path, True

    (
        model,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        opt_checkpoint,
        loaded_step,
    ) = load_atlas_model(load_path, opt, model_args, data_args, training_args, lora_args, reset_params=reset_params, eval_only=eval_only)
    logger.info(f"Model loaded from {load_path}")
    step = 0 if opt.model_path != "none" else loaded_step

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step


def save_atlas_model(
    model,
    optimizer,
    scheduler,
    retr_optimizer,
    retr_scheduler,
    step,
    opt,
    dir_path,
    name,
):

    if opt.save_optimizer and opt.shard_optim:
        optimizer.consolidate_state_dict()
        if retr_optimizer:
            retr_optimizer.consolidate_state_dict()

    if not opt.is_main:
        return 0

    def symlink_force(target, link_name):
        try:
            os.symlink(target, link_name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(link_name)
                os.symlink(target, link_name)
            else:
                raise e

    model.reader.save_pretrained(os.path.join(dir_path, "checkpoint_reader"), from_pt=True)

    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "model.pth.tar")

    optim_state = optimizer.state_dict() if opt.save_optimizer else None
    if retr_optimizer and opt.save_optimizer:
        retr_optim_state = retr_optimizer.state_dict()
    else:
        retr_optim_state = None
    checkpoint = {
        "step": step,
        "model": model_to_save.state_dict(),
        # "model_reader": model_to_save.reader.state_dict(),
        "optimizer": optim_state,
        "retr_optimizer": retr_optim_state,
        "scheduler": scheduler.state_dict(),
        "retr_scheduler": retr_scheduler.state_dict() if retr_scheduler else None,
        "opt": opt,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)
    if opt.save_optimizer and opt.shard_optim:
        optimizer._all_states = []
