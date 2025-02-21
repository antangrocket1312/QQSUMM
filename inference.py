import os
import time
from collections import defaultdict

import numpy as np
import torch

import torch.cuda
import logging
from evaluates import evaluate
from src import dist_utils, slurm, util
from src.torchrun_utils import init_distributed_mode_torchrun
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import (
    create_checkpoint_directories,
    load_or_initialize_atlas_model,
    save_atlas_model,
)
from src.options import get_options
from src.tasks import get_task
import torch.distributed as dist

from dataclasses import dataclass, field
import logging
import typing
import os
import transformers
from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    if "TORCHELASTIC_RUN_ID" in os.environ:
        init_distributed_mode_torchrun(opt)
        torch.cuda.set_device(dist.get_rank())
    else:
        slurm.init_distributed_mode(opt)
        slurm.init_signal_handler()
    # torchrun_utils.init_distributed_mode_torchrun(opt)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        unknown_args,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)
    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    logger.info(f"world size: {dist_utils.get_world_size()}")
    index, passages = load_or_initialize_index(opt)

    (
        model,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        opt,
        step,
    ) = load_or_initialize_atlas_model(opt, model_args, data_args, training_args, lora_args)

    if opt.is_distributed:
        if opt.shard_grads:
            import fairscale.nn.data_parallel

            model.reader = fairscale.nn.data_parallel.ShardedDataParallel(
                model.reader, optimizer, auto_refresh_trainable=False
            )
            if opt.train_retriever:
                model.retriever = fairscale.nn.data_parallel.ShardedDataParallel(
                    model.retriever, retr_optimizer, auto_refresh_trainable=False
                )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=True,
            )
            model._set_static_graph()

    logger.info("Start training")
    dist_utils.barrier()

    tb_logger = util.init_tb_logger(os.path.join(opt.checkpoint_dir, opt.name), is_main=opt.is_main)
    run_stats = util.WeightedAvgStats()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)

    torch.manual_seed(opt.global_rank + opt.seed)

    scale = 2.0
    grad_stats = defaultdict(lambda: [])
    task = get_task(opt, unwrapped_model.reader_tokenizer)
    index_refresh_scheduler = util.IndexRefreshScheduler(
        opt.refresh_index, opt.freeze_retriever_steps, opt.train_retriever
    )

    while step < opt.total_steps:
        print("START TRAINING STEP", step)
        data_iterator = task.data_iterator(
            opt.train_data,
            opt.global_rank,
            opt.world_size,
            repeat_if_less_than_world_size=True,
            opt=opt,
        )
        data_iterator = filter(None, map(task.process, data_iterator))
        data_iterator = task.batch_iterator(data_iterator, opt.per_gpu_batch_size, drop_last=True, shuffle=opt.shuffle)

        for i, batch in enumerate(data_iterator):
            print("TRAINING ITEM ", i)
            iter_stats = {}
            model.train()
            if not opt.use_file_passages and index_refresh_scheduler.is_time_to_refresh(step):

                if not (step == 0 and opt.load_index_path is not None):  # Dont refresh index if just loaded it
                    print("INDEXING")
                    indexing_start = time.time()
                    unwrapped_model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)
                    iter_stats["runtime/indexing"] = (time.time() - indexing_start, 1)

                    if opt.save_index_path is not None:
                        save_embeddings_and_index(index, opt)
            step += 1
            train_step_start = time.time()

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)

        metrics = evaluate(model, index, opt, data_path, step)
        log_message = f"Dataset: {dataset_name}"
        for k, v in metrics.items():
            log_message += f" | {v:.3f} {k}"
            if tb_logger:
                tb_logger.add_scalar(f"{dataset_name}/{k}", v, step)
        logger.info(log_message)




