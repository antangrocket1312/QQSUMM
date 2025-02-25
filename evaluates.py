# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict
import torch
import torch.cuda
import torch.distributed as dist

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator


@torch.no_grad()
def run_retrieval_only(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        ids = batch.get("id", [""])
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)
        retrieved_passages, _ = unwrapped_model.retrieve(
            ids,
            index,
            opt.n_context,
            query,
            query_enc["input_ids"].cuda(),
            query_enc["attention_mask"].cuda(),
            is_train=False,
            batch_metadata=batch_metadata,
            filtering_fun=task.filter,
        )
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        for k in range(len(retrieved_passages)):
            if opt.write_results:
                gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
                ex = {
                    "query": query[k],
                    "answers": gold,
                    "passages": retrieved_passages[k],
                }
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


@torch.no_grad()
def evaluate(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)
    all_cls, all_cls_pred = [], []

    for i, batch in enumerate(data_iterator):

        forward_start = time.time()
        ids = batch.get("id", [""])
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        cls_label = batch.get("label", [""])

        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")
        query_enc = unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)
        # query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)

        if not opt.use_file_passages:
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            retrieved_passages, retrieved_scores = unwrapped_model.retrieve(
                ids,
                index,
                opt.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
                is_train=False,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )

        comment_clusters = []
        for q, pas in zip(query, retrieved_passages):
            filtered_clusters, _ = unwrapped_model.cluster_retrieved_passages(q, [p['text'] for p in pas])
            filtered_clusters = [{'cluster_id': i, 'comments': c, 'cluster_size': len(c)} for i, c in enumerate(filtered_clusters)]
            # filtered_clusters = [{'comments': c, 'cluster_size': len(c)} for c in filtered_clusters]
            print("FILTERED CLUSTER SIZE: ", len(filtered_clusters))
            comment_clusters += [filtered_clusters]

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue

        reader_tokens = unwrapped_model.tokenize_reader_prompt(query, comment_clusters)

        generation, prediction, cls_label = unwrapped_model.generate(
            reader_tokens,
            query,
            cls_label,
            choices=batch["choices"] if "choices" in batch else None,
        )

        all_cls += cls_label

        if prediction is not None:
            all_cls_pred += prediction.cpu().tolist()
        else:
            all_cls_pred += cls_label

        for k, g in enumerate(generation):
            if opt.decoder_prompt_format is not None:
                query_ids = reader_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}),
                    add_special_tokens=False,
                )
                g = g[len(query_ids) + 1:]
            pred = reader_tokenizer.decode(g)
            # pred = reader_tokenizer.decode(g, skip_special_tokens=True)
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]

            if opt.write_results:
                if prediction is not None:
                    ppp = prediction[k].cpu().item()
                else:
                    ppp = cls_label[k]
                ex = {"query": query[k], "answers": gold, "generation": pred, "cls_label": cls_label[k],
                      "cls_pred": ppp}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                    ex["passages_scores"] = retrieved_scores[k]
                    ex["comment_clusters"] = comment_clusters[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)

    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = "test-result"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    index, passages = load_or_initialize_index(opt)
    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)

    logger.info("Start Evaluation")
    dist_utils.barrier()

    if not opt.use_file_passages and opt.load_index_path is None:
        indexing_start = time.time()
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

        if opt.save_index_path is not None:
            save_embeddings_and_index(index, opt)

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        if opt.retrieve_only:
            run_retrieval_only(model, index, opt, data_path, step)
        else:
            metrics = evaluate(model, index, opt, data_path, step)
            log_message = f"Dataset: {dataset_name}"
            for k, v in metrics.items():
                log_message += f" | {v:.3f} {k}"
            logger.info(log_message)