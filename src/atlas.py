# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import math
import json
import time
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
import datasets
import statistics

from src import dist_utils, util
from src.retrievers import EMBEDDINGS_DIM

logger = logging.getLogger(__name__)
IGNORE_INDEX: int = -100
BERT_MAX_SEQ_LENGTH: int = 512


BASE_PROMPT = """You will be provided with a question and a JSON list of relevant review comments, delimited by triple quotes.
The question asks the opinions of user reviews about a product, and can be answered by the list of comment clusters in the provided JSON list. Each element in the JSON has been has been clustered to represent a common opinion answering the question, accompanied by the quantity.

You were tasked to generate a quantitative summary that covers all opinions captured in the JSON list in answering the questions.

Perform the following actions to solve this task:
- For every element in the JSON list, find the key point that represent the common opinion across the comments of the cluster
- Generate a long-form quantitative summary including all extracted key points and the cluster size, following the below template:
'While answering about [Question]:
+ [Cluster size] of comments believe that [Key Point 1] (ID: [Cluster ID])
+ [Cluster size] of comments believe that [Key Point 2] (ID: [Cluster ID])
...'

Below are fundamental rules:
+ Larger cluster means higher support for the key point and with a bigger cluster size, the quantity must be higher
+ Only use number to report the cluster size for each key point, avoiding vague terms (e.g., some, most)
+ Ensure that each key point extracted from a cluster is distinctive and doesn't redundantly cover aspects mentioned in larger clusters

 """

INPUT_TEMPLATE = """Question: \"\"\"%s\"\"\"
JSON List: \"\"\"%s\"\"\""""

def nearest_lower_divisor(number, upper_limit):
    # Start from the upper_limit and go downwards
    for i in range(upper_limit, 0, -1):
        if number % i == 0:
            return i
    return None  # In case no divisor is found, though 1 is always a divisor


def encode_passages(batch, tokenizer, max_length):
    bsz = len(batch)
    n = max([len(example) for example in batch])
    batch = [example + [""] * (n - len(example)) for example in batch]
    batch = reduce(lambda a, b: a + b, batch)
    tokens = tokenizer(
        batch,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
    return tokens


def encode_clusters(batch, tokenizer, max_length):
    cluster_tok_batch = []
    for clusters in batch:
        cluster_tok = []
        for clus in clusters:
            tokens = tokenizer(
                clus['comments'],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
                truncation=True,
            )
            cluster_tok += [tokens]
        cluster_tok_batch += [cluster_tok]
    return cluster_tok_batch


# def encode_reader_prompt(batch, tokenizer, max_length):
#     tokens = tokenizer(
#         batch,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=max_length,
#         truncation=True,
#     )
#
#     # Copy the input_ids to labels, forward function of LLM automatically shift right
#     tokens['labels'] = tokens.input_ids[:]
#     # Create labels by shifting the input_ids to the right by one position
#     # tokens['labels'] = tokens.input_ids[:]  # Copy the input_ids to labels
#     # tokens['labels'][:-1] = tokens['input_ids'][1:]  # Shift labels to the right
#     # tokens['labels'][-1] = -100  # Mask the last position
#
#     return tokens

def encode_reader_prompt_train(batch, tokenizer, max_length):
    tokens = tokenizer(
        batch,
        # datasets.Dataset.(batch),
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        # padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
        return_length=False
    )
    # print("TOKENS", tokens)
    # print(type(tokens))

    dataloader_params = {
        "batch_size": len(batch),
        "collate_fn": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "drop_last": False,
        'prefetch_factor': None
    }
    train_dataloader = DataLoader(datasets.Dataset.from_dict(tokens), **dataloader_params)
    for step, inputs in enumerate(train_dataloader):
        print("DATALOADER LOOPING")
        tokens = inputs
    # print("DATALOADER TOKENS", tokens)
    return tokens


def encode_reader_prompt_inference(batch, tokenizer):
    tokens = tokenizer(
        batch,
        return_tensors='pt'
    )
    return tokens


class Atlas(nn.Module):
    def __init__(self, opt, reader, retriever, reader_tokenizer, retriever_tokenizer):
        super(Atlas, self).__init__()

        self.reader = reader
        self.retriever = retriever
        self.reader_tokenizer = reader_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.opt = opt
        if self.opt.mt:
            self.cls_loss = nn.CrossEntropyLoss(reduction='none')

        if self.opt.lclm or self.opt.lglm or self.opt.fc_only or self.opt.lgret or self.opt.lcret:
            # Read the BM25 retrieved file as reference for Distillation Training
            self.all_fc = json.load(open(self.opt.fc_file))
            self.all_fc = {int(k): v for k, v in self.all_fc.items()}

            # Read the KP Matched Clusters file as reference for Distillation Training
            self.all_cluster = json.load(open(self.opt.cluster_file))
            self.all_cluster = {int(k): v for k, v in self.all_cluster.items()}

            self.mse_loss = torch.nn.MSELoss()

        # self.READER_ALL_TOKENS = list(self.reader_tokenizer.vocab.values())

    def add_reader(self, reader):
        self.reader = reader

    def _get_fp16_retriever_copy(self):
        if hasattr(self.retriever, "module"):
            retriever_to_copy = self.retriever.module
        else:
            retriever_to_copy = self.retriever
        return copy.deepcopy(retriever_to_copy)
        # return retriever_to_copy
        # return copy.deepcopy(retriever_to_copy).half().eval()

    @torch.no_grad()
    def build_index(self, index, passages, gpu_embedder_batch_size, logger=None):
        n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
        retrieverfp16 = self._get_fp16_retriever_copy()

        total = 0
        for i in range(n_batch):
            batch = passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
            batch = [self.opt.retriever_format.format(**example) for example in batch]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, gpu_embedder_batch_size),
                truncation=True,
            )

            embeddings = retrieverfp16(**_to_cuda(batch_enc), is_passages=True)
            index.embeddings[:, total : total + len(embeddings)] = embeddings.T
            total += len(embeddings)
            if i % 500 == 0 and i > 0:
                logger.info(f"Number of passages encoded: {total}")
        dist_utils.barrier()
        logger.info(f"{total} passages encoded on process: {dist_utils.get_rank()}")

        if not index.is_index_trained():
            logger.info(f"Building faiss indices")
            index.train_index()

    @torch.no_grad()
    def _retrieve(
        self,
        ids,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        is_train,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
    ):
        self.retriever.eval()
        if len(query) > 0:
            query_emb = self.retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
        else:
            query_emb = torch.empty((0, EMBEDDINGS_DIM)).cuda()  # TODO: broken
        if self.training:
            self.retriever.train()

        search_start = time.time()
        if filtering_fun is not None:
            passages, scores = index.search_knn(query_emb, topk * self.opt.filtering_overretrieve_ratio)
            passages, scores = filtering_fun(batch_metadata, passages, scores, topk, training=self.training)
        else:
            passages, scores = index.search_knn(query_emb, topk)
        iter_stats["runtime/search"] = (time.time() - search_start, 1)

        return passages, scores, query_emb

    @torch.no_grad()
    def retrieve_with_rerank(
        self,
        ids,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        is_train,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={}
    ):
        bsz = len(query)
        to_rerank = self.opt.n_to_rerank_with_retrieve_with_rerank

        # first, do the retrieval
        passages, _, query_emb = self._retrieve(
            ids,
            index,
            to_rerank,
            query,
            query_ids_retriever,
            query_mask_retriever,
            is_train,
            batch_metadata,
            filtering_fun,
            iter_stats,
        )

        retrieverfp16 = self._get_fp16_retriever_copy()
        fstr = self.opt.retriever_format
        flat_passage_strings = [fstr.format(**p) for ps in passages for p in ps]
        encoder_batch_size = min(len(flat_passage_strings), self.opt.per_gpu_embedder_batch_size)
        passage_emb, output_passages, output_scores = (
            query_emb.new_zeros(len(flat_passage_strings), query_emb.shape[-1]),
            [],
            [],
        )

        for b in range(0, len(flat_passage_strings), encoder_batch_size):
            batch = flat_passage_strings[b : b + encoder_batch_size]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                truncation=True,
            )
            batch_emb = retrieverfp16(**_to_cuda(batch_enc), is_passages=True).to(query_emb)
            passage_emb[b : b + encoder_batch_size] = batch_emb

        passage_emb = passage_emb.view(bsz, min(passage_emb.size()[0], to_rerank), 768)
        # passage_emb = passage_emb.view(bsz, to_rerank, 768)

        # Rerank based on retriever scores
        # DOT PRODUCT OF EMBEDDINGS (TO GET SIMILARITY SCORE BETWEEN THE QUERY AND EACH PASSAGE)
        retriever_scores = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
        # top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, topk, dim=1)

        final_output_passages, final_output_scores = ([], [])
        if self.opt.train_with_rank_threshold or not is_train:
            # Train with threshold or inference with threshold

            if self.opt.max_retrieval is not None:
                top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, min(len(passages[0]), self.opt.max_retrieval), dim=1)
                for i in range(bsz):
                    output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
                    output_scores.append(top_retriever_scores[i].tolist())
            else:
                print("SCORE DESC BY THRES")
                top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, len(passages[0]), dim=1)
                for i in range(bsz):
                    output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
                    output_scores.append(top_retriever_scores[i].tolist())
                # output_passages = passages
                # output_scores = retriever_scores

            for i in range(bsz):
                output_passage = []
                output_score = []
                for passage, score in zip(output_passages[i], output_scores[i]):
                # for passage, score in zip(passages[i], retriever_scores[i]):
                    if score >= self.opt.rank_threshold:
                        output_passage += [passage]
                        output_score += [score]
                final_output_passages += [output_passage]
                final_output_scores += [output_score]
            # print("TOP RETRIEVER SCORES", len(output_scores[0]), output_scores[0])
        else:
            # Default training, dynamically taking the size of query-relevant comments as topk
            top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, min(len(passages[0]), len(self.all_fc[int(ids[0])])), dim=1)
            # # FOR MANUALLY INSPECTING AND TUNING FOR THE RETRIEVAL THRESHOLD
            # print("RETRIEVER SCORES", retriever_scores)
            # print("TOP RETRIEVER SCORES", top_retriever_scores.size(), top_retriever_scores)
            for i in range(bsz):
                final_output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
                final_output_scores.append(top_retriever_scores[i].tolist())

        return final_output_passages, final_output_scores
        # return output_passages, output_scores

    @torch.no_grad()
    def retrieve(self, *args, **kwargs):
        retrieve_func = self.retrieve_with_rerank if self.opt.retrieve_with_rerank else self._retrieve
        passages, scores = retrieve_func(*args, **kwargs)[:2]
        return passages, scores

    def append_query(self, query, passages):
        return [self.opt.encoder_format.format(query=query, **p) for p in passages]

    def retriever_tokenize(self, query):
        if self.retriever_tokenizer:
            query_enc = self.retriever_tokenizer(
                query,
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            query_enc = _to_cuda(query_enc)
        else:
            query_enc = None
        return _to_cuda(query_enc)

    # def reader_tokenize(self, query, target, target_tokens):
    #     if target_tokens is None:
    #         if self.opt.decoder_prompt_format is not None:
    #             modified_query = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
    #             target = [q + t for (q, t) in zip(modified_query, target)]
    #
    #             query_mask = self.reader_tokenizer(
    #                 modified_query,
    #                 max_length=self.opt.target_maxlength,
    #                 padding="max_length",
    #                 truncation=True,
    #                 return_tensors="pt",
    #                 add_special_tokens=False,
    #             )["attention_mask"]
    #
    #         if self.opt.decoder_format is not None:
    #             target = [self.opt.decoder_format.format(target=t) for t in target]
    #         target = [t + "</s>" if not t.endswith("</s>") else t for t in target]
    #
    #         target_tokens = self.reader_tokenizer(
    #             target,
    #             max_length=self.opt.target_maxlength,
    #             padding="max_length",
    #             truncation=True,
    #             return_tensors="pt",
    #             add_special_tokens=False,
    #         )
    #
    #     decoder_input_ids = self.reader._shift_right(target_tokens["input_ids"])
    #     labels = target_tokens["input_ids"].masked_fill(~target_tokens["attention_mask"].bool(), IGNORE_INDEX)
    #
    #     # If decoder prompt is not None mask labels such that the model is not trained to predict the prompt
    #     if self.opt.decoder_prompt_format is not None:
    #         query_mask = self.reader_tokenizer(
    #             modified_query,
    #             max_length=self.opt.target_maxlength,
    #             padding="max_length",
    #             truncation=True,
    #             return_tensors="pt",
    #             add_special_tokens=False,
    #         )["attention_mask"]
    #
    #         padding = torch.zeros(
    #             (
    #                 query_mask.size(0),
    #                 target_tokens["input_ids"].size(-1) - query_mask.size(-1),
    #             )
    #         )
    #         query_mask = torch.cat([query_mask, padding], dim=1)
    #         labels = labels.masked_fill(query_mask.bool(), IGNORE_INDEX)
    #
    #     return labels.cuda(), decoder_input_ids.cuda()

    # def reader_tokenize_new(self, target):
    #     target = [t + "</s>" if not t.endswith("</s>") else t for t in target]
    #     target_tokens = self.reader_tokenizer(
    #         target,
    #         max_length=self.opt.target_maxlength,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #         add_special_tokens=False,
    #     )
    #    ...

    def tokenize(self, query, target, target_tokens):
        if query is None and target is None:
            return None, None, None

        assert (
            target_tokens is None or self.opt.decoder_prompt_format is None
        ), "decoder_prompt_format not compatible with target tokenized in iterator"

        query_enc = self.retriever_tokenize(query) if not self.opt.use_file_passages else None
        return query_enc

        # labels, decoder_input_ids = self.reader_tokenize_new(target)
        # labels, decoder_input_ids = self.reader_tokenize(query, target, target_tokens)
        # return query_enc, labels, decoder_input_ids

    def tokenize_reader_prompt(self, query, comment_clusters, target=None):
        if target != None:  # Training mode
            reader_toks = []
            reader_prompt_kps = []
            for q, clusters, t in zip(query, comment_clusters, target):
                kps = t.split("\n")
                reader_prompt_kp = []
                for cumulative_clusters, cumulative_kps  in zip([clusters[:i] for i in range(1, len(clusters) + 1)],
                                                                [kps[:i] for i in range(1, len(kps) + 1)]):
                    if len(cumulative_kps) == 1:
                        prompt = "<s>[INST] " + BASE_PROMPT + INPUT_TEMPLATE % (q, str(cumulative_clusters)) + "Summary: [/INST]" + "\n" + cumulative_kps[-1] + "</s>"
                        reader_prompt_kp += [prompt]
                    else:
                        prompt = "<s>[INST] " + BASE_PROMPT + INPUT_TEMPLATE % (q, str(cumulative_clusters)) + "Summary:" + "\n" + "\n".join(cumulative_kps[:-1]) + " [/INST]" + "\n" + cumulative_kps[-1] + "</s>"
                        reader_prompt_kp += [prompt]
                    # print(prompt)

                reader_prompt_kps += [reader_prompt_kp]

            for single_batch in reader_prompt_kps:
                single_batch_tok = []
                for reader_prompt_kp in single_batch:
                    reader_tok = encode_reader_prompt_train([reader_prompt_kp], self.reader_tokenizer, self.opt.text_maxlength)
                    reader_tok = _to_cuda(reader_tok)
                    single_batch_tok += [reader_tok]
                reader_toks += [single_batch_tok]

        else: # Inference mode
            reader_prompt = ["<s>[INST] " + BASE_PROMPT + INPUT_TEMPLATE % (q, str(clusters)) + "Summary: [/INST]"
                             for q, clusters in zip(query, comment_clusters)]
            # print("READER PROMPT INFERENCE", reader_prompt)
            reader_toks = encode_reader_prompt_inference(reader_prompt, self.reader_tokenizer)
            reader_toks = _to_cuda(reader_toks)

        # reader_tok = encode_reader_prompt(reader_prompt, self.reader_tokenizer, self.opt.text_maxlength)
        return reader_toks

    # def format_prompt(self, q, passages):
    #     prompt = BASE_PROMPT + INPUT_TEMPLATE % (q, [p['text'] for p in passages])
    #     return [prompt]

    def tokenize_passages(self, query, passages):
        if len(query) == 0:
            return None, None

        # query_passages = [self.format_prompt(q, p) for q, p in zip(query, passages)]
        # query_passages = [self.append_query(q, p) for q, p in zip(query, passages)]

        fstr = self.opt.retriever_format
        retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        if self.retriever_tokenizer:
            retriever_tok = encode_passages(
                retriever_passages,
                self.retriever_tokenizer,
                min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            retriever_tok = _to_cuda(retriever_tok)
        else:
            retriever_tok = None

        return retriever_tok
        # reader_tok = encode_passages(query_passages, self.reader_tokenizer, self.opt.text_maxlength)
        # reader_tok = _to_cuda(reader_tok)
        # print("QUERY PASSAGES", query_passages)
        # print("RETRIEVER PASSAGES", retriever_passages)
        # return reader_tok, retriever_tok

    def tokenize_clusters(self, query, comment_clusters):
        if len(query) == 0:
            return None, None

        if self.retriever_tokenizer:
            cluster_tok = encode_clusters(
                comment_clusters,
                self.retriever_tokenizer,
                min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            cluster_tok = [[_to_cuda(clus) for clus in clusters] for clusters in cluster_tok]
        else:
            cluster_tok = None

        return cluster_tok

    def get_labels(self, decoder_input_ids, padding_token_id):
        """
        Calculate labels for a decoder-only language model.

        Args:
            decoder_input_ids (torch.Tensor): Input token IDs for the decoder. Shape: (batch_size, seq_len)
            padding_token_id (int): Token ID used for padding.

        Returns:
            torch.Tensor: Labels tensor for loss calculation. Shape: (batch_size, seq_len - 1)
        """
        # Shift decoder input IDs one position to the left
        labels = decoder_input_ids[:, 1:].contiguous()  # Shape: (batch_size, seq_len - 1)

        # Replace padding tokens with -1 to ignore during loss calculation
        labels[labels == padding_token_id] = -1

        return labels

    def perplexity_score(self, reader_output, reader_tokens, bsz):
        labels = self.get_labels(reader_tokens['input_ids'], self.reader_tokenizer.unk_token)

        with torch.no_grad():
            self.reader.eval()
            total_context = reader_tokens['input_ids'].size(1)
            # cfg.n_context = 1
            # cfg.bsz = bsz * total_context

            # Logits from the model
            logits = reader_output.logits  # (batch_size, seq_len, vocab_size)

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Gather log probabilities of ground truth tokens
            gold_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            # Mask padding tokens
            valid_mask = (labels != -1)  # Padding tokens are assumed to have label -1
            valid_gold_log_probs = gold_log_probs * valid_mask

            # Sum log-probabilities for valid tokens
            sequence_log_prob = valid_gold_log_probs.sum(dim=-1)

            # Normalize by the number of valid tokens
            num_valid_tokens = valid_mask.sum(dim=-1)
            gold_score = -sequence_log_prob / num_valid_tokens
            # print(gold_score)
            # return gold_score.cpu().item()  # Shape: (batch_size,)
            return gold_score  # Shape: (batch_size,)

            # Reshape to (batch_size, total_context) and aggregate across contexts
            # gold_score = gold_score.view(bsz, total_context)  # Reshape for per-query context aggregation
            # aggregated_gold_score = gold_score.mean(dim=-1)  # Aggregate across contexts (e.g., mean or sum)
            # return aggregated_gold_score  # Shape: (batch_size,)

    def eval_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz, mask_query):
        self.reader.eval()
        self.reader.reset_score_storage()
        cfg.bsz = reader_ids.size(0)
        cfg.n_context = reader_ids.size(1)
        reader_ids_score = reader_ids.view(reader_ids.size(0), -1)
        reader_mask_score = reader_mask.view(reader_mask.size(0), -1)
        with torch.no_grad():
            reader_output = self.reader(
                input_ids=reader_ids_score,
                attention_mask=reader_mask_score,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=False,
            )
            crossattention_scores = self.reader.get_crossattention_scores(
                cfg.n_context,
                reader_mask_score,
                labels=labels,
                ids=reader_ids,
                mode=self.opt.gold_score_mode,
                mask_query=mask_query,
            )
            gold_score = select_crossattention_scores(crossattention_scores, self.opt.gold_score_mode)

            if self.training:
                self.reader.train()
            return gold_score

    def loop_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        with torch.no_grad():
            total_context = reader_ids.size(1)
            doc_len = reader_ids.size(-1)
            self.reader.eval()
            cfg.bsz = bsz
            cfg.n_context = total_context
            reader_ids_score_eval = reader_ids.view(reader_ids.size(0), -1)
            reader_mask_score_eval = reader_mask.view(reader_mask.size(0), -1)

            # forward pass for calculating and caching the encoder states:
            reader_output_eval = self.reader(
                input_ids=reader_ids_score_eval,
                attention_mask=reader_mask_score_eval,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=False,
            )
            eval_hidden_state = reader_output_eval.encoder_last_hidden_state

            # run n_docs - 1 forward passes to calculate pp when leaving a doc out
            gold_scores = []
            for loo_index in range(total_context):
                reader_mask_loo = reader_mask.clone()
                reader_mask_loo[:, loo_index] = False  # mask out this doc
                loo_output_eval = self.reader(
                    encoder_outputs=[eval_hidden_state],
                    attention_mask=reader_mask_loo.view(bsz, (total_context) * doc_len),
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    use_cache=False,
                )
                token_loss = nn.functional.cross_entropy(
                    loo_output_eval.logits.view(-1, loo_output_eval.logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                )
                mean_loss = token_loss.view(bsz, labels.shape[-1]).sum(dim=-1) / (labels > -1).sum(-1)
                gold_scores.append(mean_loss)

            gold_score = torch.stack(gold_scores, dim=1)

            return gold_score

    @torch.no_grad()
    def emdr_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        self.reader.eval()
        cfg.n_context = 1
        cfg.bsz = bsz * self.opt.retriever_n_context
        reader_ids_score = reader_ids.view(bsz * self.opt.retriever_n_context, -1)
        reader_mask_score = reader_mask.view(bsz * self.opt.retriever_n_context, -1)
        repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, self.opt.retriever_n_context, dim=0)
        repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
        reader_output = self.reader(
            input_ids=reader_ids_score.cuda(),
            attention_mask=reader_mask_score.cuda(),
            labels=repeated_labels,
            use_cache=False,
        )
        gold_score = reader_output.logits
        return gold_score

    # TODO: May enhance the score to compare with self.all_clusters to calculate loss and then improve training
    def lgret_score(self, ids, passage_emb):
        fc_passages_embs = torch.Tensor([]).to(passage_emb)
        for id in ids:
            split_passages = util.split_fc_pas(id, self.all_fc[id])
            fc_tokens = self.tokenize_passages(["" for i in range(len(split_passages))], [split_passages])
            # print("FC TOKENS BEFORE RESHAPE ", fc_tokens)
            fc_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in fc_tokens.items()}
            # print("FC TOKENS AFTER RESHAPE ", fc_tokens)
            # torch.save(fc_tokens, f'./fc_lgret_embedding_export/fc_lgret_enc_export_{id}.pt')
            # TODO: Focus here
            fc_split_passage_emb = self.retriever(**fc_tokens, is_passages=False).to(passage_emb)
            # print("FC LENGTH ", len(self.all_fc[id]))
            # print("FC EMBEDDING SIZE ", fc_split_passage_emb.size())
            # torch.save(fc_split_passage_emb, f'./fc_lgret_embedding_export/fc_lgret_embedding_export_{id}.pt')
            fc_passage_emb = torch.mean(fc_split_passage_emb,dim=0).unsqueeze(0)
            fc_passages_embs = torch.cat((fc_passages_embs,fc_passage_emb))

        # # print("NUM OF PREDICTED PASSAGES", passage_emb.size()[1])
        # print("PREDICTED PASSAGES EMB SHAPE", passage_emb.size())
        fc_passages_embs = fc_passages_embs.unsqueeze(0).repeat((1,passage_emb.size()[1],1))
        loss = self.mse_loss(fc_passages_embs,passage_emb)
        return loss

    # def lgret_score(self, ids, passage_emb):
    #     fc_passages_embs = torch.Tensor([]).to(passage_emb)
    #     for id in ids:
    #         split_passages = util.split_fc_pas(id, self.all_fc[id])
    #         fc_tokens = self.tokenize_passages(["" for i in range(len(split_passages))], [split_passages])
    #         fc_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in fc_tokens.items()}
    #         fc_split_passage_emb = self.retriever(**fc_tokens, is_passages=False).to(passage_emb)
    #         fc_passages_embs = torch.cat((fc_passages_embs, fc_split_passage_emb))
    #
    #     fc_passages_embs = fc_passages_embs.unsqueeze(0)
    #     # print("NUM OF PREDICTED PASSAGES", passage_emb.size()[1])
    #     print("PREDICTED PASSAGES EMB SHAPE", passage_emb.size())
    #     print("FC EMB SHAPE ", fc_passages_embs.size())
    #     loss = self.mse_loss(fc_passages_embs,passage_emb)
    #     return loss


    def lgret_score_cluster(self, ids, clusters_emb, passage_emb):
        # clusters_emb = []
        fc_clusters_emb_centroid = torch.Tensor([]).to(passage_emb)
        most_similar_fc_clusters_emb_centroid = torch.Tensor([]).to(passage_emb)
        clusters_emb_centroid = torch.Tensor([]).to(passage_emb)

        # CALCULATE FC CLUSTER EMBEDDING (CENTROID)
        for id in ids:
            fc_comment_clusters = self.all_cluster[id]
            fc_clusters_tokens = self.tokenize_clusters(["" for i in range(len(fc_comment_clusters))], [fc_comment_clusters])
            print("fc_clusters_tokens", len(fc_clusters_tokens))
            fc_clusters = fc_clusters_tokens[0]
            fc_clusters_embedding_centroid = torch.Tensor([]).to(passage_emb)
            for fc_cluster_tok in fc_clusters:  # Each fc cluster
                # TODO: Focus here
                fc_cluster_emb = self.retriever(**fc_cluster_tok, is_passages=False).to(passage_emb)
                fc_cluster_emb_centroid = torch.mean(fc_cluster_emb, dim=0).unsqueeze(0)
                fc_clusters_embedding_centroid = torch.concat([fc_clusters_embedding_centroid, fc_cluster_emb_centroid])
            fc_clusters_embedding_centroid = fc_clusters_embedding_centroid.unsqueeze(0)
            fc_clusters_emb_centroid = torch.concat([fc_clusters_emb_centroid, fc_clusters_embedding_centroid])

        print("fc_clusters_emb_centroid", fc_clusters_emb_centroid.size())
        # print(fc_clusters_emb_centroid)
        most_similarity_indices = []
        for id, clusters, fc_clusters_embedding_centroid in zip(ids, clusters_emb, fc_clusters_emb_centroid):
            most_similar_fc_clusters_embedding_centroid = torch.Tensor([]).to(passage_emb)
            clusters_embedding_centroid = torch.Tensor([]).to(passage_emb)
            for cluster_emb in clusters:
                cluster_emb_centroid = torch.mean(cluster_emb, dim=0).unsqueeze(0)
                clusters_embedding_centroid = torch.concat([clusters_embedding_centroid, cluster_emb_centroid])
                print("cluster_emb_centroid", cluster_emb_centroid.size())
                print("fc_clusters_embedding_centroid", fc_clusters_embedding_centroid.unsqueeze(0).size())
                similarity_scores = torch.einsum('id, ijd->ij', [cluster_emb_centroid, fc_clusters_embedding_centroid.unsqueeze(0)])

                # METHOD 1
                # top_similarity_scores, similarity_inds = torch.topk(similarity_scores[0], 1, dim=0)
                # most_similar_fc_cluster = [fc_clusters_embedding_centroid[j] for j in similarity_inds][0]

                # METHOD 2
                top_similarity_score, similarity_idx = torch.max(similarity_scores[0], dim=0)
                most_similarity_indices += [similarity_idx]
                most_similar_fc_cluster = fc_clusters_embedding_centroid[similarity_idx].unsqueeze(0)
                most_similar_fc_clusters_embedding_centroid = torch.concat([most_similar_fc_clusters_embedding_centroid,
                                                                            most_similar_fc_cluster])

            clusters_embedding_centroid = clusters_embedding_centroid.unsqueeze(0)
            clusters_emb_centroid = torch.concat([clusters_emb_centroid, clusters_embedding_centroid])

            most_similar_fc_clusters_embedding_centroid = most_similar_fc_clusters_embedding_centroid.unsqueeze(0)
            most_similar_fc_clusters_emb_centroid = torch.concat([most_similar_fc_clusters_emb_centroid, most_similar_fc_clusters_embedding_centroid])

        loss = []
        for clusters_embedding_centroid, most_similar_fc_clusters_embedding_centroid in zip(clusters_emb_centroid, most_similar_fc_clusters_emb_centroid):
            for cluster_emb_centroid, most_similar_fc_cluster in zip(clusters_embedding_centroid, most_similar_fc_clusters_embedding_centroid):
                loss += [self.mse_loss(cluster_emb_centroid, most_similar_fc_cluster)]
                print("mse loss", loss)
        # print("clusters_emb_centroid", clusters_emb_centroid.size())
        # print("most_similar_fc_clusters_emb_centroid", most_similar_fc_clusters_emb_centroid.size())
        # loss = self.mse_loss(clusters_emb_centroid, most_similar_fc_clusters_emb_centroid)
        return loss, most_similarity_indices
            #     top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, len(passages[0]), dim=1)
            #     for i in range(bsz):
            #         output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
            #         output_scores.append(top_retriever_scores[i].tolist())
            #
            #
            #     fc_passage_emb = torch.mean(fc_split_passage_emb, dim=0).unsqueeze(0)
            #
            #     emb = self.retriever(**cluster_tok, is_passages=True).to(query_emb)
            #     clusters_embedding += [emb]
            # clusters_emb += [clusters_embedding]

    def cluster_average_similarity_to_query(self, single_cluster, query):
        cluster_tok = self.retriever_tokenizer(single_cluster, padding=True, truncation=True, return_tensors="pt")
        cluster_tok = {k: v.cuda() for k, v in cluster_tok.items()}
        # TODO: Focus here
        with torch.no_grad():
            passage_embeddings = self.retriever(**cluster_tok, is_passages=True)
        #     with torch.no_grad():
        #         passage_embeddings = model.retriever.passage_contriever(**cluster_tok)

        query_enc = self.retriever_tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        query_enc = {k: v.cuda() for k, v in query_enc.items()}
        # TODO: Focus here
        with torch.no_grad():
            query_embeddings = self.retriever.contriever(**query_enc)
            # query_embeddings = self.retriever(**cluster_tok, is_passages=False)
            # query_embeddings = self.retriever.query_contriever(**query_enc)

            passage_embeddings_similarity = []
            for pe in passage_embeddings:
                passage_embeddings_similarity += [float(query_embeddings[0] @ pe)]

        return statistics.mean(passage_embeddings_similarity)

    def cluster_retrieved_passages(self, query, sentences):
        if len(sentences) == 0:
            return [], []

        inputs = self.retriever_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        # TODO: Focus here
        with torch.no_grad():
            embeddings = self.retriever(**inputs, is_passages=True)
        #     with torch.no_grad():
        #         embeddings = model.retriever.passage_contriever(**inputs)

            clusters = util.deduplicate(sentences, embeddings, self.opt.clustering_threshold)

            filtered_clusters = []
            filtered_mean_similarity = []
            for single_cluster in clusters:
                #         _, _, _, mean_similarity_to_query = cluster_average_similarity_to_query(
                mean_similarity_to_query = self.cluster_average_similarity_to_query(
                    single_cluster,
                    query
                )
                if mean_similarity_to_query >= self.opt.mean_cluster_similarity_to_query_threshold and \
                        len(single_cluster) >= self.opt.min_cluster_size:
                    filtered_clusters += [single_cluster]
                    filtered_mean_similarity += [mean_similarity_to_query]

        return filtered_clusters, filtered_mean_similarity

    def forward(
        self,
        ids,
        asin,
        index,
        query,
        target,
        cls_label,
        target_tokens=None,
        passages=None,
        batch_metadata=None,
        filtering_fun=None,
        use_cache=False,
        train_retriever=False,
        iter_stats={},
    ):
        forward_start = time.time()
        bsz = len(query)
        if not self.opt.mt:
            del cls_label

        # query_mask_reader = (
        #     self.reader_tokenizer.batch_encode_plus(
        #         query,
        #         max_length=self.opt.text_maxlength,
        #         padding="longest",
        #         truncation=True,
        #         return_tensors="pt",
        #         add_special_tokens=False,
        #     )["attention_mask"]
        #     .bool()
        #     .cuda()
        # )
        # print("HERERERERERER")
        # print(query, target, target_tokens)
        query_enc = self.tokenize(query, target, target_tokens)
        fc_only = self.opt.fc_only

        # CLARIFY: DISTILLATION
        comment_clusters = []
        if (not self.opt.use_file_passages) and (not fc_only):
            retrieve_start = time.time()
            passages, _ = self.retrieve(
                ids,
                index,
                self.opt.retriever_n_context,
                query,
                query_enc["input_ids"],
                query_enc["attention_mask"],
                is_train=True,
                batch_metadata=batch_metadata,
                filtering_fun=filtering_fun,
                iter_stats=iter_stats,
            )

            comment_clusters = []
            for q, pas in zip(query, passages):
                filtered_clusters, _ = self.cluster_retrieved_passages(q, [p['text'] for p in pas])
                filtered_clusters = [{'cluster_id': i, 'comments': c, 'cluster_size': len(c)} for i, c in enumerate(filtered_clusters)]
                # filtered_clusters = [{'comments': c, 'cluster_size': len(c)} for c in filtered_clusters]
                print("FILTERED CLUSTER SIZE: ", len(filtered_clusters))
                comment_clusters += [filtered_clusters]

            print("IDS", ids[0])
            print("NUM OF PASSAGES BEFORE FILTER", len(passages[0]))
            # passages[0] = tuple(pas for pas in passages[0] if pas['id'] == int(ids[0]))
            passages[0] = tuple(pas for pas in passages[0] if pas['asin'] == asin[0])
            # print(passages)
            print("NUM OF PASSAGES AFTER FILTER", len(passages[0]))
            iter_stats["runtime/retrieve"] = (time.time() - retrieve_start, 1)

        # CLARIFY: BACKBONE TRAINING RUNS THIS, BUT QUITE DIFFERENT FROM THE PAPER
        # AS THEY USED FC ARTICLE NOT INITIAL PASSAGE FOR TRAINING
        if fc_only:
            search_start = time.time()
            passage_tmp = []
            comment_clusters_tmp = []
            for idx,id in enumerate(ids):
                split_passages = util.split_fc_pas(id, self.all_fc[id])
                # # 20 is the amount of empty token per additional passage to compensate in case of insufficiency
                # tmp_text = [" ".join(["[MASK]" for _ in range(100)])]
                # tmp_text_split = [" ".join(tmp_text[i: i + 100]).strip() for i in range(0, len(tmp_text), 100)]
                # # tmp_text = [" ".join(["[MASK]" for _ in range(20)])]
                # # tmp_text_split = [" ".join(tmp_text[i : i + 20]).strip() for i in range(0, len(tmp_text), 20)]
                # padding_pas = {'id':-1,'title':"","text":tmp_text_split[0]}
                # split_passages = split_passages + [padding_pas for p in range(self.opt.retriever_n_context)]
                # # passage_tmp.append(split_passages[:self.opt.retriever_n_context])
                passage_tmp.append(split_passages)
                comment_clusters_tmp.append(self.all_cluster[id])  # Get ground truth clusters

            passages = passage_tmp
            comment_clusters = comment_clusters_tmp
            iter_stats["runtime/match"] = (time.time() - search_start, 1)

        # TODO: Edit tokenize_passages to encode comment_clusters instead of passages
        # TODO: Replace tokenize_passages with tokenize_clusters
        clusters_tokens = self.tokenize_clusters(query, comment_clusters)
        # print(cluster_tokens)
        retriever_tokens = self.tokenize_passages(query, passages)
        print(target)
        # print(comment_clusters)

        if fc_only:  # FOR WARMUP ONLY
            reader_tokens_kp_list = self.tokenize_reader_prompt(query, comment_clusters, target)
            print("OK 1")
            if self.opt.use_gradient_checkpoint_reader:
                self.reader.gradient_checkpointing_enable()
            print("OK 2")
            loss_values = []
            reader_output = []
            reader_loss = []
            if len(reader_tokens_kp_list) > 0:
                for reader_tokens in reader_tokens_kp_list[0]:
                    print(reader_tokens.keys())
                    single_kp_output = self.reader(**reader_tokens)
                    reader_output += [single_kp_output]
                    single_kp_reader_loss = single_kp_output.loss
                    print("SINGLE KP LOSS", single_kp_reader_loss)
                    loss_values.append(single_kp_reader_loss)
                reader_loss = loss_values
            print("OK 3")

        # print("TOTAL CONTEXT", reader_tokens_kp_list[0][0]["input_ids"].size(1))
        retriever_loss = []
        # CLARIFY: DISTILLATION FOR RETRIEVER, HERE WE FOCUS ON lgret (ARTICLE-LEVEL RETRIEVAL DISTILLATION)
        if train_retriever:

            if self.opt.use_gradient_checkpoint_retriever:
                self.retriever.gradient_checkpointing_enable()

            query_emb = self.retriever(**query_enc, is_passages=False)

            if "std" in self.opt.gold_score_mode:
                retriever_tokens = {k: v[:, :n_context_training] for k, v in retriever_tokens.items()}
            retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}

            # Get the embedding of the predicted retrieved clusters
            clusters_emb = []
            for clusters in clusters_tokens:
                clusters_embedding = []
                for cluster_tok in clusters:
                    emb = self.retriever(**cluster_tok, is_passages=True).to(query_emb)
                    clusters_embedding += [emb]
                clusters_emb += [clusters_embedding]

            # Get the embedding of the predicted retrieved passages
            with torch.no_grad():
                passage_emb = self.retriever(**retriever_tokens, is_passages=True).to(query_emb)
            passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
            # retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
            cluster_score = []
            for clusters_embedding in clusters_emb:
                for emb in clusters_embedding:
                    emb = emb.view(bsz, -1, emb.size(-1))
                    cluster_score += [torch.einsum("id, ijd->ij", [query_emb, emb]) / np.sqrt(query_emb.size(-1))]

            loss_lgret = []
            if self.opt.lgret:
                retrieval_loss_lgret= self.lgret_score(ids,passage_emb)
                loss_lgret_cluster, most_similarity_indices = self.lgret_score_cluster(ids,clusters_emb,passage_emb)
                # if not torch.isnan(loss_lgret_cluster):
                #     loss_lgret = loss_lgret + loss_lgret_cluster

                for i, each_loss_lgret_cluster in enumerate(loss_lgret_cluster):
                    if i == 0:
                        loss_lgret += [retrieval_loss_lgret + each_loss_lgret_cluster]
                    else:
                        loss_lgret += [each_loss_lgret_cluster]

                # loss_lgret = self.lgret_score_cluster(ids,clusters_emb,passage_emb)

            # READER LOSS SPECIFICALLY ON KPS PRODUCED BY PREDICTED CLUSTERS
            # most_similar_bullet_target = [[bullet for i, bullet in enumerate(t.split("\n")) if i in most_similarity_indices] for t in target]
            most_similar_bullet_target = [[t.split("\n")[index] for index in most_similarity_indices] for t in target]
            print("MOST SIMILAR INDICE", most_similarity_indices)
            print("MOST SIMILAR BULLET", most_similar_bullet_target)
            print("SUMMARY LENGTH", len(target[0].split("\n")))
            assert len(most_similar_bullet_target[0]) == len(most_similarity_indices)
            most_similar_bullet_target = ["\n".join(t) for t in most_similar_bullet_target]
            reader_tokens_kp_list = self.tokenize_reader_prompt(query, comment_clusters, most_similar_bullet_target)
            print("OK 1 ENHANCED")
            if self.opt.use_gradient_checkpoint_reader:
                self.reader.gradient_checkpointing_enable()
            print("OK 2 ENHANCED")
            loss_values = []
            reader_output = []
            reader_loss = []
            if len(reader_tokens_kp_list[0]) > 0:
                print("reader_tokens_kp_list", reader_tokens_kp_list)
                for reader_tokens in reader_tokens_kp_list[0]:
                    print(reader_tokens.keys())
                    single_kp_output = self.reader(**reader_tokens)
                    reader_output += [single_kp_output]
                    single_kp_reader_loss = single_kp_output.loss
                    print("SINGLE KP LOSS", single_kp_reader_loss)
                    loss_values.append(single_kp_reader_loss)
                reader_loss = loss_values
            print("OK 3 ENHANCED")

            if "eval" in self.opt.gold_score_mode:
                gold_score = self.eval_score(
                    reader_ids,
                    reader_mask,
                    decoder_input_ids,
                    labels,
                    cfg,
                    bsz,
                    query_mask_reader,
                )
            elif "loop" in self.opt.gold_score_mode:
                gold_score = self.loop_score(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)
            elif "ppmean" in self.opt.gold_score_mode:
                # Ignore, as perplexity is not part of the training objective for decode-only LLM like Mistral
                # gold_score = reader_loss # TODO Improve here, incorporate supervisory signal from reader for retriever_losss
                # gold_score = self.perplexity_score(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)
                # gold_score = None
                gold_score_total = []
                if len(reader_tokens_kp_list) > 0:
                    for single_kp_output, single_kp_reader_tokens in zip(reader_output, reader_tokens_kp_list[0]):
                        gold_score_each_kp = self.perplexity_score(single_kp_output, single_kp_reader_tokens, bsz)
                        gold_score_total += [gold_score_each_kp]

                if len(gold_score_total) > 0:
                    gold_score = gold_score_total
                #     # Stack tensors into a single tensor
                #     stacked_tensor = torch.stack(gold_score_total)  # Shape: (2, 1)
                #
                #     # Compute the mean
                #     gold_score = stacked_tensor.mean()  # Mean of all elements
                #     # gold_score = statistics.mean(gold_score_total)
                else:
                    gold_score = None
            elif "emdr" in self.opt.gold_score_mode:
                gold_score = self.emdr_score(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)

            loss_lcret = None
            if self.opt.lcret:
                tmp_id = ids[0]
                split_text = util.split_fc_text(self.all_fc[tmp_id])
                all_fc_score = None

                for st in split_text:
                    fc_enc, fc_labels, fc_decoder_input_ids = self.tokenize([st], [st], target_tokens)
                    fc_score = self.perplexity_score(reader_ids, reader_mask, fc_decoder_input_ids, fc_labels, cfg, bsz)
                    if all_fc_score is None:
                        all_fc_score = fc_score
                    else:
                        all_fc_score = torch.cat((all_fc_score,fc_score))

                fc_enc, _, _ = self.tokenize(split_text, [st], target_tokens)
                fc_embs = self.retriever(**fc_enc, is_passages=False)
                fc_retriever_score = torch.mm(fc_embs,passage_emb.squeeze(0).transpose(0,1))
                topk_v,topk_idx = fc_retriever_score.topk(1)
                fc_retriever_score = topk_v.transpose(0,1)
                topk_all_fc_score = torch.gather(all_fc_score,dim=1,index=topk_idx).transpose(0,1)
                all_fc_score = topk_all_fc_score
                fc_retriever_score = fc_retriever_score / np.sqrt(query_emb.size(-1))
                fc_retriever_score = fc_retriever_score.float()
                loss_lcret =self.kldivloss(fc_retriever_score, all_fc_score)

            if self.opt.use_gradient_checkpoint_retriever:
                self.retriever.gradient_checkpointing_disable()

            # self.reader.reset_score_storage()

            if self.training:
                self.reader.train()


        if self.opt.lclm:
            docs_crossattention_scores = self.reader.get_crossattention_scores_lm(
                n_context_training,
                reader_mask_training.cuda(),
                ids=reader_ids_training.cuda(),
                mask_query=query_mask_reader.cuda(),
                labels=labels,
                mode="all",
            )
            docs_gold_scores = select_crossattention_scores(docs_crossattention_scores, "attn")

        # loss_lclm = None
        if self.opt.lclm:
            passage_tmp = []
            for idx,id in enumerate(ids):
                split_passages = util.split_fc_pas(id, self.all_fc[id])
                passage_tmp.append(split_passages[:self.opt.retriever_n_context])
                split_text = util.split_fc_text(self.all_fc[id])

            fc_tokens, _ = self.tokenize_passages(query, passage_tmp)
            fc_ids = fc_tokens["input_ids"]  # FIXME
            fc_mask = fc_tokens["attention_mask"].bool()

            n_context_training = fc_ids.size(1)
            # n_context_training = min(self.opt.n_context, fc_ids.size(1))
            fc_ids_training = fc_ids[:, :n_context_training].contiguous()
            fc_mask_training = fc_mask[:, :n_context_training].contiguous()

            fc_ids_training = fc_ids_training.view(fc_ids.size(0), -1)
            fc_mask_training = fc_mask_training.view(fc_mask.size(0), -1)

            fc_gold_score = self.eval_score(
                    fc_ids,
                    fc_mask,
                    decoder_input_ids,
                    labels,
                    cfg,
                    bsz,
                    query_mask_reader,
                )

            with torch.no_grad():
                fc_enc, _, _ = self.tokenize(split_text, [""], target_tokens)
                fc_embs = self.retriever(**fc_enc, is_passages=False)
                fc_retriever_score = torch.mm(fc_embs,passage_emb.squeeze(0).transpose(0,1))
                fc_retriever_score = fc_retriever_score.transpose(0,1)
                topk_v,topk_idx = fc_retriever_score.topk(1)

            all_fc_score = torch.zeros_like(docs_gold_scores)
            for ii,idx in enumerate(topk_idx):
                all_fc_score[0,ii] = fc_gold_score[0][idx.item()]
            loss_lclm = self.kldivloss(docs_gold_scores, all_fc_score)

        # mt_loss= None
        if self.opt.mt:
            choices_ids = torch.stack([
                self.reader_tokenizer(
                    answer_choice, return_tensors="pt", truncation=True, add_special_tokens=False
                ).input_ids.squeeze(0)
                for answer_choice in ["false","unclear","true"]
            ])
            flat_choices_ids = choices_ids.flatten(0, 1).unsqueeze(0)
            decoder_choices_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.reader_tokenizer.pad_token_id).long()
            lm_target = lm_target.to(reader_ids_training.device)

            model_output = self.reader(
                input_ids=reader_ids_training,
                attention_mask=reader_mask_training,
                decoder_input_ids=decoder_choices_ids.to(reader_ids_training.device),
                labels=lm_target,
                use_cache=False,
            )
            choices_scores = (
                self.cls_loss(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1))
                .view(1, 3, -1)
                .sum(dim=-1)
            )
            cls_label = torch.LongTensor(cls_label).to(choices_scores.device)
            mt_loss = self.cls_loss(-choices_scores, cls_label).mean(dim=-1)

        # CLARIFY: DISTILLATION FOR lm, HERE WE FOCUS ON lglm (ARTICLE-LEVEL GENERATION DISTILLATION)
        loss_lglm = None
        if self.opt.lglm:
            # reader_logits = reader_output.logits
            all_logits = []
            for single_kp_output in reader_output:
                logits = single_kp_output["logits"]  # Assuming logits is a tensor
                all_logits.append(logits)
            # Concatenate all logits along a new dimension if needed
            if all_logits:
                concatenated_logits = torch.cat(all_logits, dim=0)  # Concatenate along batch dimension
                reader_logits = torch.mean(concatenated_logits)  # Compute mean of all logits
                print(f"Mean Logit: {reader_logits.item()}")
            else:
                print("No logits were collected.")

            passage_tmp = []
            comment_clusters_tmp = []
            for idx,id in enumerate(ids):
                split_passages = util.split_fc_pas(id, self.all_fc[id])
                # passage_tmp.append(split_passages[:self.opt.retriever_n_context])
                passage_tmp.append(split_passages)
                comment_clusters_tmp.append(self.all_cluster[id])  # Get ground truth clusters

            fc_tokens_kp_list = self.tokenize_reader_prompt(query, comment_clusters_tmp, target)

            # fc_ids = fc_tokens["input_ids"]  # FIXME
            # fc_mask = fc_tokens["attention_mask"].bool()
            # n_context_training = fc_ids.size(1)
            # # n_context_training = min(self.opt.n_context, fc_ids.size(1))
            # fc_ids_training = fc_ids[:, :n_context_training].contiguous()
            # fc_mask_training = fc_mask[:, :n_context_training].contiguous()
            #
            # fc_ids_training = fc_ids_training.view(fc_ids.size(0), -1)
            # fc_mask_training = fc_mask_training.view(fc_mask.size(0), -1)

            # fc_output = self.reader(
            #     input_ids=fc_ids_training,
            #     attention_mask=fc_mask_training,
            #     decoder_input_ids=decoder_input_ids,
            #     labels=labels,
            #     use_cache=False,
            # )

            loss_lglm = None
            if reader_output:
                fc_logits_values = []
                fc_output = []
                fc_logits = 0
                if len(fc_tokens_kp_list) > 0:
                    for fc_tokens in fc_tokens_kp_list[0]:
                        single_fc_kp_output = self.reader(**fc_tokens)
                        fc_output += [single_fc_kp_output]
                        single_kp_reader_logits = single_fc_kp_output.logits
                        fc_logits_values.append(single_kp_reader_logits)

                    # Calculate the mean loss
                    if fc_logits_values:  # Ensure the list is not empty
                        concatenated_logits = torch.cat(fc_logits_values, dim=0)  # Concatenate along batch dimension
                        fc_logits = torch.mean(concatenated_logits)  # Compute mean of all logits
                        print(f"Mean FC Logit: {fc_logits.item()}")
                    else:
                        print("No logits were collected.")

                # fc_output = self.reader(**fc_tokens)
                # fc_logits = fc_output.logits
                # loss_lglm = self.cross_entropy_loss(reader_logits.squeeze(0), fc_logits.squeeze(0))
                print("LOGITS")
                print(reader_logits.shape)
                print(fc_logits.shape)
                loss_lglm = self.mse_loss(reader_logits.squeeze(0), fc_logits.squeeze(0))

        if self.opt.use_gradient_checkpoint_reader:
            self.reader.gradient_checkpointing_disable()

        # COMPARES WITH GROUND TRUTH DURING DISTILLATION
        if train_retriever:
            if self.opt.compute_crossattention_stats or "std" in self.opt.gold_score_mode:
                crossattention_scores = self.reader.get_crossattention_scores(
                    n_context_training,
                    reader_mask_training.cuda(),
                    ids=reader_ids_training.cuda(),
                    mask_query=query_mask_reader.cuda(),
                    labels=labels,
                    mode="all",
                )
            if "std" in self.opt.gold_score_mode:
                gold_score = select_crossattention_scores(
                    crossattention_scores, self.opt.gold_score_mode
                ).detach()

            # retriever_score = retriever_score / np.sqrt(query_emb.size(-1))
            # print("RETRIEVER SCORE SIZE ", retriever_score.size())

            if self.opt.compute_crossattention_stats:
                with torch.no_grad():
                    for k, v in crossattention_scores.items():
                        corr = torch.corrcoef(torch.stack([gold_score.view(-1), v.view(-1)]))
                        corr = corr[0, 1].item()
                        if np.isnan(corr):
                            corr = 0.0
                        iter_stats[f"corr/{k}"] = (corr, len(query))

            if gold_score is not None:
                # retriever_score = retriever_score.float()
                for gold_score_each_kp, cluster_score_each_kp in zip(gold_score, cluster_score):
                    gold_score_each_kp = gold_score_each_kp.float()
                    retriever_loss_each_kp = self.kldivloss(cluster_score_each_kp, gold_score_each_kp)  # kldivloss loss
                    retriever_loss += [retriever_loss_each_kp]

                # gold_score = gold_score.float()
                # retriever_score = retriever_score.float()
                # if self.opt.gold_score_mode == "emdr":
                #     retriever_loss = self.logprob(retriever_score, gold_score, labels)
                # else:
                #     retriever_loss = self.kldivloss(retriever_score, gold_score)

        # self.reader.reset_score_storage()
        # CALCULATE THE FINAL DISTILLATION GENERATION LOSS,
        # NOTE THAT THEY ALSO COMBINE IT WITH NORMAL LOSS AGAINST GROUND TRUTH (PAPER DOES NOT DECLARE)
        # if self.opt.lglm: # HERE YES
        if self.opt.lglm and loss_lglm != None: # HERE YES
            loss_lglm = self.opt.lglm_cof*loss_lglm
            print("LOSS LGLM", loss_lglm)
            print("READER LOSS", reader_loss)
            reader_loss = reader_loss + loss_lglm
            iter_stats["loss/reader_loss_lglm"] = (loss_lglm.item(), len(query))
        if self.opt.lclm:
            loss_lclm = self.opt.lclm_cof*loss_lclm
            reader_loss = reader_loss + loss_lclm
            iter_stats["loss/reader_loss_lclm"] = (loss_lclm.item(), len(query))

        if self.opt.mt:
            mt_loss = self.opt.mt_cof*mt_loss
            reader_loss = (1-self.opt.mt_cof)*reader_loss + mt_loss
            iter_stats["loss/reader_mt_loss"] = (mt_loss.item(), len(query))

        print("READER LOSS", reader_loss)
        # iter_stats["loss/reader_loss"] = (reader_loss.item(), len(query))

        # if retriever_loss is not None:
        if len(retriever_loss) > 0:
            # if loss_lgret is not None:
            if loss_lgret is not None:
                if len(loss_lgret) > 0 and len(loss_lgret) == len(retriever_loss):
                    updated_retriever_loss = []
                    for retriever_loss_each_kp, each_loss_lgret_cluster in zip(retriever_loss, loss_lgret):
                        updated_retriever_loss += [retriever_loss_each_kp + each_loss_lgret_cluster]
                    retriever_loss = updated_retriever_loss
                    print("UPDATED lgret to retriever_loss", retriever_loss)
                else:
                    retriever_loss = [retrieval_loss_lgret]
                    # loss_lgret = [retrieval_loss_lgret]
                # retriever_loss = retriever_loss + loss_lgret
                # iter_stats["loss/loss_lgret"] = (loss_lgret.item(), len(query))

            # if loss_lcret is not None:
            #     retriever_loss = retriever_loss + loss_lcret
            #     iter_stats["loss/loss_lcret"] = (loss_lcret.item(), len(query))

            # iter_stats["loss/retriever_loss"] = (retriever_loss.item(), len(query))

        iter_stats["runtime/forward"] = (time.time() - forward_start, 1)
        # if train_retriever and retriever_loss is None:
        #     retriever_loss = loss_lgret
        print("RETRIEVER LOSS", retriever_loss)
        return reader_loss, retriever_loss

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)

    def logprob(self, score, gold_score, labels):
        with torch.no_grad():
            repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
            repeated_labels[repeated_labels == IGNORE_INDEX] = 0

            mask_labels = labels >= 0

            gold_log_prob = torch.nn.functional.log_softmax(gold_score / self.opt.temperature_gold, dim=-1)
            gold_log_probs = torch.gather(gold_log_prob, dim=-1, index=repeated_labels[..., None]).view(
                gold_log_prob.size(0), -1
            )
            gold_log_probs = gold_log_probs.view(score.size(0), score.size(1), -1)

        log_score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        log_prob = gold_log_probs + log_score[..., None]
        logsumprobs = torch.logsumexp(log_prob, dim=1)
        loss = -1 * torch.sum(logsumprobs * mask_labels) / torch.sum(mask_labels)

        return loss

    @torch.no_grad()
    def compute_reader_loss_and_logits(self, tokens):
    # def compute_reader_loss_and_logits(self, tokens, decoder_input_ids, labels):

        # cfg = self.reader.encoder.config
        # cfg.bsz = tokens["input_ids"].size(0)
        # cfg.n_context = tokens["input_ids"].size(1)
        # # cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))
        #
        # reader_loss = self.reader(
        #     input_ids=tokens["input_ids"].cuda().view(tokens["input_ids"].size(0), -1),
        #     attention_mask=tokens["attention_mask"].cuda().view(tokens["attention_mask"].size(0), -1),
        #     decoder_input_ids=decoder_input_ids.cuda(),
        #     labels=labels.cuda(),
        #     use_cache=False,
        # )
        reader_loss = self.reader(**tokens)
        return reader_loss[0].cpu().item(), reader_loss[1]

    @torch.no_grad()
    def generate(self, tokens, query, cls_label,choices=None):
        # cfg = self.reader.encoder.config
        # cfg.bsz = tokens["input_ids"].size(0)
        # cfg.n_context = tokens["input_ids"].size(1)
        # # cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        # tokens = {k: v.view(v.size(0), -1) for k, v in tokens.items()}

        bos_token_id = None

        prefix_allowed_tokens_fn = None
        if self.opt.decoder_prompt_format is not None:
            prefix_str = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
            prefix_allowed_tokens_fn = self.get_prefix_allowed_tokens_fn(prefix_str)

        outputs = self.reader.generate(
            inputs=tokens["input_ids"].cuda(),
            temperature=0,
            do_sample=False,
            # temperature=0.7,
            # do_sample=True,
            top_p=0.95,
            top_k=40,
            max_new_tokens=256
            # max_new_tokens=512
            # input_ids = tokens["input_ids"].cuda(),
            # attention_mask=tokens["attention_mask"].cuda(),
            # num_return_sequences=1,
            # max_length=self.opt.generation_max_length,
            # min_length=self.opt.generation_min_length,
            # num_beams=self.opt.generation_num_beams,
            # length_penalty=self.opt.generation_length_penalty,
            # forced_bos_token_id=bos_token_id,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        # print(outputs)

        prediction = None
        if self.opt.mt:
            choices_ids = torch.stack([
                self.reader_tokenizer(
                    answer_choice, return_tensors="pt", truncation=True, add_special_tokens=False
                ).input_ids.squeeze(0)
                for answer_choice in ["false","unclear","true"]
            ])
            flat_choices_ids = choices_ids.flatten(0, 1).unsqueeze(0)
            decoder_choices_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.reader_tokenizer.pad_token_id).long()
            lm_target = lm_target.to(tokens["input_ids"].cuda())

            model_output = self.reader(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                decoder_input_ids=decoder_choices_ids.to(tokens["input_ids"].cuda()),
                labels=lm_target,
                use_cache=False,
            )
            choices_scores = (
                self.cls_loss(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1))
                .view(1, 3, -1)
                .sum(dim=-1)
            )
            _, prediction = choices_scores.min(dim=1)

        return outputs,prediction,cls_label

    def get_prefix_allowed_tokens_fn(self, prefix_str: Optional[str] = None):
        if prefix_str:
            prefix_tokens_ids = self.reader_tokenizer.batch_encode_plus(prefix_str, add_special_tokens=False)[
                "input_ids"
            ]

            def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
                if input_ids.shape[-1] > len(prefix_tokens_ids[batch_id]):
                    return self.READER_ALL_TOKENS

                return prefix_tokens_ids[batch_id][input_ids.shape[-1] - 1]

        else:
            prefix_allowed_tokens_fn = None

        return prefix_allowed_tokens_fn


def select_crossattention_scores(scores, mode):
    if "eval" in mode or "attn" in mode or "ppmean" in mode :
        return scores["normssum"]
    elif "std" in mode:
        return scores[mode[len("std") :]]


def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}
