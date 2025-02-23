import argparse
import os
import sys

module_path = os.path.join("./KeyPoint-Analysis/KPG/")
sys.path.insert(0, module_path)
from rouge_setbase import preprocess_dataset, compute_rouge, compute_rouge_max

root_dir_pth = "../"
sys.path.insert(1, '../')
from utils.postprocessing import *
from utils.g_eval import *
import pandas as pd
import statistics
import openai
from multiprocessing import Pool
from softF1 import *
import tensorflow as tf


def parse_extracted_json(processed_df):
    processed_df['claim_split_predicted'] = processed_df['claim_split_predicted'].apply(
        lambda x: re.sub(r"\n+ *", "", x.replace("json", "").replace("`", "")))
    mask = processed_df['claim_split_predicted'].apply(
        lambda x: len(re.findall(r"(: *)\'((?:[^':]*\'+[^':,]*)+)\'( *)", x, re.DOTALL)) > 0)
    processed_df.loc[mask, 'claim_split_predicted'] = processed_df.loc[mask, 'claim_split_predicted'].apply(
        lambda x: re.sub(r"(: *)\'((?:[^':]*\'+[^':]*)+)\'( *,)", r'\1"""\2"""\3', x))
    processed_df['claim_split_predicted'] = processed_df['claim_split_predicted'].apply(extract_claims)

    return processed_df


def other_kps_same_query(grp):
    other_kps_list = []
    for i, row in grp.iterrows():
        my_kp = row['key_point']
        other_kps = [other_kp for other_kp in grp['key_point'] if other_kp != my_kp]
        other_kps_list += [other_kps]

    grp['key_point_given'] = other_kps_list

    return grp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qqsum_output_path", type=str, default='../output/atlas-xl-seed2-lgret-lglm',
                        help="Output directory of QQSUM-RAG")
    # parser.add_argument("--model", type=str, default='gpt-3.5-turbo',
    #                     help="The LLM of OpenAI used for KPG")
    parser.add_argument("--openai_api_key", type=str, required=True,
                        help="The API key, use for prompting OpenAI's endpoint models for KPG")
    parser.add_argument("--num_workers", type=int, default=1, help="The number of workers for the KPG task")

    args = parser.parse_args()
    qqsum_output_path = args.qqsum_output_path

    # Read predicted output file of QQSUM-RAG
    ground_truth_df = pd.read_pickle("../data/test/test.pkl")
    ground_truth_df = ground_truth_df.drop(columns=['reviews', 'reviewText', 'title', 'fact_check_article'])
    ground_truth_df['retrieved_relevant_sent_len'] = ground_truth_df['retrieved_relevant_sent'].str.len()

    predicted_df = pd.read_json(qqsum_output_path + "/test-result.jsonl", lines=True)
    predicted_df['passages'] = predicted_df['passages'].apply(lambda x: [pas['text'] for pas in x])
    predicted_df['passages_len'] = predicted_df['passages'].str.len()
    predicted_df = predicted_df[
        ['query', 'generation', 'passages', 'passages_scores', 'passages_len', 'comment_clusters', 'id']]
    df = ground_truth_df.merge(predicted_df, on=['query', 'id'])

    # Post-processing
    df['summary'] = df['generation'].apply(lambda x: re.findall("\[\/INST\] *((.+\n*)+)$", x)[0][0].replace("</s>", ""))
    df['summary'] = df['summary'].apply(lambda x: re.sub("(Therefore|Thus)(.+\n*)+$", "", x))
    df['final_summary'] = df['summary'].apply(lambda x: re.findall("(While[^\n]+\n+(\+ *[0-9]+[^\n]+\n*)+)", x))
    df['final_summary'] = df['final_summary'].apply(lambda x: [e[0] for e in x])
    df['final_summary_text'] = df['final_summary'].apply(lambda x: "\n\n".join(x[:1]))

    # Post-process generated KP summary into KPs
    df['my_category'] = 1
    num_workers = 1
    inputs = [(qqsum_output_path + "/post_processed_cache/rd",
               domain,
               df[df['my_category'] == domain].reset_index(drop=True)
               )
              for domain in df['my_category'].unique()]
    start_time = time.time()
    with Pool(num_workers) as processor:
        data = processor.starmap(prompted_claim_split_generation, inputs)
    print("TIME ELAPSED", time.time() - start_time)
    processed_df = pd.concat(data)

    processed_df = processed_df[processed_df['comment_clusters'].str.len() > 0]
    processed_df = processed_df[processed_df['final_summary'].str.len() > 0]
    processed_df = parse_extracted_json(processed_df)
    processed_df = processed_df.reset_index(drop=True)
    processed_df = processed_df.apply(match_claim_with_cluster, axis=1)
    processed_df = processed_df.rename(columns={'key_points': 'key_point_given'})

    claim_split_predicted = pd.json_normalize(
        processed_df.to_dict(orient='records'),
        "matching_comment_clusters", ["category", "asin", "id", "query", "passages", 'passages_len']
    )
    claim_split_predicted = claim_split_predicted.groupby(['asin', 'category', 'id', 'query']).filter(
        lambda grp: len(grp) > 1)
    claim_split_predicted = claim_split_predicted.groupby(['asin', 'category', 'id', 'query'], sort=False).apply(
        other_kps_same_query).reset_index(drop=True)
    merged_df = claim_split_predicted.explode(['key_point_given'])
    merged_df = merged_df.reset_index(drop=True)

    print("################################ KP QUALITY EVALUATION (Redundancy) ################################")
    # BERTScore
    softp_data = merged_df.groupby(['category', 'asin', 'id', 'query', 'key_point']) \
        .apply(lambda grp: "".join([cand + "=" for cand in (grp['key_point_given'].tolist())])).reset_index(
        name='multi_cands')
    cands, refs = preprocess_text(softp_data, metrics="softPrecision")
    P, R, F = bert_scorer.score(cands, refs)
    redundancy = P.mean()
    print("###", "BERTScore", "Redundancy (RD):", redundancy)

    # BARTScore
    softp_data = merged_df.groupby(['category', 'asin', 'id', 'query', 'key_point']) \
        .apply(lambda grp: "".join([cand + "=" for cand in (grp['key_point_given'].tolist())])).reset_index(
        name='multi_cands')
    cands, refs = preprocess_text(softp_data, metrics="softPrecision")
    refs = balance_ref_num(refs)
    P = bart_scorer.multi_ref_score(cands, refs, agg="max", batch_size=4)  # agg means aggregation, can be mean or max
    redundancy = math.tanh(math.exp((mean(P)) / 2 + 1.3))
    print("###", "BARTScore", "Redundancy (RD):", redundancy)

    # BLEURTScore
    softp_data = merged_df.sort_values(by=['category', 'asin', 'id', 'query', 'key_point'])
    df_compare_precision = softp_data[['key_point', 'key_point_given']].rename(
        columns={'key_point': 'candidate', 'key_point_given': 'reference'})
    candidates = df_compare_precision['candidate']
    references = df_compare_precision['reference']
    result = calculatingScore(references, candidates)
    df_compare_precision["BLEURT Score"] = result
    df_bestkp_pair_precision = df_compare_precision.loc[
        df_compare_precision.groupby(["candidate"])["BLEURT Score"].idxmax()]
    redundancy = df_bestkp_pair_precision["BLEURT Score"].mean()
    print("###", "BLEURTScore", "Redundancy (RD):", redundancy)

    # G-Eval
    num_workers = 1
    merged_df['my_category'] = 1
    inputs = [(qqsum_output_path + "/g_eval_cache/kp_redundancy",
               domain,
               merged_df[merged_df['my_category'] == domain].reset_index(drop=True)
               )
              for domain in merged_df['my_category'].unique()]
    start_time = time.time()
    with Pool(num_workers) as processor:
        data = processor.starmap(prompted_g_eval_kp_relevancy, inputs)
    print("TIME ELAPSED", time.time() - start_time)
    g_eval_df = pd.concat(data)
    g_eval_df['g_eval_scores'] = g_eval_df['g_eval_scores'].apply(process_g_eval)
    g_eval_df['g_eval_redundancy'] = g_eval_df['g_eval_scores'].apply(lambda x: statistics.mean(x))
    softp_data = g_eval_df.groupby(['category', 'asin', 'id', 'query', 'key_point'])[
        'g_eval_redundancy'].max().reset_index()
    redundancy = softp_data['g_eval_redundancy'].mean()
    redundancy = (redundancy - 1) / 4
    print("###", "G-Eval", "Redundancy (RD):", redundancy)
