import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *
import copy
import pandas as pd
import numpy as np
import ast
import time
from pathlib import Path
from os import listdir
from tqdm import tqdm
import re


def get_kp_from_summary(summ, model="gpt-4o-mini"):
    base_prompt = get_prompt("experiment_summary_post_process_kps")
    input_template = "\nNow perform the task on the following input:\nQuantitative Summary: %s\n"
    input_text = input_template % (summ)
    prompt = base_prompt + input_text

    retries = 5
    while retries > 0:
        try:
            response = get_completion(prompt, model)
            return response
        except Exception as e:
            if e:
                if "exceeded your current quota" in str(e).lower():
                    raise e
                print(e)
                print('Timeout error, retrying...')
                retries -= 1
                if "limit reached for" in str(e).lower():
                    time.sleep(30)
                else:
                    time.sleep(5)
            else:
                raise e

    print('API is not responding, moving on...')
    return None


def prompted_claim_split_generation(root_path, domain, domain_df, save_step=10):
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    claim_split_predicted_list = []

    file_names = listdir(src_path)
    postfix = [re.split("[_.]", name)[1]
               for name in listdir(src_path)
               ]
    start = 0
    if 'done' in postfix:
        print(domain, ": ", "Loaded saved file. Done")
        new_domain_df = pd.read_pickle(f"{src_path}/{domain}_done.pkl")
        return new_domain_df
    elif len(postfix) > 0:
        last_index = max([int(idx) for idx in postfix if idx != 'done'])
        last_domain_df = pd.read_pickle(f"{src_path}/{domain}_{last_index}.pkl")
        claim_split_predicted_list = last_domain_df['claim_split_predicted'].tolist()
        start = last_index
        print(domain, "Loaded saved file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        predicted_summary = row['final_summary_text']
        if len(row['comment_clusters']) == 0:
            claim_split_predicted_list += [np.nan]
        else:
            claim_split_predicted = get_kp_from_summary(predicted_summary)
            claim_split_predicted_list += [claim_split_predicted]
            time.sleep(0.1)

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'claim_split_predicted', claim_split_predicted_list)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'claim_split_predicted', claim_split_predicted_list)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df


def extract_claims(claim_split_response):
    if pd.notnull(claim_split_response):
        return ast.literal_eval(claim_split_response)
    else:
        None


def match_claim_with_cluster(row):
    claim_split_predicted = row['claim_split_predicted']
    comment_clusters = row['comment_clusters']
    matching_comment_clusters = []
    for idx, claim in enumerate(claim_split_predicted):
        updated_comment_clusters = copy.copy(comment_clusters)
        claim['prevalence'] = re.findall(r"[0-9]+", claim['prevalence'])[0]
        if idx == 0 and len(comment_clusters) >= 2 and int(claim['prevalence']) == comment_clusters[0]['cluster_size'] + comment_clusters[1]['cluster_size']:
            matching_cluster = copy.copy(comment_clusters[0])
            matching_cluster['comments'] += comment_clusters[1]['comments']
            matching_cluster['cluster_size'] += comment_clusters[1]['cluster_size']

            matching_cluster['key_point'] = claim['key_point']
            matching_cluster['prevalence'] = int(claim['prevalence'])
            matching_comment_clusters += [matching_cluster]
            updated_comment_clusters = updated_comment_clusters[2:]
            comment_clusters = updated_comment_clusters
        else:
            for cluster in comment_clusters:
                diff = abs(int(claim['prevalence']) - int(cluster['cluster_size'])) / int(cluster['cluster_size'])
                if int(claim['prevalence']) <= int(cluster['cluster_size']) and diff <= 0.1:
                    matching_cluster = copy.copy(cluster)
                    matching_cluster['key_point'] = claim['key_point']
                    matching_cluster['prevalence'] = int(claim['prevalence'])
                    matching_comment_clusters += [matching_cluster]
                    updated_comment_clusters = updated_comment_clusters[1:]
                    break
            comment_clusters = updated_comment_clusters
    row['matching_comment_clusters'] = matching_comment_clusters
    return row


def post_process_kps(claim_split_predicted):
    claim_split_predicted = claim_split_predicted.groupby(['category', 'asin', 'id', 'query']).\
        apply(lambda grp: grp.sort_values(by=['prevalence']).head(2))
    claim_split_predicted = claim_split_predicted.reset_index(drop=True)
    mask = claim_split_predicted['key_point'].apply(lambda x: len(x.split())) < 10
    claim_split_predicted = claim_split_predicted[mask]
    return claim_split_predicted