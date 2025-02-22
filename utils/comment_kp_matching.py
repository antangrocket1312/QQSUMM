import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *


def get_comment_kp_annotation_completion(query, key_point, comments, model="gpt-4o-mini"):
    base_prompt = get_prompt("experiment_comment_kp_matching")
    input_template = "\nQuery: \"\"\"{query}\"\"\"\nKey Point: \"\"\"{key_point}\"\"\"\nReview Sentence: \"\"\"{review_sentence}\"\"\"\n"
    input_text = input_template.format(query=query, key_point=key_point, review_sentence=comments)
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


def prompted_comment_kp_annotation(root_path, domain, domain_df, save_step=100):
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    entailments = []

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
        entailments = last_domain_df['matching_response'].tolist()
        start = last_index
        print(domain, "Loaded saved file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        query = row['query']
        key_point = row['key_point']
        comments = row['passages']

        entailment = get_comment_kp_annotation_completion(query, key_point, comments)
        entailments += [entailment]

        time.sleep(0.1)

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'matching_response', entailments)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'matching_response', entailments)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df