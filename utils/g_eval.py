import os
import sys
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *


def g_eval(prompt, source, system_output, model):
    cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
    ct, ignore = 0, 0
    while True:
        try:
            _response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": cur_prompt}],
                temperature=2,
                max_tokens=10,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=20
            )
            time.sleep(0.5)

            all_responses = [_response['choices'][i]['message']['content'] for i in
                             range(len(_response['choices']))]
            ct += 1
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
            else:
                ignore += 1
                print('ignored', ignore)

                break

    return all_responses


def g_eval_relevancy(source, system_output, model="gpt-4-0613"):
    prompt = get_prompt("experiment_g_eval_relevancy")
    return g_eval(prompt, source, system_output, model)


def g_eval_redundancy(source, system_output, model="gpt-4-0613"):
    prompt = get_prompt("experiment_g_eval_redundancy")
    return g_eval(prompt, source, system_output, model)


def prompted_g_eval_kp_relevancy(root_path, domain, domain_df, save_step=10):
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    g_eval_scores = []

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
        g_eval_scores = last_domain_df['g_eval_scores'].tolist()
        start = last_index
        print(domain, "Loaded saved file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        g_eval_scores += [g_eval_relevancy(row['key_point_given'], row['key_point'])]

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'g_eval_scores', g_eval_scores)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'g_eval_scores', g_eval_scores)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df


def prompted_g_eval_kp_redundancy(root_path, domain, domain_df, save_step=10):
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    g_eval_scores = []

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
        g_eval_scores = last_domain_df['g_eval_scores'].tolist()
        start = last_index
        print(domain, "Loaded saved file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        g_eval_scores += [g_eval_redundancy(row['key_point_given'], row['key_point'])]

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'g_eval_scores', g_eval_scores)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'g_eval_scores', g_eval_scores)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df


def process_g_eval(g_eval_annotation):
    g_eval_scores = []
    for annotation in g_eval_annotation:
        score_find = re.findall("[0-9]", annotation)
        if len(score_find) > 0 and 1 <= int(score_find[0]) <= 5:
            g_eval_scores += [int(score_find[0])]

    return g_eval_scores