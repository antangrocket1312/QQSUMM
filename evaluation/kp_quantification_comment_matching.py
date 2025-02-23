import argparse
from multiprocessing import Pool
import openai
import sys
sys.path.insert(1, '../')
from utils.postprocessing import *
from utils.comment_kp_matching import *


def parse_extracted_json(processed_df):
    processed_df['claim_split_predicted'] = processed_df['claim_split_predicted'].apply(
        lambda x: re.sub(r"\n+ *", "", x.replace("json", "").replace("`", "")))
    mask = processed_df['claim_split_predicted'].apply(
        lambda x: len(re.findall(r"(: *)\'((?:[^':]*\'+[^':,]*)+)\'( *)", x, re.DOTALL)) > 0)
    processed_df.loc[mask, 'claim_split_predicted'] = processed_df.loc[mask, 'claim_split_predicted'].apply(
        lambda x: re.sub(r"(: *)\'((?:[^':]*\'+[^':]*)+)\'( *,)", r'\1"""\2"""\3', x))
    processed_df['claim_split_predicted'] = processed_df['claim_split_predicted'].apply(extract_claims)

    return processed_df


def entailment_detail_labels(response_text):
    if "not at all" in response_text.lower():
        return "not at all"
    elif "somewhat not well" in response_text.lower():
        return "somewhat not well"
    elif "somewhat well" in response_text.lower():
        return "somewhat well"
    elif "very well" in response_text.lower():
        return "very well"


def entailment_labels(response_text):
    if "not at all" in response_text.lower():
        return 0
    elif "somewhat not well" in response_text.lower():
        return 0
    elif "somewhat well" in response_text.lower():
        return 1
    elif "very well" in response_text.lower():
        return 1


def calculate_precision(grp):
    if pd.isna(grp['matching_label'].iloc[0]):
        return None

    precision = len(grp[grp['matching_label'] == 1]) / len(grp)
    return pd.DataFrame({'precision': [precision]})


def calculate_recall(grp):
    recall = len(grp[grp['predicted_label'] == 1]) / len(grp)
    return pd.DataFrame({'recall': [recall]})


def calculate_precision_prevalence(grp):
    if pd.isna(grp['matching_label'].iloc[0]):
        return None
    return pd.DataFrame({'correct_prevalence': [len(grp[grp['matching_label'] == 1])]})


def calculate_recall_prevalence(grp):
    return pd.DataFrame({'expected_prevalence': [len(grp[grp['matching_label'] == 1])]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qqsum_output_path", type=str, default='../output/atlas-xl-seed2-lgret-lglm',
                        help="Output directory of QQSUM-RAG")
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
    inputs = [(qqsum_output_path + "/post_processed_cache",
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

    claim_split_predicted = pd.json_normalize(
        processed_df.to_dict(orient='records'),
        "matching_comment_clusters", ["id", "query", "passages", 'passages_len']
    )
    claim_split_predicted['prevalence'] = claim_split_predicted['prevalence'].astype(int)
    evaluation_df = claim_split_predicted.explode(['passages'])

    # Comment-KP Matching Annotation
    df['my_category'] = 1
    num_workers = 1
    inputs = [(qqsum_output_path + "/experiment_comment_kp_matching_cache",
               domain,
               df[df['my_category'] == domain].reset_index(drop=True)
               )
              for domain in df['my_category'].unique()]
    start_time = time.time()
    with Pool(num_workers) as processor:
        data = processor.starmap(prompted_claim_split_generation, inputs)
    print("TIME ELAPSED", time.time() - start_time)
    processed_evaluation_df = pd.concat(data)

    processed_evaluation_df = processed_evaluation_df.drop(columns=['passages_len'])
    processed_evaluation_df['matching_detail_label'] = processed_evaluation_df['matching_response'].apply(entailment_detail_labels)
    processed_evaluation_df['matching_label'] = processed_evaluation_df['matching_response'].apply(entailment_labels)

    print("################################ KP QUANTIFICATION EVALUATION (Comment-KP Matching) ################################")
    # Comment-KP Matching Evaluation
    # Precision
    evaluation_df = claim_split_predicted.explode(['comments'])
    ref_df = processed_evaluation_df[['id', 'query', 'my_category', 'key_point', 'passages', 'matching_detail_label', 'matching_label']]
    ref_df = ref_df.rename(columns={'passages': 'comments'})
    precision_df = evaluation_df.merge(ref_df.drop_duplicates(['id', 'query', 'my_category', 'key_point', 'comments']), how='left')
    stat_df = precision_df.groupby(['id']).apply(calculate_precision)
    precision = stat_df['precision'].mean()
    # Recall
    evaluation_df = claim_split_predicted.explode(['comments'])
    evaluation_df['predicted_label'] = 1
    mask = processed_evaluation_df['matching_label'] == 1
    ref_df = processed_evaluation_df[mask][['id', 'query', 'my_category', 'key_point', 'passages', 'matching_detail_label', 'matching_label']]
    ref_df = ref_df.rename(columns={'passages': 'comments'})
    evaluation_df = evaluation_df.drop_duplicates(['comments', 'cluster_size', 'key_point', 'prevalence', 'id', 'query'])
    recall_df = ref_df.merge(evaluation_df, how='left')
    recall_df['predicted_label'] = recall_df['predicted_label'].fillna(0)
    stat_df = recall_df.groupby(['id']).apply(calculate_recall)
    recall = stat_df['recall'].mean()
    F1 = 2 * (precision * recall) / (precision + recall)
    # QuantErr
    precision_prevalence = precision_df.groupby(['id', 'key_point'], sort=False).apply(calculate_precision_prevalence).reset_index().drop(columns=['level_2'])
    recall_prevalence = recall_df.groupby(['id', 'key_point'], sort=False).apply(calculate_recall_prevalence).reset_index().drop(columns=['level_2'])
    mae_df = precision_prevalence.merge(recall_prevalence)
    quant_err = mae_df.apply(lambda x: abs(x['correct_prevalence'] - x['expected_prevalence']), axis=1).mean()
    print("###", "Precision", "Comment-KP Matching:", precision)
    print("###", "Recall", "Comment-KP Matching:", recall)
    print("###", "F1", "Comment-KP Matching:", F1)
    print("###", "QuantErr", "Comment-KP Matching:", quant_err)
