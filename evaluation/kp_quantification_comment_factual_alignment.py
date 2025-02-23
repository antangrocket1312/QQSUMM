import argparse
from multiprocessing import Pool
import openai
import sys
sys.path.insert(1, '../')
from utils.postprocessing import *
from utils.g_eval import *
from alignscore import AlignScore


def parse_extracted_json(processed_df):
    processed_df['claim_split_predicted'] = processed_df['claim_split_predicted'].apply(
        lambda x: re.sub(r"\n+ *", "", x.replace("json", "").replace("`", "")))
    mask = processed_df['claim_split_predicted'].apply(
        lambda x: len(re.findall(r"(: *)\'((?:[^':]*\'+[^':,]*)+)\'( *)", x, re.DOTALL)) > 0)
    processed_df.loc[mask, 'claim_split_predicted'] = processed_df.loc[mask, 'claim_split_predicted'].apply(
        lambda x: re.sub(r"(: *)\'((?:[^':]*\'+[^':]*)+)\'( *,)", r'\1"""\2"""\3', x))
    processed_df['claim_split_predicted'] = processed_df['claim_split_predicted'].apply(extract_claims)

    return processed_df


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

    kp_matching_df = pd.json_normalize(
        processed_df.to_dict(orient='records'),
        "matching_comment_clusters", ["asin", "id", "query", "passages", 'passages_len', 'final_summary_text']
    )
    kp_matching_df = kp_matching_df[kp_matching_df['prevalence'] >= 3]

    print("################################ KP QUANTIFICATION EVALUATION (Factual Alignment) ################################")
    evaluation_df = kp_matching_df
    align_scorer = AlignScore(
        model='roberta-base',
        batch_size=8,
        device='cuda:0',
        ckpt_path='./AlignScore/checkpoints/AlignScore-base.ckpt',
        evaluation_mode='nli_sp'
    )
    evaluation_df['context'] = evaluation_df['comments'].apply(lambda x: " ".join(x))
    results = align_scorer.score(
        contexts=evaluation_df['context'].tolist(),
        claims=evaluation_df['key_point'].tolist(),
    )
    evaluation_df['align_score'] = results
    eval_results = evaluation_df.groupby(['id']).apply(lambda grp: grp['align_score'].mean()).reset_index()
    eval_results = eval_results.rename(columns={0: 'precision'})
    factual_alignment = eval_results['precision'].mean()
    print("###", "AlignScore", "Comment-KP Factual Alignment:", factual_alignment)
