## The AmazonKP Dataset
We proposed AmazonKP, a new dataset specialized for training and evaluating models for the QQSUM task.
The dataset can be accessed under the ```data/``` folder, 
following the [```train/```](/data/train) and [```test/```](/data/test) subdirectories for the train and test set.

Files in each sub-directory:
```
data
├── train
│   ├── train.jsonl
│   ├── copora
│       ├── input_reviews.jsonl
│       ├── gold_comment_clusters.jsonl
│       ├── gold_retrieved_comments.jsonl
├── test
│   ├── test.jsonl
│   ├── copora
│       ├── input_reviews.jsonl
├── full
│   ├── amazon_kp_dataset.jsonl
│   ├── amazon_kp_dataset.csv
│   ├── amazon_kp_dataset.pkl
```

File description:
* ```train.jsonl``` or ```test.jsonl``` : Input data file containing the input question, i.e., query, and the final key point (KP) summary ground truth. 
* ```input_reviews.jsonl```: Complementary input product review comments of questions in  ```train.jsonl``` or ```test.jsonl```
* ```gold_comment_clusters.jsonl```: Clusters of comments formed by comment-KP matching annotation (Stage 2)
* ```gold_retrieved_comments.jsonl```: query-relevant comments aggregated from annotated clusters

[//]: # (```)
[//]: # (├── corpus)
[//]: # (│   ├── docs.jsonl)
[//]: # (│   ├── docs_test_full.jsonl)
[//]: # (│   ├── fc_articles.json)
[//]: # (│   ├── fc_clusters.with_id.json)
[//]: # (├── train.jsonl)
[//]: # (├── test.jsonl)
[//]: # (```)
[//]: # (Files in each folder:)
[//]: # (* ```.pkl```: data in .pkl format, accessible via Pandas library.)
[//]: # (* ```.csv```: data in .csv format.)
[//]: # (* ```.jsonl```: data in .jsonl format &#40;only for Yelp raw data&#41;.)

*NOTE: For training or evaluating QQSUM-RAG on your own dataset instead of AmazonKP, please make sure to follow the data format and structure given in the above `jsonl` files*


### AmazonKP Dataset Curation (optional)
AmazonKP is curated based on a three-stage human-LLM collaborative annotation pipeline.
Optionally, we provide the code for you to reproduce the curation of AmazonKP, which consists of 3 stages:

![AmazonKP_Annotation](../diagram/AmazonKP_Annotation.png)

- **Stage 1:** Extracting key points (KPs) from gold community answer. We provided the code for prompting LLM to extract KPs from gold community answer in ...
- **Stage 2:** LLM-based and Manual Comment-KP Matching. We provided the code for prompting LLM to perform comment-KP Matching in  ...
- **Stage 3:** KP-based Summary