<div align="center">

# QQSUM: Query-Focus Quantitative Summarization

</div>

This repository maintains the code, data, and model checkpoints for the paper *QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering*

[//]: # (# Code to release soon.)

## Installation
It is recommended to set up the environment and install required libraries using conda. 
It is also recommended that the machine should have GPUs to perform inference at a reasonable time.  
### 1. Create new virtual environment by
```bash
conda create --name pakpa python=3.9
conda activate pakpa
```
### 2. Install Pytorch
#### Windows or Linux
##### Using GPUs
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
##### Using CPU
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
#### Mac
```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```
For other versions, please visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 3. Additional packages
We need some additional packages to run the code. The list of packages is listed in ```requirements.txt```. On the main directory of the repository, run:
```bash
pip install -r requirements.txt
```

## QQSUM: Task Introduction
![QQSUM_Task](QQSUM_Task.png)

We explored a new task, Quantitative Query-Focused Summarization (QQSUM), to generate comprehensive answers capturing diverse viewpoints along with their prevalence for review-based product question answering.

## The AmazonKP Dataset
We proposed AmazonKP, a new dataset specialized for training and evaluating models for the QQSUM task. 
The dataset can be accessed under the ``amazonkp``/ folder.
Files in each folder:
* ```.pkl```: data in .pkl format, accessible via Pandas library.
* ```.csv```: data in .csv format.
* ```.jsonl```: data in .jsonl format (only for Yelp raw data).

AmazonKP is curated based on a three-stage human-LLM collaborative annotation pipeline.
Additionally, we provide the code for reproducing the curation of AmazonKP, which consists of 3 stages
![AmazonKP_Annotation](AmazonKP_Annotation.png)

- **Stage 1:** Extracting key points (KPs) from gold community answer. We provided the code for prompting LLM to extract KPs from gold community answer in ...
- **Stage 2:** LLM-based and Manual Comment-KP Matching. We provided the code for prompting LLM to perform comment-KP Matching in  ...
- **Stage 3:** KP-based Summary


## The QQSUM-RAG Model
![QQSUM_Task](QQSUM_RAG_Model.png)

### Training

### Inference

### Model Performance
