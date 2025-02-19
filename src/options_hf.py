from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RAGArguments:
    # basic parameters
    name: str = field(
        default="experiment_name",
        metadata={"help": "Name of the experiment - also used as directory name"}
    )
    checkpoint_dir: str = field(
        default="./checkpoint/",
        metadata={"help": "Models are saved here"}
    )
    model_path: str = field(
        default="none",
        metadata={
            "help": "Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever)"
        }
    )
    per_gpu_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/CPU for training."}
    )
    per_gpu_embedder_batch_size: int = field(
        default=512,
        metadata={"help": "Embedder's batch size per GPU."}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"}
    )
    main_port: int = field(
        default=-1,
        metadata={"help": "Main port (for multi-node jobs)"}
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed for initialization"}
    )
    log_freq: int = field(
        default=100,
        metadata={"help": "Log train stats <log_freq> steps during training"}
    )
    eval_freq: int = field(
        default=500,
        metadata={"help": "Evaluate model every <eval_freq> steps during training"}
    )
    save_freq: int = field(
        default=30,
        metadata={"help": "Save model every <save_freq> steps during training"}
    )
    train_data: List[str] = field(
        default_factory=list,
        metadata={"help": "List of space-separated paths to jsonl-formatted train sets"}
    )
    eval_data: List[str] = field(
        default_factory=list,
        metadata={"help": "List of space-separated paths to jsonl-formatted evaluation sets"}
    )
    write_results: bool = field(
        default=False,
        metadata={"help": "Save evaluation results to file"}
    )
    dont_write_passages: bool = field(
        default=False,
        metadata={"help": "If writing results, pass this flag to not write passages as part of dumped results"}
    )

    # Optim Options
    warmup_steps: int = field(
        default=5,
        metadata={"help": "Number of learning rate warmup steps"}
    )
    total_steps: int = field(
        default=30,
        metadata={"help": "Total number of training steps"}
    )
    scheduler_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of steps for the scheduler, if None then scheduler_total_step = total_step"}
    )
    accumulation_steps: int = field(
        default=2,
        metadata={"help": "Gradient accumulation steps"}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )
    lr: float = field(
        default=4e-5,
        metadata={"help": "Learning rate"}
    )
    lr_retriever: float = field(
        default=4e-5,
        metadata={"help": "Learning rate for retriever"}
    )
    clip: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping"}
    )
    scheduler: str = field(
        default="linear",
        metadata={
            "help": "Learning rate schedule to use",
            "choices": ["linear", "cosine", "fixed"]
        }
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Amount of weight decay to apply in training"}
    )
    save_optimizer: bool = field(
        default=False,
        metadata={"help": "Pass flag to save optimizer state in saved checkpoints"}
    )
    epsilon: float = field(
        default=1e-6,
        metadata={"help": "AdamW epsilon value"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "AdamW alpha value"}
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "AdamW beta2 value"}
    )
    refresh_index: str = field(
        default="-1",
        metadata={"help": "Index refresh schedule. -1 to never refresh."}
    )
    shuffle: bool = field(
        default=False,
        metadata={"help": "Shuffle data for training"}
    )
    # memory optimizations:
    precision: str = field(
        default="fp32",
        metadata={
            "help": "Numerical precision - recommend bf16 if available, fp16 likely to be unstable for training",
            "choices": ["fp16", "fp32", "bf16"]
        }
    )
    shard_optim: bool = field(
        default=False,
        metadata={
            "help": "Shard optimizer state over available GPUs (sharded data parallel), recommended for larger models"}
    )
    shard_grads: bool = field(
        default=False,
        metadata={"help": "Shard gradients over available GPUs (sharded data parallel), recommended for larger models"}
    )
    use_gradient_checkpoint_reader: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing in the reader"}
    )
    use_gradient_checkpoint_retriever: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing for retriever"}
    )

    # Modelling
    reader_model_type: str = field(
        default="./Mistral-7B-Instruct-v0.2-GPTQ",
        metadata={
            "help": "T5 Architecture for reader FID model, e.g. google/t5-xl-lm-adapt",
            "choices": [
                "t5-small",
                "t5-base",
                "t5-large",
                "t5-3b",
                "t5-11b",
                "google/t5-v1_1-base",
                "google/t5-v1_1-large",
                "google/t5-v1_1-xl",
                "google/t5-v1_1-xxl",
                "google/t5-base-lm-adapt",
                "google/t5-large-lm-adapt",
                "google/t5-xl-lm-adapt",
                "google/t5-xxl-lm-adapt",
                "google/flan-t5-base",
                "google/flan-t5-xl",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "./Mistral-7B-Instruct-v0.2-GPTQ"
            ]
        }
    )
    text_maxlength: int = field(
        default=2048,
        metadata={
            "help": "Maximum number of tokens in input text segments (concatenated question + passage). Inputs longer than this will be truncated."
        }
    )
    target_maxlength: int = field(
        default=200,
        metadata={
            "help": "Maximum length of target outputs in tokens when training the model. Targets longer than this will be truncated. No truncation if -1."
        }
    )
    n_context: int = field(
        default=1,
        metadata={"help": "Number of top k passages to pass to reader"}
    )

    # Retriever modelling options
    passages: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path"}
    )
    max_passages: int = field(
        default=-1,
        metadata={"help": "Maximum number of passages to index. -1 to read all passages in passage files"}
    )
    retriever_model_path: str = field(
        default="facebook/contriever",
        metadata={"help": "Path to contriever model to init from (overridden if passing a value to --model_path)"}
    )
    retrieve_only: bool = field(
        default=False,
        metadata={"help": "Pass this to prevent loading a reader, and only run retrieval evaluation"}
    )
    train_retriever: bool = field(
        default=False,
        metadata={"help": "Pass to train retriever as well as reader"}
    )
    use_file_passages: bool = field(
        default=False,
        metadata={
            "help": 'Uses passages in "passages" field in train or eval jsonl files rather than retrieving passages'}
    )
    retriever_n_context: int = field(
        default=5,
        metadata={"help": "Number of top k passages to use to train the retriever with"}
    )
    gold_score_mode: str = field(
        default="ppmean",
        metadata={
            "help": "Retriever training method. `pdist` is the name used in the paper for `ppmean`. `adist` is the name used in the paper for `evalnormsum`",
            "choices": ["evalnormsum", "loop", "ppmean", "emdr", "pdist", "adist", "none"]
        }
    )
    closed_book: bool = field(
        default=False,
        metadata={
            "help": "Don't use retrieval - reduces to T5. Overrides n_context, n_context_retriever, and encoder_format if they are set"}
    )
    temperature_score: float = field(
        default=0.01,
        metadata={"help": "Softmax temperature for retriever"}
    )
    temperature_gold: float = field(
        default=0.01,
        metadata={"help": "Softmax temperature for target distribution for retriever distillation"}
    )
    compute_crossattention_stats: bool = field(
        default=False,
        metadata={"help": "Compute cross-attention statistics"}
    )
    filtering_overretrieve_ratio: int = field(
        default=2,
        metadata={
            "help": "If filtering, over-retrieve the topK by this factor, and then filter out undesirable results"}
    )
    freeze_retriever_steps: int = field(
        default=-1,
        metadata={"help": "Freezes retriever for n steps"}
    )
    query_side_retriever_training: bool = field(
        default=False,
        metadata={
            "help": "Pass to enable query-side fine-tuning of retriever (unties the parameters of the contriever encoder's passage and query encoders)"}
    )
    retrieve_with_rerank: bool = field(
        default=False,
        metadata={"help": "Pass this to enable reranking with fresh passage encoder for retriever"}
    )
    n_to_rerank_with_retrieve_with_rerank: int = field(
        default=512,
        metadata={
            "help": "Number of passages to rerank when passing --retrieve_with_rerank. Higher is slower but more accurate"}
    )
    train_with_rank_threshold: bool = field(
        default=False,
        metadata={"help": "Pass this to enable training the retriever using rank_threshold"}
    )
    rank_threshold: float = field(
        default=1.3,
        metadata={"help": "Threshold for ranking"}
    )
    max_retrieval: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of comments to retrieve before rank_threshold"}
    )
    rerank_retriever_model_path: str = field(
        default="",
        metadata={"help": "Path to contriever model for FC data to init from"}
    )

    # input and output formatting options:
    decoder_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "Format for decoder, model will train on the format and evaluation will be performed with the format contrary to the decoder_prompt_format option"
        }
    )
    decoder_prompt_format: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Format for decoder prompting, for instance "what is the answer to {query}:"'
        }
    )
    encoder_format: str = field(
        default="{query} title: {title} context: {text}",
        metadata={
            "help": "Format string for reader's encoder preprocessing"
        }
    )
    retriever_format: str = field(
        default="{title} {text}",
        metadata={
            "help": "Format string for retriever's encoder preprocessing"
        }
    )

    # Generation options
    generation_max_length: int = field(
        default=200,
        metadata={"help": "Maximum length for generation"}
    )
    generation_min_length: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum length for generation"}
    )
    generation_length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for generation"}
    )
    generation_num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for generation"}
    )

    # para for justilm:
    '''
    Different loss settings:
        1. Article-level retrieval loss and generation loss: --lgret --lglm 
        2. Article-level retrieval loss and chunk-level generation loss: --lgret --lclm
        3. Chunk-level retrieval loss and generation loss: --lcret --lclm
        4. Chunk-level retrieval loss and Article-level generation loss: --lcret --lglm
        5. MT setting with Article-level generation loss: --mt --label_file --lglm  
        6. MT setting with Chunk-level generation loss: --mt --label_file --lclm  
    '''
    lgret: bool = field(
        default=False,
        metadata={"help": "Article-level retrieval loss"}
    )
    lcret: bool = field(
        default=False,
        metadata={"help": "Chunk-level retrieval loss"}
    )
    lglm: bool = field(
        default=False,
        metadata={"help": "Article-level generation loss"}
    )
    lclm: bool = field(
        default=False,
        metadata={"help": "Chunk-level generation loss"}
    )
    fc_only: bool = field(
        default=False,
        metadata={"help": "Fully connected layer only"}
    )
    mt: bool = field(
        default=False,
        metadata={"help": "MT setting"}
    )
    lglm_cof: float = field(
        default=0.15,
        metadata={"help": "Coefficient for article-level generation loss"}
    )
    lclm_cof: float = field(
        default=100.0,
        metadata={"help": "Coefficient for chunk-level generation loss"}
    )
    mt_cof: float = field(
        default=0.1,
        metadata={"help": "Coefficient for MT loss"}
    )
    fc_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to fully connected file"}
    )
    label_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to label file"}
    )

    # Index Options
    load_index_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path for loading the index, passage embeddings and passages"}
    )
    save_index_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path for saving the index and/or embeddings"}
    )
    save_index_n_shards: int = field(
        default=128,
        metadata={
            "help": "How many shards to save an index to file with. Must be an integer multiple of the number of workers."}
    )
    index_mode: str = field(
        default="faiss",
        metadata={
            "help": "Use flat torch index or a faiss index for retrieving the k nearest neighbors",
            "choices": ["flat", "faiss"]  # The choices are added here
        }
    )
    faiss_index_type: str = field(
        default="pq",
        metadata={
            "help": "IVFFlat, IndexFlatIP, IVFScalarQuantizer or IndexIVFPQ with faiss-gpu",
            "choices": ["ivfflat", "flat", "ivfsq", "ivfpq", "pq"]  # The choices are added here
        }
    )
    faiss_code_size: int = field(
        default=192,
        metadata={"help": "Parameter for PQ/SQ quantization"}
    )

