## KP Textual Quality

## Comment-KP Matching

## Comment-KP Factual Consistency
To reproduce comment-KP Factual Consistency evaluation, note that AlignScore are trained and evaluated using PyTorch 1.12.1.
We recommend set up a separate Anaconda environment running Pytorch 1.12.1 in reference to the [AlignScore](https://github.com/yuh-zha/AlignScore)'s repo setup instruction.
```bash
conda create --name alignscore python=3.9
conda activate alignscore
conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd AlignScore
pip install .
python -m spacy download en_core_web_sm
```

Importantly, to run AlignScore, please download ``AlignScore-base`` checkpoint and place the model to ``AlignScore/checkpoints``

**AlignScore-base:** https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt

[//]: # (After download ``AlignScore-base`` model checkpoint, please move it to the ``/checkpoints`` in this directory.)