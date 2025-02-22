## KP Quality Evaluation ##
* ```KP_Quality_Evaluation_sP_sR_sF1.ipynb```: Perform sP/sR/sF1 set-level evaluation of individual generated KPs with reference KPs 
(extracted from gold community answer - Stage 1 of AmazonKP curation).
* ```KP_Quality_Evaluation_RD.ipynb```: Perform Redundancy (RD) evaluation among individual generated KPs 

## KP Quantification Evaluation ##
* ```KP_Quantification_Evaluation_Matching.ipynb```: Perform comment-KP matching to measure the matching *precision* (correctness of predicted matches) and *recall* (coverage of ground-truth matches) of generated KPs and comments in their respective clusters. 
* ```KP_Quantification_Evaluation_Factual_Alignment.txt```: Perform factual alignment evaluation between generated KPs and comments in their respective clusters.


[//]: # (## KP Textual Quality)

[//]: # (### Results)

[//]: # ()
[//]: # (## Comment-KP Matching)

[//]: # (### Results)

[//]: # ()
[//]: # (## Comment-KP Factual Consistency)

[//]: # (### Setup &#40;Important&#41;)

[//]: # (To reproduce comment-KP Factual Consistency evaluation, note that AlignScore are trained and evaluated using PyTorch 1.12.1.)

[//]: # (We recommend set up a separate Anaconda environment running Pytorch 1.12.1 in reference to the [AlignScore]&#40;https://github.com/yuh-zha/AlignScore&#41;'s repo setup instruction.)

[//]: # (```bash)

[//]: # (conda create --name alignscore python=3.9)

[//]: # (conda activate alignscore)

[//]: # (conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge)

[//]: # (cd AlignScore)

[//]: # (pip install .)

[//]: # (python -m spacy download en_core_web_sm)

[//]: # (```)

[//]: # ()
[//]: # (Importantly, to run AlignScore, please download ``AlignScore-base`` checkpoint and place the model to ``AlignScore/checkpoints``)

[//]: # ()
[//]: # (**AlignScore-base:** https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt)

[//]: # ()
[//]: # ([//]: # &#40;After download ``AlignScore-base`` model checkpoint, please move it to the ``/checkpoints`` in this directory.&#41;)
[//]: # ()
[//]: # (### Results)