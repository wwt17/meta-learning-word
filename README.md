# Meta-Learning Word

This repository contains the implementation of a research project based on the paper: [Rapid Word Learning Through Meta In-Context Learning](https://arxiv.org/abs/2502.14791). The project focuses on Meta-training for IN-context learNing Of Words (Minnow), which trains language models to generate new examples of a word's usage given a few in-context examples.

## Environment settings

* Python 3.11
* Required packages: see [requirements.txt](requirements.txt).

## Build dataset

You can download the meta-learning and language modeling datasets [here](https://drive.google.com/file/d/1-7nDfNB5xq7JswRc2FhUOwItA8yCS63R/view?usp=sharing).

For other evaluation datasets, please follow the instructions below.

### Preprocess

Create .txt file(s) containing one sentence per line.

#### CHILDES

Use the code [here](https://github.com/wwt17/lm-povstim-with-childes/tree/master/data/CHILDES).

#### BabyLM

Download and unzip the [data](https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip).

Then, run
```bash
python babylm_data_processing.py --data_path ${BABYLM_DATA_PATH}
```
where `${BABYLM_DATA_PATH}` is the path to the BabyLM split, such as `babylm_data/babylm_10M`.

### Build meta-learning and language modeling dataset

`data_processing.py` processes the preprocessed .txt dataset above and generates the dataset for training, validation, and test. Its argument `--dataset ${DATA_PATH}` is the path to the directory containing the preprocessed dataset. Its argument `--word_use_data_dir ${WORD_USE_DATA_DIR}` is the directory containing the generated dataset, which will have path `${WORD_USE_DATA_DIR}/${DATA_PATH}/word` (`${DATASET_DIR}`). `${WORD_USE_DATA_DIR}` defaults to `word_use_data`.

Note `data_processing.py` will use SpaCy `en_core_web_trf` model to produce POS tags for the whole preprocessed dataset and cache them in a file. This may take a long time, so you may run the model on GPU by setting `use_gpu_for_spacy_model = True` in the code. However, doing so will [change the locale due to a bug of CUDA](https://github.com/explosion/spaCy/issues/11909), so you will have to set back `use_gpu_for_spacy_model = False` after you obtain the cache and retry.

#### CHILDES

```bash
python data_processing.py --dataset ${CHILDES_DATA_PATH} --lower --remove_sents_less_than_n_words 1 --plot_word_frequency --plot_pos --min_n_examples 5 --max_freq 200 --seed 0
```

#### BabyLM

##### 10M
```bash
python data_processing.py --dataset ${BABYLM_DATA_PATH} --lower --remove_sents_less_than_n_words 1 --remove_sents_longer_than_n_tokens 70 --plot_word_frequency --plot_pos --min_n_examples 5 --max_freq 15 --seed 0
```
where `${BABYLM_DATA_PATH}` is the path to the BabyLM 10M split, such as `babylm_data/babylm_10M`.

##### 100M
```bash
python data_processing.py --dataset ${BABYLM_DATA_PATH} --lower --remove_sents_less_than_n_words 1 --remove_sents_longer_than_n_tokens 70 --plot_word_frequency --plot_pos --min_n_examples 10 --max_freq 100 --split_ratio 96 2 2 --seed 0
```
where `${BABYLM_DATA_PATH}` is the path to the BabyLM 100M split, such as `babylm_data/babylm_100M`.

#### Get dataset statistics

This command get the statistics of the dataset in `${DATASET_DIR}`, excluding words with frequency <= 9 from the vocabulary (so they will be treated as unks) and plotting the length distribution between 0 and 70 and the number of uses distribution between 10 and 100:
```bash
python data_loading.py stat --data ${DATASET_DIR} --freq_cutoff 9 --length_range 0 70 --n_uses_range 10 100
```

### Process evaluation datasets

#### Chimeras
Download the [dataset file](https://github.com/NLPrinceton/ALaCarte/blob/master/data-chimeras/dataset.txt) and rename it to `chimeras.txt` (we will use `${PATH}` to refer to its path). Please be sure not to modify the file in any sense (including opening and resaving it with other encodings). The file is in mixed UTF-8 and ISO-8859-1 encoding.

Then process the dataset file:
```bash
python generation_evaluation_data_processing.py chimeras ${PATH}
```
It will generate a JSON file with the same stem in the same folder as `${PATH}`, which can be loaded by our code.

#### CoLLEGe-DefGen
Download the dataset from the [CoLLEGe paper site](https://college-concept-learning.github.io/) (may need to ask the author for the permission to access the dataset). It should be a HuggingFace dataset (we will use `${PATH}` to refer to its path).

Then process the dataset file:
```bash
python generation_evaluation_data_processing.py defgen ${PATH}
```
It will generate a JSON file with the same stem in the same folder as `${PATH}`, which can be loaded by our code.

#### Oxford
Download and extract the dataset as described [here](https://github.com/shonosuke/ishiwatari-naacl2019#download-dataset). Say the path to the extracted directory is `${DEFINITION_DATA_PATH}`. Let `PATH=${DEFINITION_DATA_PATH}/oxford/test` (you may replace `test` with another split or `oxford` with another subfolder if you want).

Then process the dataset file:
```bash
python generation_evaluation_data_processing.py ishiwatari ${PATH}
```
It will generate a JSON file with the same stem in the same folder as `${PATH}`, which can be loaded by our code.

## Training models from scratch

Run `main.py`. You can use `runner.py` to create and submit Slurm jobs:
```bash
python runner.py --job_name_base meta-word --config runner_config/config.py --run_name_flag name --submit
```

You may read and change the config file as you need.

You may need to change the Slurm script header to fit your environment. Default header file is `runner_config/header.slurm`.

You can download the pretrained checkpoints [here](https://drive.google.com/file/d/1btcqU6oCGXiLOBBzEVx2aUb_Hk3B-EFg/view?usp=sharing).

## Finetuning pretrained models

Simply change the config to `runner_config/finetune_config.py`:
```bash
python runner.py --job_name_base meta-word --config runner_config/finetune_config.py --run_name_flag name --submit
```

You can download the model finetuned from Llama-3-8B on Hugging Face: `wwtforever/Meta-Llama-3-8B-Minnow-babylm-10m`, and the model finetuned from Llama-3-8B-Instruct on Hugging Face: `wwtforever/Meta-Llama-3-8B-Instruct-Minnow-babylm-10m` (replace the `pretrained_model` argument with these names in the `Llama_finetuned` grid in `runner_config/evaluation_config.py` and check out other arguments for how to run evaluation on these models).

## Evaluation

Run `evaluation.py`. On a dataset (given by `--data_dir`), it first evaluates the classification accuracies (the number(s) of classes are given by `--eval_n_classes`), then for each word in the dataset it generate the next example (or definition) by greedy decoding, top-p sampling, and beam search. You can use `runner.py` to create and submit Slurm jobs:
```bash
python runner.py --job_name_base meta-word-eval --config runner_config/evaluation_config.py --submit
```
The standard output will consist of classification accuracies and examples along with the generations. It should be saved (in a slurm output file, for examples) for further evaluations below.

### Evaluate generations (definitions) with ground-truths
Run `evaluate_generation.py` on the output file(s) from `evaluation.py`. You may also read `evaluate_generation_runner.py` for how to run `evaluate_generation.py` in different settings, and modify/run it for your evaluations.


### Compare generations (definitions)
Run `compare_generation.py` on the two output files from `evaluation.py`.
