# Meta-Learning Word

Meta-learning a new word given a few use examples (i.e. sentences illustrating the use of the new word) in context.

## Environment settings

* Python 3.11
* Required packages: see [requirements.txt](requirements.txt).

## Build dataset

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

## Training models from scratch

Run `main.py`. You can use `runner.py` to create and submit Slurm jobs:
```bash
python runner.py --job_name_base meta-word --config runner_config/config.py --run_name_flag name --submit
```

You may read and change the config file as you need.

You may need to change the Slurm script header to fit your environment. Default header file is `runner_config/header.slurm`.

## Finetuning pretrained models

Simply change the config to `runner_config/finetune_config.py`:
```bash
python runner.py --job_name_base meta-word --config runner_config/finetune_config.py --run_name_flag name --submit
```

## Evaluation

Run `evaluation.py`. You can use `runner.py` to create and submit Slurm jobs:
```bash
python runner.py --job_name_base meta-word-eval --config runner_config/evaluation_config.py --submit
```