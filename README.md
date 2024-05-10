# Meta-Learning Word

Meta-learning a new word given a few use examples (i.e. sentences illustrating the use of the new word) in context.

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
python data_processing.py --dataset ${BABYLM_DATA_PATH} --lower --remove_sents_less_than_n_words 1 --remove_sents_longer_than_n_tokens 70 --plot_word_frequency --plot_pos --min_n_examples 10 --max_freq 100 --seed 0
```
where `${BABYLM_DATA_PATH}` is the path to the BabyLM 100M split, such as `babylm_data/babylm_100M`.

#### Get dataset statistics

This command get the statistics of the dataset in `${DATASET_DIR}`, excluding words with frequency <= 9 from the vocabulary (so they will be treated as unks) and plotting the length distribution between 0 and 70 and the the number of uses distribution between 10 and 100:
```bash
python data_loading.py stat --data ${DATASET_DIR} --freq_cutoff 9 --length_range 0 70 --n_uses_range 10 100
```