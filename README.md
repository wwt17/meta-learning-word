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

#### CHILDES

```bash
python data_processing.py --dataset ${CHILDES_DATA_PATH} --lower --remove_sents_less_than_n_words 1 --plot_word_frequency --plot_pos --min_n_examples 5 --max_freq 200 --seed 0 --word_use_data_dir word_use_data
```

#### BabyLM

```bash
python data_processing.py --dataset ${BABYLM_DATA_PATH} --lower --remove_sents_less_than_n_words 1 --remove_sents_longer_than_n_tokens 70 --plot_word_frequency --plot_pos --min_n_examples 5 --max_freq 15 --seed 0 --word_use_data_dir word_use_data
```