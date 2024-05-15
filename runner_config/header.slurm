#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=turing|volta
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww2135@nyu.edu