#!/bin/bash

#SBATCH -e logs/snake_conv.err
#SBATCH -o logs/snake_conv.out
#SBATCH -J snake_conv

#SBATCH --partition=cccmd
#SBATCH --mem=6G
#SBATCH --cpu-freq=high
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gonzalo.hernandez@uam.es
#SBATCH --account=ada2_serv

set -eo pipefail -o nounset


###
python main.py 0
