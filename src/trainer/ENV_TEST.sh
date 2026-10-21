#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name="env_test"
#SBATCH --output=env_tes2t.out
#SBATCH --gres=gpu:1
#SBATCH --qos=longrunning


python ENVIROMENT_test.py
