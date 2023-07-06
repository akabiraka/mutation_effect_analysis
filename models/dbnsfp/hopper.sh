#!/bin/bash
#SBATCH --job-name=dbNSFP
#SBATCH --output=/projects/ashehu/akabir4/projects/mutation_effect_analysis/logs/popu_freq-%j.out
#SBATCH --error=/projects/ashehu/akabir4/projects/mutation_effect_analysis/logs/popu_freq-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

## cpu 
#SBATCH --partition=contrib                  # submit   to the normal(default) partition
#SBATCH --mem=16GB                # Request nGB RAM per core
#SBATCH --time=0-08:00   # Total time needed for job: Days-Hours:Minutes


## Load the relevant modules needed for the job
module load openjdk # openjdk/11.0.2-qg
source /projects/ashehu/akabir4/venvs/hopper_variant_effect_analysis_mine/bin/activate


## for full scale run it took ~32 minutes
java -cp data/dbnsfp/dbNSFP43a/ search_dbNSFP43a -i models/dbnsfp/datasets/popu_freq.ssv -o models/dbnsfp/outputs_dbnsfp/popu_freq.txt -w 1-6,38,47,77,83,90,118,141,162,168,173

## java -cp data/dbnsfp/dbNSFP43a/ search_dbNSFP43a -i models/dbnsfp/datasets/patho_and_likelypatho.ssv -o models/dbnsfp/outputs_dbnsfp/patho_and_likelypatho.txt -w 1-6,38,47,77,83,90,118,141,162,168,173

## java -cp data/dbnsfp/dbNSFP43a/ search_dbNSFP43a -i models/dbnsfp/datasets/pmd.ssv -o models/dbnsfp/outputs_dbnsfp/pmd.txt -w 1-6,38,47,77,83,90,118,141,162,168,173