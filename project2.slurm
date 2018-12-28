#!/bin/bash

#SBATCH --job-name=RCS2017_project2a
#SBATCH --account=nn1000k
#SBATCH --time=15:02:00
#SBATCH --ntasks=12
#SBATCH --mem-per-cpu=1G

source /cluster/bin/jobsetup
module purge   #clear any inherited modules
set -o errexit  #exit on errors
module load python3/3.5.0

echo 'hello'
hostname
## Copy files to work directory:
cp $HOME/project2.py $SCRATCH
cp $HOME/inputNN $SCRATCH
cp $HOME/logisRegresANA.py $SCRATCH

### Mark outfiles for automatic copying to $SUBMITDIR HOME:
chkfile stateNN
## Run command:
cd $SCRATCH

python3 project2.py
hostname
sleep 60

