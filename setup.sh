#/bin/bash

module load conda
conda activate /grand/CSC249ADCD08/twang/env/base-clone-polaris

which python
python -V

export RADICAL_PILOT_DBURL=$(cat /home/twang3/myWork/exalearn_project/run_polaris/rp_dburl_polaris)

export RADICAL_LOG_LVL=DEBUG
export RADICAL_PROFILE=TRUE
export RADICAL_SMT=1

export PS1="[$CONDA_PREFIX] \u@\H:\w> "

echo "Need to setup libgfortran manually!"
echo "This need to be done inside RCT!!!"
export LD_LIBRARY_PATH=/home/twang3/g2full_polaris/GSASII/bindist:$LD_LIBRARY_PATH

