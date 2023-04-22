#!/bin/bash

for node in {1,2,3,4,5,6,7,8}
do
	echo $node
	/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/run_ml.sh $node > log_step1_part45_ml_node_${node}_cpr_2
done
