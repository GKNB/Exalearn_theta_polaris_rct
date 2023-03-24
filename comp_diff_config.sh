#!/bin/bash

dir0=/home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/
dir1=/home/twang3/myWork/exalearn_project/run_theta/parallel_rct/test_config/

echo "Compare cubic"
diff ${dir0}/config_1001460_cubic.txt ${dir1}/configs_phase0/config_1001460_cubic.txt
diff ${dir1}/configs_phase0/config_1001460_cubic.txt ${dir1}/configs_phase1/config_1001460_cubic.txt
echo ""
echo ""
echo ""
echo ""
echo ""

echo "Compare trigonal_part1"
diff ${dir0}/config_1522004_trigonal_part1.txt ${dir1}/configs_phase0/config_1522004_trigonal_part1.txt
diff ${dir1}/configs_phase0/config_1522004_trigonal_part1.txt ${dir1}/configs_phase1/config_1522004_trigonal_part1.txt
echo ""
echo ""
echo ""
echo ""
echo ""

echo "Compare trigonal_part2"
diff ${dir0}/config_1522004_trigonal_part2.txt ${dir1}/configs_phase0/config_1522004_trigonal_part2.txt
diff ${dir1}/configs_phase0/config_1522004_trigonal_part2.txt ${dir1}/configs_phase1/config_1522004_trigonal_part2.txt
echo ""
echo ""
echo ""
echo ""
echo ""

echo "Compare tetragonal"
diff ${dir0}/config_1531431_tetragonal.txt ${dir1}/configs_phase0/config_1531431_tetragonal.txt
diff ${dir1}/configs_phase0/config_1531431_tetragonal.txt ${dir1}/configs_phase1/config_1531431_tetragonal.txt

