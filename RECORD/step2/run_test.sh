num_phase=4

phase_idx=0
echo $phase_idx
echo "PI = $phase_idx, Task 1" >> log_task_1

for((phase_idx=1;phase_idx<num_phase;phase_idx++))
do
	echo $phase_idx
	echo "PI = $phase_idx, Task 1" >> log_task_1; if [[ $phase_idx != 1 ]]; then echo "PI = $phase_idx, Task 3" >> log_task_3; fi &
	echo "PI = $phase_idx, Task 2" >> log_task_2 &
	wait
done

echo $phase_idx
echo "PI = $phase_idx, Task 2" >> log_task_2


