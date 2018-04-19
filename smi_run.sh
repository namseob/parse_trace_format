echo $$

trap 'nvidia-smi --id=0 --query-gpu=timestamp,memory.free,memory.used --format=csv -lms 1 2>&1 | tee gpu_log.csv' SIGTRAP
#trap '' SIGQUIT

i=0
while [ 1 -eq 1 ]
do
	i=$(($i+1))
done
