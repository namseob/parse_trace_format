./rm.sh

w_d=0.00004

if [ $3 -eq 0 ];then
	w_d=0
fi

echo ${w_d}

rm /home/titanxp/log_nvvp/$5_batch-$1_step-$2_decay-$3_optimizer-$4.nvvp

#nvprof --unified-memory-profiling per-process-device -o /home/titanxp/log_nvvp/sgd_no_decay_batch-$1_step-$2.nvvp python train_image_classifier.py \
#nvprof --events all --metrics all -o /home/titanxp/log_nvvp/batch-$1_step-$2_decay-$3_optimizer-$4.nvvp python train_image_classifier.py \

if [ $6 -ne 0 ];then
	kill -5 $6
	pid=`ps -ef | grep nvidia-smi | grep -v grep | awk '{print $2}'`
fi


nvprof --print-gpu-trace --metrics gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput,l1_cache_global_hit_rate,l2_l1_read_throughput,l2_l1_write_throughput,dram_read_throughput,dram_write_throughput,dram_utilization -o /home/titanxp/log_nvvp/$5_batch-$1_step-$2_decay-$3_optimizer-$4.nvvp python train_image_classifier.py \
--dataset_name=imagenet \
--dataset_split_name=train \
--dataset_dir=/home/titanxp/imagenet2 \
--model_name=$5 \
--batch_size=$1 \
--max_number_of_steps=$2 \
--optimizer=$4 \
--weight_decay=${w_d} \
--save_summaries_secs=0 \
--save_interval_secs=0 \
--gpu_allow_growth=True \

while [ "$?" -eq 1 ]
do
	:
done


if [ $6 -ne 0 ];then
	kill -9 $pid
	kill -9 $6

	mv ./gpu_log.csv /home/titanxp/log_csv/[$5]batch-$1_step-$2_decay-$3_optimizer-$4_trace.csv
	python print_plot.py --csv_file=/home/titanxp/log_csv/[$5]batch-$1_step-$2_decay-$3_optimizer-$4_trace.csv
fi
