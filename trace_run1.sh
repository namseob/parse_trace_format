./rm.sh

w_d=0.00004

if [ $3 -eq 0 ];then
	w_d=0
fi

echo ${w_d}

rm /tmp/tfmodel/*

if [ $6 -ne 0 ];then
	kill -5 $6
	pid=`ps -ef | grep nvidia-smi | grep -v grep | awk '{print $2}'`
fi

python train_image_classifier.py \
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
--trace_every_n_steps=True \

'''
while [ "$?" -eq 1 ]
do
	:
done

if [ $6 -ne 0 ];then
	kill -9 $pid
	kill -9 $6
fi

i=1
while [ "$i" -le "$2" ]
do
	mv /tmp/tfmodel/tf_trace-$i.json /home/titanxp/log_json/[$5]batch-$1_step-$2_decay-$3_optimizer-$4_trace-$i.json
	i=$(($i+1))
done

if [ $6 -ne 0 ];then
	mv ./gpu_log.csv /home/titanxp/log_csv/[$5]batch-$1_step-$2_decay-$3_optimizer-$4_trace.csv
	python print_plot.py --csv_file=/home/titanxp/log_csv/[$5]batch-$1_step-$2_decay-$3_optimizer-$4_trace.csv
fi
'''
