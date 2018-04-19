./rm.sh

rm /tmp/tfmodel/*

python train_image_classifier.py \
--dataset_name=imagenet \
--dataset_split_name=validation \
--dataset_dir=/home/titanxp/imagenet2 \
--model_name=vgg_19 \
--batch_size=$1 \
--save_summaries_secs=0 \
--save_interval_secs=0 \
--gpu_allow_growth=True \
--trace_every_n_steps=True \
