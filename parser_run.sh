#python parser_gpu_operation_stat.py --json_file=/home/titanxp/log_json/$1.json 2>&1 | tee /home/titanxp/log_json/$1.txt
python trace_format.py --json_file=/home/titanxp/log_json/$1.json 2>&1 | tee /home/titanxp/log_json/$1_test.txt
