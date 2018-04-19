import argparse
import json
import copy
from collections import OrderedDict

import time

# event type
EV_DURATION_BEGIN			= 'B'
EV_DURATION_END				= 'E'

EV_COMPLETE						= 'X'
EV_INSTANT						= 'I'
EV_COUNTER						= 'C'

EV_FLOW_EVENT_START		= 's'
EV_FLOW_EVENT_STEP		= 't'
EV_FLOW_EVENT_END			= 'f'

EV_OBJECT_CREATED			= 'N'
EV_OBJECT_SNAPSHOT		= 'O'
EV_OBJECT_DESTROYED		= 'D'

EV_METADATA						= 'M'

# event category
EV_CAT_TENSOR					= 'Tensor'
EV_CAT_OP							= 'Op'

# event description
EV_NAME = 'name'
EV_TS		= 'ts'
EV_DUR	= 'dur'
EV_PH		=	'ph'

# event args
EV_ARGS_NAME_MEMCPYHTOD	=	'MEMCPYHtoD'
EV_ARGS_NAME_CONV2D			= 'Conv2D'

# log_file path
LOG_PATH	= "/home/titanxp/log_json/"
EXTENSION	= "json"
#sample_filename	= "[vgg_19]batch-32_step-[steps]_decay-0_optimizer-sgd_trace-[step]"
sample_filename	= ""

# debugging variable
PRINT_STEP		=	72
PRINT_ENABLE	= 0

'''
Input : trace_evetns
Input type : list

return : (whether overlapsed or not, overlapsed time)
return type: tuple(bool, float)
'''
def check_pipelining_memcpy_forward(trace_events):
	memcpy_end_ts_list = []
	conv2d_start_ts_list = []
	for trace_ev in trace_events:
		# extract 'MEMCPYHtoD' event
		if (trace_ev[EV_NAME] == EV_ARGS_NAME_MEMCPYHTOD):
			memcpy_end_ts_list.append(trace_ev[EV_TS]+trace_ev[EV_DUR])
		memcpy_end_ts_list.sort()

		# extract 'Conv2D' event
		if (trace_ev[EV_NAME] == EV_ARGS_NAME_CONV2D):
			conv2d_start_ts_list.append(trace_ev[EV_TS])
		conv2d_start_ts_list.sort()

	#compare the last HtoD memcpy's time with the first conv2d's time
	overlapsed_time = memcpy_end_ts_list[-1] - conv2d_start_ts_list[0]

	##### debugging #####
	global PRINT_ENABLE
	if PRINT_ENABLE == 1:
		#print("memcpy : " + str(time.ctime(memcpy_end_ts_list[-1])))
		#print("conv2d : " + str(time.ctime(conv2d_start_ts_list[0])))
		print("memcpy : " + str(memcpy_end_ts_list[-1]))
		print("conv2d : " + str(conv2d_start_ts_list[0]))
		print("difference : " + str((memcpy_end_ts_list[-1] - conv2d_start_ts_list[0]) / 1000) + "ms")
		PRINT_ENABLE = 0
	#####################

	if(overlapsed_time > 0.0):
		return (True, overlapsed_time)
	
	return (False, 0.0)

def count_file_with_pipelining(steps):
	split_filename = sample_filename.split('_')

	index = 0
	for splited in split_filename:
		if splited.find('steps') != -1:
			break;
		index += 1

	split_filename[index] = split_filename[index].replace('[steps]', str(steps))

	overlapsed_time_list = []
	count = 0
	for i in range(1, steps+1):
		temp_split_filename = copy.deepcopy(split_filename)
		temp_split_filename[-1] = temp_split_filename[-1].replace('[step]', str(i))

		trace_filename = '_'.join(temp_split_filename)
		trace_file_path = LOG_PATH + trace_filename + '.' + EXTENSION

		##### debugging #####
		#print(trace_file_path)
		global PRINT_ENABLE
		global PRINT_STEP
		if i is PRINT_STEP:
			PRINT_ENABLE = 1
		#####################

		with open(trace_file_path, 'rt') as data_file:
			trace_data = json.load(data_file, object_pairs_hook=OrderedDict)
			trace_events = trace_data['traceEvents']
	
			tup_result = check_pipelining_memcpy_forward(trace_events)
				
			if(tup_result[0]):
				count += 1
				overlapsed_time_list.append(tup_result[1])

	print("count : " + str(count))
	print("total : " + str(steps))
	print("ratio : " + str(count/steps))
	print("average_overlapsed_time : " + str(sum(overlapsed_time_list)/1000/count) + "ms")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, help="model_name")
	parser.add_argument('--batch_size', type=str, help="batch_size")
	parser.add_argument('--decay', type=str, help="decay")
	parser.add_argument('--optimizer', type=str, help="optimizer")

	parser.add_argument('--num_steps', type=int, help="num_steps")

	args = parser.parse_args()
	sample_filename	+= "[" + args.model_name + "]"
	sample_filename += "batch-" + args.batch_size + "_"
	sample_filename += "step-[steps]_"
	sample_filename += "decay-" + args.decay + "_"
	sample_filename += "optimizer-" + args.optimizer + "_"
	sample_filename += "trace-[step]"
	
	#print(sample_filename)
	count_file_with_pipelining(args.num_steps)
