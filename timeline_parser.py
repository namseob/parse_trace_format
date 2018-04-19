import json
import argparse
from collections import OrderedDict
from collections import Counter
import matplotlib.pyplot as plt

# Event Descriptions
event_type       = 'ph'
event_category   = 'cat'
event_name       = 'name'
event_args       = 'args'
event_timestamp  = 'ts'
event_thread_timestamp = 'tts'
event_process_id = 'pid'
event_thread_id  = 'tid'

# Event type
complete_ev = 'C'

# Category Type
memory_cat = "Memory"

# Name
gpu_bfc_name        = "GPU_0_bfc"
host_bfc_name       = "cuda_host_bfc"
cpu_name            = "cpu"

# Arguments

mega_bytes = 1024*1024
milli_sec  = 1000

class traceException(Exception):
    def __init__(self, msg, v1_ts, v2_gpu, v3_host, v4_cpu):
        self.msg    = msg
        self.ts     = v1_ts
        self.gpu    = v2_gpu
        self.host   = v3_host
        self.cpu    = v4_cpu
    def __str__(self):
        s = '{0:s} ts:{1:d} gpu:{2:d} host:{3:d}'.format(self. msg,self.ts, self.gpu, self.host, self.cpu)
        return s
    

def show_memory(key, val):
    if key == event_type and val == complete_ev:
        print(key, val)
        print(val[0])
        #for name, val in args.items():
        #    print(name, int(val))

def show_memory(event):
    ts       = -1
    gpu_mem  = -1
    host_mem = -1
    cpu_mem  = -1

    if event[event_type] == complete_ev and event[event_category] == memory_cat:
        args = event[event_args]
        ts   = int(event[event_timestamp])

        if args:
            for name, val in args.items():
                if name == gpu_bfc_name:
                    #print(name, val)
                    gpu_mem = int(val)
                elif name == host_bfc_name:
                    print(name, val)
                    host_mem = int(val)
                elif name == cpu_name:
                    cpu_mem = int(val)

            return ts, gpu_mem, host_mem, cpu_mem

    return ts, gpu_mem, host_mem, cpu_mem

def plot_gpu_memory_variation(timestamp_list, gpu_mem_list, title):
    #print(gpu_mem_list)
    max_mem_size = max(gpu_mem_list)
    s = "Max Memory Size = " + str(max_mem_size) + 'MB'
    print(s)
    
    plt.plot(timestamp_list, gpu_mem_list)
    plt.xlabel('time(sec)')
    plt.ylabel('MB')
    plt.title(title)
    # plt.set(xlabel = 'time(s)', ylabel='MB', title=title)
    plt.legend([s])
    plt.grid(True)
    plt.show()

def plot_cpu_memory_variation(timestamp_list, gpu_mem_list, title):
    #print(gpu_mem_list)
    max_mem_size = max(gpu_mem_list)
    s = "Max Memory Size = " + str(max_mem_size) + 'MB'
    print(s)
    
    plt.plot(timestamp_list, gpu_mem_list)
    plt.xlabel('time(msec)')
    plt.ylabel('MB')
    plt.title(title)
    # plt.set(xlabel = 'time(s)', ylabel='MB', title=title)
    plt.legend([s])
    plt.grid(True)
    plt.show()

def plot_memory(gpu_mem, host_mem, cpu_mem):
    x_gpu_ts, y_gpu_mem = zip(*gpu_mem)
    x_cuda_host_ts, y_cuda_host_mem = zip(*host_mem)
    x_cpu_ts, y_cpu_mem = zip(*cpu_mem)

    gpu_mem_label = "gpu memory" + "(Max Size:" + str(max(y_gpu_mem)) + "MB)"
    cuda_host_mem_label = "cuda host memory" + "(Max Size:" + str(max(y_cuda_host_mem)) + "MB)"
    cpu_mem_label = "cpu memory" + "(Max Size:" + str(max(y_cpu_mem)) + "MB)"


    ax = plt.subplot(111)
    ax.plot(x_gpu_ts, y_gpu_mem, 'r', label=gpu_mem_label)
    ax.plot(x_cuda_host_ts, y_cuda_host_mem, 'b',label=cuda_host_mem_label)
    ax.plot(x_cpu_ts, y_cpu_mem, 'g', label=cpu_mem_label)
    ax.legend()
    
    # plt.plot(x_gpu_ts, y_gpu_mem, 'r', x_cuda_host_ts, y_cuda_host_mem, 'b', x_cpu_ts, y_cpu_mem, 'g')
    plt.xlabel('time(msec)')
    plt.ylabel('MB')
    plt.title('Memory Size Map')
    plt.grid(True)
    plt.show()


def transform_timestamp_lists(ts_lists):
    init_clock = ts_lists[0]
    tf_ts_list = [(i - init_clock)/milli_sec for i in ts_lists]
    return tf_ts_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', help = 'json file to be parsed')
    args = parser.parse_args()
    print(args.json_file)

    with open(args.json_file, 'r') as js_f:
        data = json.load(js_f, object_pairs_hook=OrderedDict)
        trace_events = data['traceEvents']

        print(len(trace_events))
        for ev in trace_events:
            print(str(ev))

        gpu_mem_list        = []
        host_mem_list       = []
        cpu_mem_list        = []
        gpu_timestamp_list  = []
        host_timestamp_list = []
        cpu_timestamp_list  = []

        for ev in trace_events:
            ts, gpu_mem, host_mem, cpu_mem = show_memory(ev)
            if gpu_mem >= 0 and ts >= 0:
                gpu_mem_list.append(gpu_mem)
                gpu_timestamp_list.append(ts)
            elif host_mem >=0 and ts >= 0:
                host_mem_list.append(host_mem)
                host_timestamp_list.append(ts)
            elif cpu_mem >= 0 and ts >=0:
                cpu_mem_list.append(cpu_mem)
                cpu_timestamp_list.append(ts)
            else:
                pass
            #    try:
            #        raise traceException("Oh My God... exception error", ts, gpu_mem, host_mem, cpu_mem)
            #    except traceException as te:
            #        print(te)
        

        #initial_gpu_time = gpu_timestamp_list[0]
        #gpu_timestamp_list = [(i - initial_gpu_time)/milli_sec for i in gpu_timestamp_list]
        #gpu_mem_list = [int(i/mega_bytes) for i in gpu_mem_list]

        #initial_host_time = host_timestamp_list[0]
        #host_timestamp_list = [(i - initial_host_time)/milli_sec for i in host_timestamp_list]
        #host_mem_list = [int(i/mega_bytes) for i in host_mem_list]
        # print(timestamp_list)
        # for i in timestamp_list:
        #    print(i)

        gpu_mem_list  = [int(i/mega_bytes) for i in gpu_mem_list]
        host_mem_list = [int(i/mega_bytes) for i in host_mem_list]
        cpu_mem_list  = [int(i/mega_bytes) for i in cpu_mem_list]

        gpu_timestamp_list  = transform_timestamp_lists(gpu_timestamp_list)
        host_timestamp_list = transform_timestamp_lists(host_timestamp_list)
        cpu_timestamp_list  = transform_timestamp_lists(cpu_timestamp_list)
    
        # print(gpu_mem_list)
        ts_gpu_mem  = list(zip(gpu_timestamp_list, gpu_mem_list))
        ts_host_mem = list(zip(host_timestamp_list, host_mem_list))
        ts_cpu_mem  = list(zip(cpu_timestamp_list, cpu_mem_list))
        # for i in ts_gpu_mem:
        #    print(i)
        
        #fout = open('./gpu_timestamap.txt', 'wt')
        #fout.write(str(ts_gpu_mem))
        #fout.close()

        #fout = open('./host_timestamap.txt', 'wt')
        #fout.write(str(ts_host_mem))
        #fout.close()

        #plot_gpu_memory_variation(gpu_timestamp_list, gpu_mem_list, "VGG-19 GPU Memory Size")
        #plot_cpu_memory_variation(host_timestamp_list, host_mem_list, "CUDA Host BFC")

        plot_memory(ts_gpu_mem, ts_host_mem, ts_cpu_mem)
