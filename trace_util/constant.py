# event descriptions
EV_NAME = 'name'
EV_CAT	= 'cat'
EV_PH		= 'ph'
EV_TS		= 'ts'
EV_DUR	= 'dur'
EV_TTS	=	'tts'
EV_PID	=	'pid'
EV_TID	=	'tid'
EV_ARGS	=	'args'

'''
[ event types ]
Duration, Complete, Instant, Counter, Flow, Object, Metadata ...
'''
## event phase(EV_PH) lists for event types
DURATION_BEGIN = 'B'
DURATION_END		=	'E'

COMPLETE				=	'X'

INSTANT				=	'I'

COUNTER				=	'C'

FLOW_START	=	's'
FLOW_STEP	=	't'
FLOW_END	=	'f'

OBJECT_CREATED		=	'N'
OBJECT_SNAPSHOT	=	'O'
OBJECT_DESTROYED	=	'D'

METADATA					=	'M'

## event category(EV_CAT)
OP					=	'Op'
MEMORY			=	'Memory'  # have pid for allocator
DATAFLOW		=	'DataFlow'
TENSOR			=	'Tensor'

## event name(EV_NAME) lists
#### for 'OP'
PROCESS_NAME	=	'process_name'
MEMCPY_HTOD = 'MEMCPYHtoD'
MEMCPY_DTOH = 'MEMCPYDtoH'
#### for 'MEMORY'
GPU = 'GPU_0_bfc'
CUDA_HOST = 'cuda_host_bfc'
CPU = 'cpu'
#### for 'DATAFLOW'
#### for 'TENSOR'


## event args(EV_ARGS) lists
#### for 'METADATA'
'''
[METADATA args description]
name(value)
'''
GPU_SCHEDULE	=	'/job:localhost/replica:0/task:0/device:GPU:0 Compute'
GPU_COMPUTE 	= '/device:GPU:0/stream:all Compute'
GPU_MEMCPY 		= '/device:GPU:0/memcpy Compute'
GPU_TENSOR		=	'/job:localhost/replica:0/task:0/device:GPU:0 Tensors'

CPU_COMPUTE	=	'/job:localhost/replica:0/task:0/device:CPU:0 Compute'
CPU_TENSOR	=	'/job:localhost/replica:0/task:0/device:CPU:0 Tensors'

ALLOCATOR = 'Allocators'

#### for 'OBJECT_SNAPSHOT'
'''
[OBJECT_SNAPSHOT args description]
snapshot(dict)
'''
SNAPSHOT = 'snapshot'

####### args in snapshot
TENSOR_DESC = 'tensor_description'

####### args in tensor_desc
DTYPE = 'dtype'
SHAPE = 'shape'
ALLOC_DESC	=	'allocation_description'

######## allocation description
REQ_BYTES	=	'requested_bytes:'
ALLOC_BYTES	=	'allocated_bytes:'


GRADIENTS = "gradients"
MODEL = "resnet_v2_152"
BLOCK = "block"
UNIT = "unit"
EDGE = "edge"
BATCHNORM = "/BatchNorm"
CONV2D = "/Conv2D"
