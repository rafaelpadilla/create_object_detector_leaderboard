
import torch    

total_gpus = torch.cuda.device_count()
gpus = {idx: torch.device(f"cuda:{idx}") for idx in range(total_gpus)}

def is_gpu_in_use(gpu_number):
    gpu_number = int(gpu_number)
    usage = torch.cuda.memory_allocated(gpu_number)/1024**3
    return usage > 1e-3

def get_free_gpu():
    ret = []
    for gpu_nb, device in gpus.items():
        if not is_gpu_in_use(gpu_nb):
            ret.append(device)
    return ret

import torch.multiprocessing as mp
    
def infer(free_gpu, queue):
    tensor = queue.get()

    while tensor is not None:
        # print(".")
        # free_gpu = get_free_gpu()
        # if not free_gpu:
        #     continue
        # free_gpu = free_gpu[0]
        
        item = tensor[0,0].item()
        print(f'Working on {item} @ GPU {free_gpu}')
        tensor = tensor.to(free_gpu)
        # Process
        for i in range(9999):
           tensor @ tensor.T
        print(f'Finished {item}')
        
        tensor = queue.get()

my_items = []
my_items.append(torch.ones((999,999)))
my_items.append(2*torch.ones((2*999,999)))
my_items.append(3*torch.ones((999,999)))
my_items.append(4*torch.ones((999,999)))
my_items.append(5*torch.ones((999,999)))

total_gpus = len(gpus)
split_list = [my_items[i::total_gpus] for i in range(total_gpus)]

processes = []
for gpu, _list in zip(gpus, split_list):
    gpu = gpus[gpu]
    queue = mp.Queue()
    p = mp.Process(target=infer, args=(gpu, queue))
    for item in _list:
        queue.put(item)
    queue.put(None)
    processes.append(p)

for p in processes:
    p.start()

for p in processes:
    p.join()  # wait for all subprocesses to finish

print('All work completed')
a = 123
