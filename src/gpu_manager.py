import torch
from typing import Optional, Callable, List, Dict
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, wait
from torch.multiprocessing import set_start_method

class MultiGPUManager:
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        self.gpus = {idx: torch.device(f"cuda:{idx}") for idx in range(num_gpus)}

    def _run_on_gpu(self, target_fn, lst, gpu_to_use, results, kwargs):
        with ThreadPoolExecutor() as executor:
            while len(lst) != 0:
                executors = []
                _kwargs = lst.pop()
                _kwargs["device"] = gpu_to_use
                kwargs.update(_kwargs)
                ex = executor.submit(target_fn, **kwargs)
                executors.append(ex)
                complete_futures, incomplete_futures = wait(executors, return_when="FIRST_COMPLETED")
                for f in complete_futures:
                    result = f.result()
                    results.put(result)
            results.put(None)
        
    def run(self, target_fn: Callable, iterable_items: List, kwargs: Optional[Dict] = {}, spawn=True):
        total_gpus = len(self.gpus)
        total_runners = len(iterable_items)
        k, m = divmod(total_runners, total_gpus)
        split_list = [iterable_items[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(total_gpus)]

        processes = []
        
        if spawn:
            set_start_method("spawn", force=True)
        
        for device_id, lst in zip(self.gpus, split_list):
            device = self.gpus[device_id]
            res = Queue()
            p = Process(target=self._run_on_gpu, args=(target_fn, lst, device, res, kwargs))
            p.start()
            processes.append((p, res))

        for p, _ in processes:
            p.join()  # wait for all subprocesses to finish

        results = []
        for _, response in processes:    
            res = response.get()
            while res:
                results.append(res)
                res = response.get()        
        return results
    