import gpustat
import random

def setup_gpus():
    """Adapted from https://github.com/bamos/setGPU/blob/master/setGPU.py
    """
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    best_gpu = min(pairs, key=lambda x: x[1])[0]
    return best_gpu