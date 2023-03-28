import torch
import time
import random

def f(x):
    time.sleep(0.3)
    return x

N = 100
start = time.time()
tasks = [f(i) for i in range(N)]
print(tasks)
print("tasks {:.4f}".format(time.time() - start))

start = time.time()
futures = [torch.jit.fork(f, i) for i in range(N)]
results = [torch.jit.wait(fut) for fut in futures]
print(results)
print("futures {:.4f}".format(time.time() - start))
