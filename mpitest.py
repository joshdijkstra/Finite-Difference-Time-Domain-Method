from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    full_data = list(np.arange(size) + 1)
else:
    full_data = None
data = comm.scatter(full_data)

print("rank", rank, "has data", data)

newData = comm.gather(data, root = 0)

if rank == 0:

    print("master collected", newData)
