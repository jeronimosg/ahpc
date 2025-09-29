"""
 Applied High Performance Computing
 
 Task Farming with MPI
 
 Assignment: Make an MPI task farm. A "task" is a randomly generated integer.
 To "execute" a task, the worker sleeps for the given number of milliseconds.
 The result of a task should be send back from the worker to the master. It
 contains the rank of the worker

 Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
 Date:   September 2025. Based on C++ code from February 2022
 License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""
import numpy as np
import time

# To run an MPI program we always need to include the MPI headers
from mpi4py import MPI
# Read more here: https://mpi4py.readthedocs.io/en/stable/tutorial.html

NTASKS = 5000  # number of tasks
RANDOM_SEED = 1234

def master(nworker, comm):

    # set up a random number generator and results array
    np.random.seed(RANDOM_SEED)
    task = np.random.randint(low=0, high=31, size=NTASKS)   # set up some "tasks", notice interval is [low,high[
    result = np.zeros(NTASKS, dtype=int)

    tstart = time.perf_counter()

    """
    IMPLEMENT HERE THE CODE FOR THE MASTER
    ARRAY task contains tasks to be done. Send one element at a time to workers
    ARRAY result should at completion contain the ranks of the workers that did
    the corresponding tasks
    """


    tend = time.perf_counter()

    # Print out a status on how many tasks were completed by each worker
    if nworker == 0 : # Avoid division by zero if no workers
        print("No workers available.")
        return
    
    workdone = np.zeros(nworker, dtype=int)
    for worker in range(1, nworker + 1):
        tasksdone = 0
        for itask in range(NTASKS):
            if result[itask] == worker:
                tasksdone += 1
                workdone[worker-1] += task[itask]
        #print(f"Master: Worker {worker} solved {tasksdone} tasks with work {workdone[worker-1]} units")

    print(f'Minimum work done by a worker:', np.min(workdone))
    print(f'Maximum work done by a worker:', np.max(workdone))
    print(f'Average work done by a worker: {np.mean(workdone):.2f}')
    print(f'Std dev of work done by a worker: {np.std(workdone):.2f}')
    print(f'Expected runtime without overheads:          {np.max(workdone)/1000.0:.3f} seconds')
    print(f'Minimum runtime with perfect load balancing: {np.sum(workdone)/(nworker*1000.0):.3f} seconds')
    print(f'Runtime for master process                   {(tend - tstart):.3f} seconds')

# call this function to complete the task. It sleeps for task milliseconds
def task_function(task):
    time.sleep(task / 1000.0)  # convert milliseconds to seconds

def worker(rank, comm):
    """
    IMPLEMENT HERE THE CODE FOR THE WORKER
    Use a call to "task_function" to complete a task
    """
    pass

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    nrank = comm.Get_size()    # get the total number of ranks
    rank = comm.Get_rank()     # get the rank of this process

    if rank == 0:              # rank 0 is the master
        master(nrank-1, comm)  # there is nrank-1 worker processes
    else:                      # ranks in [1:nrank] are workers
        worker(rank, comm)
