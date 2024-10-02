import plannerMethods
import time
import datetime
from mpi4py import MPI
from repast4py import context as ctx
import repast4py 
from repast4py import parameters
from repast4py import schedule
from repast4py import core
from math import ceil
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import pickle
import csv
import os
import sys
import redis
from model1_3_FB_classes import *

r = redis.Redis(host='localhost', port=6379, db=0)
r.flushdb()
r.set("python_to_octave", "Start")

version="1.3"

comm = MPI.COMM_WORLD
rank    = comm.Get_rank()
rankNum = comm.Get_size() 

# create the context to hold the agents and manage cross process
# synchronization
context = ctx.SharedContext(comm)

# Initialize the default schedule runner, HERE to create the t() function,
# returning the tick value
runner = schedule.init_schedule_runner(comm)

# tick number
def t():
    return int(runner.schedule.tick)

#Initializes the repast4py.parameters.params dictionary with the model input parameters.
params = parameters.init_params("model1.yaml", "")
params = {**params,**{'r':r,'runner':runner,'rank':rank,'rankNum':rankNum}}


if os.path.isdir(params["log_file_root"]+"."+str(rank)):
    os.system("rm -R "+params["log_file_root"]+"."+str(rank))  
os.makedirs(params["log_file_root"]+"."+str(rank)) 

#copy in the output folder the starting set of parameters
os.system("cp model1.yaml "+params["log_file_root"]+"."+str(rank)+"/")
os.system("cp firm-features.csv "+params["log_file_root"]+"."+str(rank)+"/")
os.system("cp plannerMethods.py "+params["log_file_root"]+"."+str(rank)+"/")

if rank==0:
    i=0
    while os.path.isdir(params["log_file_root"]+"."+str(rankNum+i)):
        os.system("rm -R "+params["log_file_root"]+"."+str(rankNum+i))
        i+=1
    

    
#moves to the right folder (that you must create and initialize with a firm-features.csv file)
if not os.path.isdir(params["log_file_root"]+"."+str(rank)):
    print("There is no "+params["log_file_root"]+"."+str(rank) + " starting folder!")  
    sys.exit(0)
else: os.chdir(params["log_file_root"]+"."+str(rank))


        
#dentro a home/model1: "ls "+"../"+params["log_file_root"]+"."+str(rankNum+i))

#generate random seed
repast4py.random.init(rng_seed=params['myRandom.seed'][rank]) #each rank has a seed
rng = repast4py.random.default_rng 


#timer T()
startTime=-1
def T():
    global startTime
    if startTime < 0:
        startTime=time.time()
    return time.time() - startTime
T() #launches the timer

#cpuTimer Tc()
startCpuTime=-1
def Tc():
    global startCpuTime
    if startCpuTime < 0:
        startCpuTime=time.process_time()
    return time.process_time() - startCpuTime
Tc() #launches the cpu timer

agent_cache={} # dict with uid as keys and agents' tuples as values

def run(params: Dict):
    
    model = Model(params) 
    model.start()
    
run(params)