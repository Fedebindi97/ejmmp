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

agent_cache={} # dict with uid as keys and agents' tuples as values

#Initializes the repast4py.parameters.params dictionary with the model input parameters.
params = parameters.init_params("model1.yaml", "")
repast4py.random.init(rng_seed=params['myRandom.seed'][0]) #each rank has a seed (FOR NOW HARDCODED - FIND WAYS TO PARAMETRIZE)

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

#generate random seed
#params = parameters.init_params("model1.yaml", "")
rng = repast4py.random.default_rng 

# built-here function to check whether at least one item in a list is != 0
def any(iterable):
    for element in iterable:
        if element != 0:
            return True
    return False

# tick number
def t():
    return int(runner.schedule.tick)

class Firm(core.Agent):

    TYPE = 0
    
    def __init__(self, local_id: int, rank: int, labor:int, capital:float, minOrderDuration:int,\
                 maxOrderDuration:int, recipe: float, laborProductivity: float, maxOrderProduction: float,\
                 assetsUsefulLife: float, plannedMarkup: float, orderObservationFrequency: int, productionType: int,\
                 sectorialClass: int):
        super().__init__(id=local_id, type=Firm.TYPE, rank=rank) #uid
        self.labor=labor
        self.capital=capital
        self.capitalQ= 0
        self.unavailableLabor=0
        self.unavailableCapitalQ=0
        self.minOrderDuration=minOrderDuration
        self.maxOrderDuration=maxOrderDuration
        self.recipe = recipe
        self.laborProductivity=laborProductivity
        self.maxOrderProduction=maxOrderProduction
        self.assetsUsefulLife=assetsUsefulLife
        self.plannedMarkup=plannedMarkup
        self.orderObservationFrequency=orderObservationFrequency
        self.productionType=productionType
        self.sectorialClass=sectorialClass 
        
        self.lostProduction=0
        self.inventories=0
        self.inProgressInventories=0
        self.appRepository=[] #aPP is aProductiveProcess
        
        self.profits=0
        self.revenues=0
        self.totalCosts=0
        self.totalCostOfLabor=0
        self.totalCostOfCapital=0
        self.addedValue=0
        self.initialInventories=0
        self.grossInvestmentQ=0
        self.myBalancesheet=np.zeros((params['howManyCycles'], 20))

        self.movAvQuantitiesInEachPeriod=[]
        self.movAvDurations=[]
        
        self.productiveProcessIdGenerator=0
        self.consumptionVariation=0
        self.invGoodsCapacity=0
        self.consGoodsCapacity=0
      
        self.theCentralPlanner=0
        
    # activated by the Model
    def estimatingInitialPricePerProdUnit(self):

        total =  (1/self.laborProductivity)*params['wage']
        total += (1/self.laborProductivity)*self.recipe*params['costOfCapital']/params['timeFraction']
        total += (1/self.laborProductivity)*self.recipe/(self.assetsUsefulLife * params['timeFraction']) 
        if params['usingMarkup']: total *= (1+self.plannedMarkup)
        total *= ((self.maxOrderDuration+self.minOrderDuration)/2)
        return total       
        
        
    def settingCapitalQ(self, investmentGoodPrices):
        #############pt temporary solution
        #we temporary use this vector with a unique position as there is only one investment good at the moment
        self.priceOfDurableProductiveGoodsPerUnit = investmentGoodPrices[0] #1 
        self.currentPriceOfDurableProductiveGoodsPerUnit = investmentGoodPrices[0] #1  # the price to be paid to acquire 
                                                                                    # new capital in term of quantity

            
        #pt TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP
        #############   underlying idea:
        #               the actual initial price of durable productive goods (per unit of quantity) must be
        #               consistent with the initial cost of production of the durable productive goods;
        #
        #               the recipe set the ratio K/L where K is expressed in value;
        #
        #               having a price we implicitly set the "quantity";
        #
        #               substitution costs will consider both the change of the quantity and of the price
        #               at which the firm will pay the new productive goods;
        #
        #               the used v. unused capital measures are calculated as addenda of the capital in quantity
        #
        #               the costOfCapital (ratio of interests or rents) will be applied to the current value
        #               of the capital, after calculating the changes in quantity and then in value (considering 
        #               changes in q. and their value using the price of the new acquisitions)
        #
        #               as it evolves over time, the mean price of durable productive goods is an idiosyncratic
        #               property of the firm
        #
        #               L productivity is expressed in quantity as orders are expressed in quantity 

        self.capitalQ=self.capital/self.priceOfDurableProductiveGoodsPerUnit
        #if self.uid[0]==20000: print(t(),self.uid[0],"self.capitalQ",self.capitalQ,"row103",flush=True)

    
        
    def dealingMovAvElements(self, freq, x, y):

        self.movAvQuantitiesInEachPeriod.append(x/y)
        if len(self.movAvQuantitiesInEachPeriod) > freq: self.movAvQuantitiesInEachPeriod.pop(0) 
            
        self.movAvDurations.append(y)
        if len(self.movAvDurations) > freq: self.movAvDurations.pop(0)

        
    def receivingNewOrder(self, productionOrder: float, orderDuration):

        #creates a statistics of the values of the received order
        self.dealingMovAvElements(self.orderObservationFrequency, productionOrder, orderDuration)
        
        #decision on accepting or refusing the new order
        productionOrderQuantityByPeriod=productionOrder/orderDuration
        requiredLabor=np.ceil(productionOrderQuantityByPeriod/self.laborProductivity)
        requiredCapitalQ=requiredLabor*self.recipe/self.priceOfDurableProductiveGoodsPerUnit
        
        #create a new aPP or skip the order
        if requiredLabor <= self.labor and requiredCapitalQ <= self.capitalQ: 
            self.productiveProcessIdGenerator += 1
            productiveProcessId=(self.uid[0],self.uid[1],self.uid[2],self.productiveProcessIdGenerator)
            aProductiveProcess = ProductiveProcess(productiveProcessId,productionOrderQuantityByPeriod, \
                                                   requiredLabor, requiredCapitalQ, orderDuration,\
                                                   self.priceOfDurableProductiveGoodsPerUnit,\
                                                   self.assetsUsefulLife)
            self.appRepository.append(aProductiveProcess)

    def produce(self,model)->tuple: 
        
        #total values of the firm in the current interval unit
        self.currentTotalCostOfProductionOrder=0
        self.currentTotalOutput=0
        self.currentTotalCostOfUnusedFactors=0
        self.currentTotalLostProduction=0
        self.currentTotalCostOfLostProduction=0
        
        avgRequiredLabor=0
        avgRequiredCapitalQ=0
        
        if t()==0: self.initialInventories=0 
        else: self.initialInventories=self.inventories+self.inProgressInventories

        # activity within a time unit

        #catching key info values
        model.keyInfoTable[t(),2]+=self.labor
        model.keyInfoTable[t(),4]+=self.capitalQ

        for aProductiveProcess in self.appRepository:  

            if not aProductiveProcess.hasResources and \
                         self.labor - self.unavailableLabor >= aProductiveProcess.requiredLabor and\
                         self.capitalQ - self.unavailableCapitalQ >= aProductiveProcess.requiredCapitalQ:
                self.unavailableLabor += aProductiveProcess.requiredLabor
                self.unavailableCapitalQ += aProductiveProcess.requiredCapitalQ
                aProductiveProcess.hasResources = True 
                    
            if aProductiveProcess.hasResources: #resources may be just assigned above
                #production
                (aPPoutputOfThePeriod, aPPrequiredLabor, aPPrequiredCapitalQ, aPPlostProduction,\
                 aPPcostOfLostProduction) = aProductiveProcess.step()

                self.currentTotalOutput += aPPoutputOfThePeriod

                #catching key info values
                model.keyInfoTable[t(),0]+=aPPoutputOfThePeriod
                model.keyInfoTable[t(),1]+=aPPrequiredLabor
                model.keyInfoTable[t(),3]+=aPPrequiredCapitalQ

                cost = aPPrequiredLabor*params['wage'] \
                       + aPPrequiredCapitalQ*self.priceOfDurableProductiveGoodsPerUnit \
                                                         *params['costOfCapital']/params['timeFraction']\
                       + aPPrequiredCapitalQ*self.priceOfDurableProductiveGoodsPerUnit/ \
                         (self.assetsUsefulLife * params['timeFraction'])             
                                                       
                self.currentTotalCostOfProductionOrder += cost
                
                self.currentTotalLostProduction += aPPlostProduction
                self.currentTotalCostOfLostProduction += aPPcostOfLostProduction               
        
                if not params['usingMarkup']: self.plannedMarkup=0
                if aProductiveProcess.failure:
                    #consider markup
                    self.inProgressInventories -= cost*(aProductiveProcess.productionClock-1)*(1+self.plannedMarkup)
                    
                    #NB this is an approximation because in multiperiodal production processes the
                    #   priceOfDurableProductiveGoodsPerUnit may change, but it is a realistic
                    #   approximation in firm accounting               
                    
                else:
                    if aProductiveProcess.productionClock < aProductiveProcess.orderDuration:
                        self.inProgressInventories += cost * (1+self.plannedMarkup) #consider markup
                    else:
                        self.inventories+=cost*aProductiveProcess.orderDuration*(1+self.plannedMarkup)
                        self.inProgressInventories -= cost*(aProductiveProcess.orderDuration-1) *(1+self.plannedMarkup)
                        #consider markup (it is added in the final and subtracted by the inProgress)

        self.currentTotalCostOfUnusedFactors =  (self.labor - self.unavailableLabor)*params['wage'] + \
                                        (self.capitalQ - self.unavailableCapitalQ)*\
                                         self.priceOfDurableProductiveGoodsPerUnit*\
                                         params['costOfCapital']/params['timeFraction'] + \
                                         (self.capitalQ - self.unavailableCapitalQ) *\
                                            self.priceOfDurableProductiveGoodsPerUnit/ \
                                            (self.assetsUsefulLife * params['timeFraction'])
                                         # considering substitutions also for the idle capital

        #print("ORDER MOV AV",self.uid, sum(self.movAvQuantitiesInEachPeriod)/ len(self.movAvQuantitiesInEachPeriod), flush=True)
        #print(len(self.movAvQuantitiesInEachPeriod))
        avgRequiredLabor=np.ceil( ((sum(self.movAvQuantitiesInEachPeriod)/len(self.movAvQuantitiesInEachPeriod))/self.laborProductivity ))\
                *( sum(self.movAvDurations)/ len(self.movAvDurations) ) * (1 + params["capacityMargin"])
        # * (1 + params["capacityMargin"]) to avoid too frequently refusing incoming orders       
        
        #avgRequiredLabor=np.ceil( ((sum(self.movAvQuantitiesInEachPeriod)/len(self.movAvQuantitiesInEachPeriod)) /self.laborProductivity )\
        #        *( sum(self.movAvDurations)/ len(self.movAvDurations) ))
        
        #total cost of labor
        self.totalCostOfLabor= self.labor*params['wage']

        
        #labor adjustments (frequency at orderObservationFrequency)
        if t() % self.orderObservationFrequency == 0 and t() > 0:
            labor0=self.labor
            laborTmp=self.labor
            if self.labor > (1+params['tollerance']) * avgRequiredLabor:
                laborTmp = np.ceil((1+params['tollerance']) * avgRequiredLabor) #max accepted q. of L (firing)
            if self.labor < (1/(1+params['tollerance'])) * avgRequiredLabor:
                laborTmp = np.ceil((1/(1+params['tollerance'])) * avgRequiredLabor) #min accepted q. of L (hiring)
            self.labor=laborTmp

        
        #capital adjustments (frequency at each cycle)
        #here the following variables are disambiguated between actual and desired values, so they appear in a double shape:
        # i) capital and capitalQ, ii) desiredCapitalSubstistutions and desiredCapitalQsubstitutions
        
        self.capitalBeforeAdjustment=self.capital
        desiredCapitalQsubstitutions=0
        desiredCapitalSubstitutions=0
        requiredCapitalQincrement=0
        requiredCapitalIncrement=0

        if t() >= self.orderObservationFrequency: #no corrections before the end of the first correction interval
                                                 #where orders are under the standard flow of the firm
            capitalQmin= self.capitalQ/(1+params['tollerance'])
            capitalQmax= self.capitalQ*(1+params['tollerance'])
            
            avgRequiredCapital=avgRequiredLabor*self.recipe
            avgRequiredCapitalQ=avgRequiredCapital/self.currentPriceOfDurableProductiveGoodsPerUnit
            
            requiredCapitalSubstitution=self.capital/(self.assetsUsefulLife * params['timeFraction'])
            requiredCapitalSubstitutionQ=self.capitalQ/(self.assetsUsefulLife * params['timeFraction']) 
            
            #obsolescence  and deterioration effect
            self.capitalQ-=requiredCapitalSubstitutionQ
            self.capital-=requiredCapitalSubstitution
            
            a=(-requiredCapitalSubstitutionQ)
            
            #case I
            if avgRequiredCapitalQ < capitalQmin:
                b=avgRequiredCapitalQ-capitalQmin #being b<0
                #quantities
                if b<=a: desiredCapitalQsubstitutions=0
                if b>a: desiredCapitalQsubstitutions=abs(a)-abs(b)

                #values
                desiredCapitalSubstitutions=desiredCapitalQsubstitutions*self.currentPriceOfDurableProductiveGoodsPerUnit
            
            #case II
            if capitalQmin <= avgRequiredCapitalQ and avgRequiredCapitalQ <= capitalQmax:
                #quantities
                desiredCapitalQsubstitutions=abs(a) 
    
                #values
                desiredCapitalSubstitutions=desiredCapitalQsubstitutions*self.currentPriceOfDurableProductiveGoodsPerUnit
            
            #case III
            if avgRequiredCapitalQ > capitalQmax:
                #quantities
                desiredCapitalQsubstitutions=abs(a)
                requiredCapitalQincrement=avgRequiredCapitalQ-capitalQmax

                #values
                desiredCapitalSubstitutions=desiredCapitalQsubstitutions*self.currentPriceOfDurableProductiveGoodsPerUnit
                requiredCapitalIncrement=requiredCapitalQincrement*self.currentPriceOfDurableProductiveGoodsPerUnit
        
        self.desiredCapitalQsubstitutions=desiredCapitalQsubstitutions
        self.requiredCapitalQincrement=requiredCapitalQincrement                
        self.desiredCapitalSubstitutions=desiredCapitalSubstitutions
        self.requiredCapitalIncrement=requiredCapitalIncrement

    def allowInformationToCentralPlanner(self) -> tuple:
        return(self.desiredCapitalQsubstitutions, self.requiredCapitalQincrement,\
               self.desiredCapitalSubstitutions, self.requiredCapitalIncrement)
    
    
    def requestGoodsToTheCentralPlanner(self) -> tuple:
        return(self.desiredCapitalQsubstitutions,self.requiredCapitalQincrement,\
                           self.desiredCapitalSubstitutions, self.requiredCapitalIncrement)
    
    
    def concludeProduction(self):
        
        #action of the planner
        capitalQsubstitutions = self.investmentGoodsGivenByThePlanner[0]
        capitalQincrement = self.investmentGoodsGivenByThePlanner[1]
        capitalSubstitutions = self.investmentGoodsGivenByThePlanner[2]
        capitalIncrement = self.investmentGoodsGivenByThePlanner[3]

        #effects
        self.capitalQ+=capitalQsubstitutions+capitalQincrement 
        self.capital+=capitalSubstitutions+capitalIncrement
        self.grossInvestmentQ=capitalQsubstitutions+capitalQincrement

        #if self.uid[0]==597 and  (t()<=170 and t()>=100): print(t(),self.uid[0],"self.capital",self.capital,"self.capitalQ",self.capitalQ,flush=True)

        
        
        #total cost of capital
        self.totalCostOfCapital=self.capitalBeforeAdjustment*params['costOfCapital']/params['timeFraction']\
                                +capitalQsubstitutions*self.currentPriceOfDurableProductiveGoodsPerUnit
           

        # remove concluded aPPs from the list (backward to avoid skipping when deleting)
        for i in range(len(self.appRepository)-1,-1,-1):
            if self.appRepository[i].productionClock == self.appRepository[i].orderDuration: 
                self.unavailableLabor-=self.appRepository[i].requiredLabor
                self.unavailableCapitalQ-=self.appRepository[i].requiredCapitalQ
                del self.appRepository[i]

        return(self.currentTotalOutput, self.currentTotalCostOfProductionOrder, self.currentTotalCostOfUnusedFactors,self.inventories,\
               self.inProgressInventories, self.currentTotalLostProduction, self.currentTotalCostOfLostProduction, \
               self.labor, self.capital, self.grossInvestmentQ)
               # labor, capital modified just above
        

    def receiveSellingOrders(self, shareOfInventoriesBeingSold: float, centralPlannerBuyingPriceCoefficient: float):
        nominalQuantitySold=shareOfInventoriesBeingSold*self.inventories
        self.revenues=centralPlannerBuyingPriceCoefficient*nominalQuantitySold
        self.inventories-=nominalQuantitySold    
         
    def makeBalancesheet(self):
        self.totalCosts= self.currentTotalCostOfProductionOrder + self.currentTotalCostOfUnusedFactors
        """
        if params['usingMarkup']:
            self.inventories *= (1+self.plannedMarkup) #planned because != ex post
            self.inProgressInventories *= (1+self.plannedMarkup) 
        """
        
        self.profits= self.revenues+(self.inventories + self.inProgressInventories)\
                    -self.totalCosts-self.initialInventories 
        self.addedValue=self.profits+self.totalCosts
        
        self.myBalancesheet[t(), 0]=self.sectorialClass #i.e. row number in firms-features
        
        self.myBalancesheet[t(), 1]=self.initialInventories
        self.myBalancesheet[t(), 2]=self.totalCosts
        
        if not self.productionType in params["investmentGoods"]: self.myBalancesheet[t(), 3]=self.revenues
        else: self.myBalancesheet[t(), 4]=self.revenues

        if not self.productionType in params["investmentGoods"]: self.myBalancesheet[t(), 5]=self.inventories
        else: self.myBalancesheet[t(), 6]=self.inventories 
            
        if not self.productionType in params["investmentGoods"]: self.myBalancesheet[t(), 7]=self.inProgressInventories
        else: self.myBalancesheet[t(), 8]=self.inProgressInventories
        
        self.myBalancesheet[t(), 9]=self.profits
        self.myBalancesheet[t(), 10]=self.addedValue
        self.myBalancesheet[t(), 11]=self.currentTotalOutput
        self.myBalancesheet[t(), 12]=self.currentTotalCostOfProductionOrder
        self.myBalancesheet[t(), 13]=self.currentTotalCostOfUnusedFactors
        self.myBalancesheet[t(), 14]=self.currentTotalLostProduction
        self.myBalancesheet[t(), 15]=self.currentTotalCostOfLostProduction
        self.myBalancesheet[t(), 16]=self.totalCostOfLabor
        self.myBalancesheet[t(), 17]=self.totalCostOfCapital
        self.myBalancesheet[t(), 18]=self.grossInvestmentQ
        self.myBalancesheet[t(), 19]=self.productionType
        
        
    
    def save(self) -> Tuple: # mandatory, used by request_agents and by synchroniza
        """
        Saves the state of the Firm as a Tuple.

        Returns:
            The saved state of this instance of Firm.
        """
        # ??the structure of the save is ( ,( )) due to an incosistent use of the 
        # save output in update internal structure /fixed in v. 1.1.2???)
        return (self.uid,(self.labor,self.capital,self.minOrderDuration,self.maxOrderDuration,self.recipe,\
                self.laborProductivity,self.maxOrderProduction,self.assetsUsefulLife,self.plannedMarkup,\
                self.orderObservationFrequency,self.productionType,self.sectorialClass))

    def update(self, dynState: Tuple): # mandatory, used by synchronize
        self.labor = dynState[0]
        self.capital = dynState[1]
        self.minOrderDuration = dynState[2]
        self.maxOrderDuration = dynState[3]
        self.recipe = dynState[4]
        self.laborProductivity = dynState[5]
        self.maxOrderProduction = dynState[6]
        self.assetsUsefulLife = dynState[7]
        self.plannedMarkup = dynState[8]
        self.orderObservationFrequency = dynState[9]
        self.productionType = dynState[10]
        self.sectorialClass = dynState[11]

############################################################################################################################
###########################################################################################################################


class ProductiveProcess():
    def __init__(self, productiveProcessId: tuple, targetProductionOfThePeriod:float, requiredLabor:int,\
                 requiredCapitalQ:float, orderDuration:int, priceOfDurableProductiveGoodsPerUnit:float,\
                 assetsUsefulLife:float):
        
        self.targetProductionOfThePeriod=targetProductionOfThePeriod
        self.requiredLabor = requiredLabor
        self.requiredCapitalQ = requiredCapitalQ
        self.orderDuration = orderDuration
        self.productionClock=0
        self.hasResources= False
        self.productiveProcessId=productiveProcessId
        self.priceOfDurableProductiveGoodsPerUnit=priceOfDurableProductiveGoodsPerUnit
        self.assetsUsefulLife=assetsUsefulLife
        
    def step(self)->tuple:
        
        lostProduction=0
        costOfLostProduction=0
        self.productionClock += 1
        self.failure=False
        
        # production failure
        if params['probabilityToFailProductionChoices'] >= rng.random():
            self.failure=True
            lostProduction=self.targetProductionOfThePeriod*self.productionClock
            self.targetProductionOfThePeriod=0
            costOfLostProduction=(params['wage']* self.requiredLabor+\
                                       (params['costOfCapital']/params['timeFraction'])* self.requiredCapitalQ*\
                                        self.priceOfDurableProductiveGoodsPerUnit)*self.productionClock+\
                                        (self.requiredCapitalQ*self.priceOfDurableProductiveGoodsPerUnit)/ \
                                        (self.assetsUsefulLife * params['timeFraction']) 
            self.orderDuration = self.productionClock   

        return(self.targetProductionOfThePeriod, self.requiredLabor, self.requiredCapitalQ, \
               lostProduction, costOfLostProduction)

    

############################################################################################################################
############################################################################################################################


class CentralPlanner(core.Agent):

    TYPE = 1
    
    def __init__(self, local_id: 0, rank: 0):
        super().__init__(id=local_id, type=CentralPlanner.TYPE, rank=rank) #uid
    
        self.incrementAndSubstitutions=plannerMethods.incrementAndSubstitutions
        
        self.informationTable=np.zeros((params['howManyCycles'], 5)) #col 5 not used multiranks,
                                                                     #it only reports gross exp inv in output
        #workingUniqueOrMultiRank only
        self.informationTableMultirank=np.zeros(4)  
        self.allFirmsDesiredCapitalQsubstitutionsMultirank= 0
        self.allFirmsRequiredCapitalQincrementMultirank = 0 
        self.allFirmsDesiredCapitalSubstitutionsMultirank = 0
        self.allFirmsRequiredCapitalIncrementMultirank = 0
        #ending MultiRank, use only if rank > 0
        self.theCentralPlannerReporter=0

        self.proportionalValue=0

    def preparingActions(self, model):

        #workingUniqueOrMultiRank
        #making decisions on assigning investment goods -> one of: ['zero', 'random', 'total','proportionally']
        #IT OCCURS IN THE plannerMethods.py for all the ranks

        #getting information for actions
        if t()>0:
            #here we are summing data for each firm sectorial class  
        
            # COLLECTED INVESTMENT GOODS
            
            #the planner has to know whether it received the investment goods produced by the firms
            #and it will read it from this information table, which is updated at t-1
            
            self.informationTable[t(),0]=sum(model.totalInvGoodsRevenues[t()-1]) #inv goods bought by the planner=sum(model.totalInvGoodsRevenues[t()-1]) #inv goods bought by the planner
            self.informationTable[t(),1]=sum(model.totalInvGoodsInventories[t()-1]) #unbought and still available for the planner (stock)
            self.informationTable[t(),2]=sum(model.totalGrossInvestmentQ[t()-1])
            currentPrice=context.agent((0, 0, rank)).currentPriceOfDurableProductiveGoodsPerUnit
            self.informationTable[t(),3]=sum(model.totalGrossInvestmentQ[t()-1])*currentPrice  

        #workingUniqueOrMultiRank
        #if rank>0 sending infos to the centralPlannerReporter
        if rank > 0: self.theCentralPlannerReporter.informationTableLastCols(\
            self.informationTable[t(),0],\
            self.informationTable[t(),1],\
            self.informationTable[t(),2],\
            self.informationTable[t(),3])

    def mergeInformationTableData(self,theCentralPlannerReporterGhostList):
        #merge data from central planner reporter ghosts
        #starting with rank 0's data and adding those of the others ranks in the for cycle 
        for j in range(4):
            if rank ==0: self.informationTableMultirank[j] = self.informationTable[t(),j]
            for i in range(1,rankNum):
                self.informationTableMultirank[j]+=\
                            theCentralPlannerReporterGhostList[i-1].informationTableLastCol[j] 
                #to be used ONLY to calculate the proportionalValue in multirank runs

                
    def diffusingProductionOrders(self, float_list, int_list):
        
        #no order basic case
        if plannerMethods.noOrderGeneration:
            for aFirm in context.agents(agent_type=0):
                aFirm.receivingNewOrder(0,\
                            (aFirm.minOrderDuration + aFirm.maxOrderDuration)/2)
            return

        if t()==0:
            self.invGoodsCapacity=0
            self.consGoodsCapacity=0
            if plannerMethods.askingInvGoodsProduction == 'min' or plannerMethods.askingInvGoodsProduction == 'max':
                #comparing firms' productive capacity     
                for aFirm in context.agents(agent_type=0):
                    #if t()==30 and rank==0:print(aFirm.productionType,flush=True)
                    if aFirm.productionType in params["investmentGoods"]:
                        self.invGoodsCapacity += aFirm.labor * aFirm.laborProductivity
                        if t()==30 and rank==0:print("I",aFirm.labor * aFirm.laborProductivity,aFirm.labor,\
                                                 aFirm.laborProductivity,flush=True)
                    else:
                        self.consGoodsCapacity += aFirm.labor * aFirm.laborProductivity            
                        if t()==30 and rank==0:print("C",aFirm.labor * aFirm.laborProductivity,aFirm.labor,\
                                                 aFirm.laborProductivity,flush=True)

                self.consumptionVariation= plannerMethods.investmentVariation * self.invGoodsCapacity/self.consGoodsCapacity

        ##############################################################
        if plannerMethods.randomOrderGeneration:
            for i,aFirm in enumerate(context.agents(agent_type=0)):
                firm_float = float_list[i] 
                firm_int = int_list[i] # WE WILL NEED TO MAKE IT CONSISTENT WITH LIMITS - FOR NOW, LET'S GET THE SCRIPT TO WORK
                if plannerMethods.askingInvGoodsProduction == 'regular':
                    aFirm.receivingNewOrder(\
                        aFirm.maxOrderProduction*params["minOrderAsAShareOfMaxOrderProduction"] + \
                        firm_float * aFirm.maxOrderProduction*(1 - params["minOrderAsAShareOfMaxOrderProduction"]),\
                        rng.integers(aFirm.minOrderDuration, aFirm.maxOrderDuration+1) * plannerMethods.durationCoeff)
      
                elif (plannerMethods.askingInvGoodsProduction == 'max' and plannerMethods.investmentVariation > 0)\
                    or (plannerMethods.askingInvGoodsProduction == 'min' and plannerMethods.investmentVariation < 0):

                    #max or min (depending on how the coefficients are built)
                    if aFirm.productionType in params['investmentGoods']:
                        maxOrderProductionMod=aFirm.maxOrderProduction * (1 + plannerMethods.investmentVariation)
                        aFirm.receivingNewOrder(\
                          maxOrderProductionMod*params["minOrderAsAShareOfMaxOrderProduction"] + \
                          firm_float * maxOrderProductionMod*(1 - params["minOrderAsAShareOfMaxOrderProduction"]),\
                          rng.integers(aFirm.minOrderDuration, aFirm.maxOrderDuration+1)* plannerMethods.durationCoeff)                         

                    else:
                        maxOrderProductionMod=aFirm.maxOrderProduction * (1 - self.consumptionVariation)
                        aFirm.receivingNewOrder(\
                          maxOrderProductionMod*params["minOrderAsAShareOfMaxOrderProduction"] + \
                          firm_float * maxOrderProductionMod*(1 - params["minOrderAsAShareOfMaxOrderProduction"]),\
                          rng.integers(aFirm.minOrderDuration, aFirm.maxOrderDuration+1)* plannerMethods.durationCoeff)

                else:
                    aFirm.receivingNewOrder(0, (aFirm.minOrderDuration + aFirm.maxOrderDuration)/2)
                    print("ERROR! The investment variation coefficient must be consistent\
                    with the askingInvGoodsProduction case ('min' or 'max')")
                    
                    """
                                        
                    if plannerMethods.askingInvGoodsProduction == 'min':
                        aFirm.receivingNewOrder(rng.random()*(1/plannerMethods.investmentVariation) \
                            * aFirm.maxOrderProduction, rng.integers(aFirm.minOrderDuration, aFirm.maxOrderDuration+1))
                    if plannerMethods.askingInvGoodsProduction == 'max':
                        aFirm.receivingNewOrder(rng.random()*plannerMethods.investmentVariation \
                            * aFirm.maxOrderProduction, rng.integers(aFirm.minOrderDuration, aFirm.maxOrderDuration+1))
                    """
    
    def generateDemandOrders(self): # planner buying from firms
        #the central planner asks to firm a certain quantity of goods
        #we observe the outcome of this in the firms revenues

        for aFirm in context.agents(agent_type=0):
            shareOfInventoriesBeingSold=params['minOfInventoriesBeingSold']\
                                        + rng.random()*params['rangeOfInventoriesBeingSold']
            centralPlannerBuyingPriceCoefficient = params['centralPlannerPriceCoefficient'] #0.8 + rng.random()*0.4
            aFirm.receiveSellingOrders(shareOfInventoriesBeingSold, centralPlannerBuyingPriceCoefficient)
            
    def askFirmsInvGoodsDemand(self):
        
        self.allFirmsDesiredCapitalQsubstitutions = 0
        self.allFirmsRequiredCapitalQincrement = 0 
        self.allFirmsDesiredCapitalSubstitutions = 0
        self.allFirmsRequiredCapitalIncrement = 0

        for aFirm in context.agents(agent_type=0):
            (desiredCapitalQsubstitutions,requiredCapitalQincrement,\
                desiredCapitalSubstitutions,requiredCapitalIncrement) = aFirm.allowInformationToCentralPlanner()
            
            # TOTALIZING INVESTMENT GOODS REQUESTS        
            self.allFirmsDesiredCapitalQsubstitutions += desiredCapitalQsubstitutions
            self.allFirmsRequiredCapitalQincrement += requiredCapitalQincrement 
            self.allFirmsDesiredCapitalSubstitutions += desiredCapitalSubstitutions
            self.allFirmsRequiredCapitalIncrement += requiredCapitalIncrement

        #to report in output the gross expected investments in value
        self.informationTable[t(),4]=self.allFirmsDesiredCapitalSubstitutions+\
                                     self.allFirmsRequiredCapitalIncrement

        #workingUniqueOrMultiRank
        #if rank>0 sending infos to the centralPlannerReporter
        if rank > 0: self.theCentralPlannerReporter.invGoodsDemand(\
            self.allFirmsDesiredCapitalQsubstitutions,\
            self.allFirmsRequiredCapitalQincrement,\
            self.allFirmsDesiredCapitalSubstitutions,\
            self.allFirmsRequiredCapitalIncrement)

    def mergeInvGoodsDemand(self,theCentralPlannerReporterGhostList):
        if rank==0: 
            self.allFirmsDesiredCapitalQsubstitutionsMultirank= self.allFirmsDesiredCapitalQsubstitutions
            self.allFirmsRequiredCapitalQincrementMultirank = self.allFirmsRequiredCapitalQincrement
            self.allFirmsDesiredCapitalSubstitutionsMultirank = self.allFirmsDesiredCapitalSubstitutions
            self.allFirmsRequiredCapitalIncrementMultirank = self.allFirmsRequiredCapitalIncrement
            
        for i in range(1,rankNum):
            #print("rank",i,theCentralPlannerReporterGhostList[i-1].invGoodsDemandList,flush=True)
            self.allFirmsDesiredCapitalQsubstitutionsMultirank += \
                                       theCentralPlannerReporterGhostList[i-1].invGoodsDemandList[0]
            self.allFirmsRequiredCapitalQincrementMultirank += \
                                       theCentralPlannerReporterGhostList[i-1].invGoodsDemandList[1] 
            self.allFirmsDesiredCapitalSubstitutionsMultirank += \
                                       theCentralPlannerReporterGhostList[i-1].invGoodsDemandList[2]
            self.allFirmsRequiredCapitalIncrementMultirank += \
                                       theCentralPlannerReporterGhostList[i-1].invGoodsDemandList[3]
  
    #used if the proportionally option is active        
    def setProportionalValue(self):
        if rankNum == 1:
            if (self.allFirmsDesiredCapitalSubstitutions + self.allFirmsRequiredCapitalIncrement)!=0:
                self.proportionalValue= self.informationTable[t(),0]\
                    / (self.allFirmsDesiredCapitalSubstitutions + self.allFirmsRequiredCapitalIncrement)

        else:
            if (self.allFirmsDesiredCapitalSubstitutionsMultirank + self.allFirmsRequiredCapitalIncrementMultirank)!=0:
                self.proportionalValue = self.informationTableMultirank[0]\
                / (self.allFirmsDesiredCapitalSubstitutionsMultirank + self.allFirmsRequiredCapitalIncrementMultirank)

        
    def executeInvestmentGoodsDemandFromFirms(self):
        for aFirm in context.agents(agent_type=0):
            (desiredCapitalQsubstitutions, requiredCapitalQincrement,\
             desiredCapitalSubstitutions, requiredCapitalIncrement) = aFirm.requestGoodsToTheCentralPlanner()
            
            #UNINFORMED CENTRAL PLANNER
            
            #give all, give zero, give random quantity, regardless of its previous action
            

            
            #give zero
            if self.incrementAndSubstitutions == 'zero':
                capitalQsubstitutions = 0
                capitalQincrement = 0 
                capitalSubstitutions = 0
                capitalIncrement = 0 
        
            #give random
            if self.incrementAndSubstitutions == 'random':
                randomValue=rng.random() 
                totalQIncrementAndSubstitutions=randomValue * (desiredCapitalQsubstitutions + requiredCapitalQincrement)
                totalIncrementAndSubstitutions=randomValue * (desiredCapitalSubstitutions + requiredCapitalIncrement)
                
                if totalQIncrementAndSubstitutions >= desiredCapitalQsubstitutions:
                    #and so totalIncrementAndSubstitutions >= desiredCapitalSubstitutions
                    capitalQsubstitutions = desiredCapitalQsubstitutions
                    capitalQincrement = totalQIncrementAndSubstitutions - capitalQsubstitutions
                    capitalSubstitutions = desiredCapitalSubstitutions
                    capitalIncrement = totalIncrementAndSubstitutions - capitalSubstitutions
                else:
                    capitalQsubstitutions = totalQIncrementAndSubstitutions
                    capitalQincrement = 0
                    capitalSubstitutions = totalIncrementAndSubstitutions
                    capitalIncrement = 0

                    
            # PARTIALLY INFORMED CENTRAL PLANNER
            
            #give all
            if self.incrementAndSubstitutions == 'total':                    
                capitalQsubstitutions = desiredCapitalQsubstitutions
                capitalQincrement = requiredCapitalQincrement 
                capitalSubstitutions = desiredCapitalSubstitutions
                capitalIncrement = requiredCapitalIncrement
                
           
            # FULLY INFORMED CENTRAL PLANNER ... BUT SHY 
            
            # We introduce the informed central planner, which distritutes the goods under the label 'proportionally'
    
            if self.incrementAndSubstitutions == 'proportionally':
             
                if (self.allFirmsDesiredCapitalSubstitutions + self.allFirmsRequiredCapitalIncrement)==0: 
                    capitalQsubstitutions = 0
                    capitalQincrement = 0 
                    capitalSubstitutions = 0
                    capitalIncrement = 0 
                    
                else:

                    #if >1 firms have more K than what require and too many inventories???
                 
                    totalQIncrementAndSubstitutions=self.proportionalValue * (desiredCapitalQsubstitutions + requiredCapitalQincrement)
                    totalIncrementAndSubstitutions=self.proportionalValue * (desiredCapitalSubstitutions + requiredCapitalIncrement)
                
                    if totalQIncrementAndSubstitutions >= desiredCapitalQsubstitutions:
                        #and then totalIncrementAndSubstitutions >= desiredCapitalSubstitutions
                        capitalQsubstitutions = desiredCapitalQsubstitutions
                        capitalQincrement = totalQIncrementAndSubstitutions - capitalQsubstitutions
                        capitalSubstitutions = desiredCapitalSubstitutions
                        capitalIncrement = totalIncrementAndSubstitutions - capitalSubstitutions
                    else:
                        capitalQsubstitutions = totalQIncrementAndSubstitutions
                        capitalQincrement = 0
                        capitalSubstitutions = totalIncrementAndSubstitutions
                        capitalIncrement = 0
                
            
            # THE WISE CENTRAL PLANNER
            
            #self.informationTable[t(),1] #unbought inventories of investment goods
            # the inventories will turn out to be useful when the central planner will become wise


            
            aFirm.investmentGoodsGivenByThePlanner = (capitalQsubstitutions, capitalQincrement,\
                                                         capitalSubstitutions, capitalIncrement)

    def save(self) -> Tuple: # mandatory, used by request_agents and by synchronization
        """
        Saves the state of the CentralPlanner as a Tuple.

        Returns:
            The saved state of this CentralPlanner.
        """
        # ??the structure of the save is ( ,( )) due to an incosistent use of the 
        # save output in update internal structure /fixed in v. 1.1.2???)
        # unuseful return (self.uid,(self.incrementAndSubstitutions,)) #the comma is relevant for positional reasons
        return (self.uid,(self.proportionalValue,))

    def update(self, dynState: Tuple): # mandatory, used by synchronize
        # unuseful self.incrementAndSubstitutions = dynState[0] #just in case it should change
        self.proportionalValue=dynState[0]
        #print("rank",rank,"t",t(),"upd proportionalValue",self.proportionalValue,flush=True)

############################################################################################################################
############################################################################################################################

class CentralPlannerReporter(core.Agent):

    TYPE = 2
    
    def __init__(self, local_id: 0, rank: 0):
        super().__init__(id=local_id, type=CentralPlannerReporter.TYPE, rank=rank) #uid
    
        self.informationTableLastCol=[0,0,0,0] # superflous
        self.invGoodsDemandList=[0,0,0,0] #to avoid an error in first sync

    def informationTableLastCols(self,c0,c1,c2,c3):
        
        #workingUniqueOrMultiRank
        self.informationTableLastCol=[]
        self.informationTableLastCol.append(c0)
        self.informationTableLastCol.append(c1)
        self.informationTableLastCol.append(c2)
        self.informationTableLastCol.append(c3)

    def invGoodsDemand(self,d0,d1,d2,d3):
        #workingUniqueOrMultiRank
        self.invGoodsDemandList=[]
        self.invGoodsDemandList.append(d0)
        self.invGoodsDemandList.append(d1)
        self.invGoodsDemandList.append(d2)
        self.invGoodsDemandList.append(d3)
        #print("from invGoodsDemand, rank=",rank,"t=",t(),self.invGoodsDemandList,flush=True)
    
    def save(self) -> Tuple: # mandatory, used by request_agents and by synchronizazion
        """
        Saves the state of the CentralPlannerReporter as a Tuple.

        Returns:
            The saved state of this CentralPlannerReporter.
        """
        # ??the structure of the save is ( ,( )) due to an incosistent use of the 
        # save output in update internal structure /fixed in v. 1.1.2???)
        #return (self.uid,(self.informationTable,)) #the comma is relevant for positional reasons
        #print(rank, "save",self.informationTableLastCol,flush=True)
        return (self.uid,(self.informationTableLastCol,self.invGoodsDemandList,))

    def update(self, dynState: Tuple): # mandatory, used by synchronize
        #print(rank, "updt",dynState,flush=True)
        #print("from reporter upddat, rank=",rank,"t=",t(),dynState,flush=True)
        for i in range(4):
            self.informationTableLastCol[i]=dynState[0][i]
        for i in range(4):
            self.invGoodsDemandList[i]=dynState[1][i]

############################################################################################################################
############################################################################################################################

def restore_agent(agent_data: Tuple):

    uid=agent_data[0]
 
    if uid[1] == Firm.TYPE:
    
        if uid in agent_cache: 
            tmp = agent_cache[uid] # found
            tmp.labor = agent_data[1][0] #restore data
            tmp.capital = agent_data[1][1]
            tmp.minOrderDuration = agent_data[1][2]
            tmp.maxOrderDuration = agent_data[1][3]
            tmp.recipe = agent_data[1][4]
            tmp.laborProductivity = agent_data[1][5]
            tmp.maxOrderProduction = agent_data[1][6]
            tmp.assetsUsefulLife = agent_data[1][7]
            tmp.plannedMarkup = agent_data[1][8]
            tmp.orderObservationFrequency = agent_data[1][9]
            tmp.productionType = agent_data[1][10]
            tmp.sectorialClass = agent_data[1][11]
            
        else: #creation of an instance of the class with its data
            tmp = Firm(uid[0], uid[2],agent_data[1][0],agent_data[1][1],agent_data[1][2],agent_data[1][3],\
                       agent_data[1][4],agent_data[1][5],agent_data[1][6],agent_data[1][7],agent_data[1][8],\
                       agent_data[1][9],agent_data[1][10],agent_data[1][11])
            agent_cache[uid] = tmp
            
        return tmp

    if uid[1] == CentralPlanner.TYPE:
    
        if uid in agent_cache: 
            tmp = agent_cache[uid] # found
            tmp.incrementAndSubstitutions = agent_data[1][0] #restore data
            
        else: #creation of an instance of the class with its data
            tmp = CentralPlanner(uid[0], uid[2])
            agent_cache[uid] = tmp
            #tmp.incrementAndSubstitutions = agent_data[1][0] #not used, variable defined in init
            
        return tmp


    if uid[1] == CentralPlannerReporter.TYPE:
    
        if uid in agent_cache: 
            tmp = agent_cache[uid] # found
            #tmp.informationTable = agent_data[1][0] #restore data
            
        else: #creation of an instance of the class with its data
            tmp = CentralPlannerReporter(uid[0], uid[2])
            agent_cache[uid] = tmp
            
        return tmp


class Model:
    
    global params
    PARAMS = params
    
    def __init__(self, params: Dict):
        
        self.totalProduction=[]
        self.totalCostOfProduction=[]
        self.totalCostOfUnusedFactors=[]
        self.totalInvGoodsRevenues=[]
        self.totalConsGoodsRevenues=[]
        self.totalInvGoodsInventories=[]
        self.totalConsGoodsInventories=[]
        self.totalInProgressInvGoodsInventories=[]
        self.totalInProgressConsGoodsInventories=[]
        self.totalLostProduction=[]
        self.totalCostOfLostProduction=[]
        self.updatedLabor=[]
        self.updatedCapital=[]
        self.totalGrossInvestmentQ=[]
        self.firmData={}
        self.theCentralPlanner=0
        self.theCentralPlannerReporter=0
        self.theCentralPlannerReporterGhostList=[]

        self.keyInfoTable=np.zeros((params['howManyCycles'], 5)) 
        
        #the context and the runner are created in step 1 
      
        runner.schedule_event(          0.0,     self.initGhosts) 
        runner.schedule_event(          0.0,     self.initInvestmentGoodPrices) 
        
        runner.schedule_repeating_event(0.0,  1, self.counter)
        runner.schedule_repeating_event(0.05, 1, self.plannerPreparingActions)
        runner.schedule_repeating_event(0.06, 1, self.plannerDiffusingProductionOrders)
        runner.schedule_repeating_event(0.07, 1, self.firmsProducing)
        runner.schedule_repeating_event(0.08, 1, self.plannerPreparingAndMakingDistributionOfInvGoods)
        runner.schedule_repeating_event(0.1,  1, self.firmsConcludingProduction)
        runner.schedule_repeating_event(0.11, 1, self.firmsMakingFinancialTransactionsRelatedToCosts)
        
        runner.schedule_repeating_event(0.2,  1, self.plannerGeneratingDemandOrders) #invGoods for next period investments
        runner.schedule_repeating_event(0.21, 1, self.firmsMakingFinancialTransactionsRelatedToRevenues)
        runner.schedule_repeating_event(0.3,  1, self.enterprisesMakingBalancesheet) #enterprises=firms+banks
        
        runner.schedule_stop(params['howManyCycles'])
        runner.schedule_end_event(self.finish)
        
        ####################################################################################################
        ###################################### CREATE FIRM AGENTS ##########################################
        ####################################################################################################
        
        #importing csv file containing info about firms 
        #share of firms of that class, L min, L max, K min, K max, order duration min, order duration max, 
        #recipe, L prod, max order production, assets' useful life, planned markup, 
        #order observation frequency min, order observation frequency max, production type
        with open('firm-features.csv', newline='') as csvfile:
            firmReader= csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            
            self.rowNumber=-1
            k=0
            #for each record in firm-features.csv
            #share of firms of that class, L min, L max, K min, K max, order duration min, order duration max, recipe, 
            #L prod, max order production, assets' useful life, planned markup, order observation frequency min, 
            #order observation frequency max, production type
            for row in firmReader:
                #print(row)
                if self.rowNumber>=0:
                    for i in range(int(row[0] * params['Firm.count'])// rankNum):
                        labor= rng.integers(row[1], row[2]+1) #+1 because integers exclude extremes
                        #capital= row[3] + rng.random()*(row[4] -row[3])
                        capital= row[3] + rng.random()*(row[4] -row[3])
                        minOrderDuration= row[5]
                        maxOrderDuration= row[6]
                        recipe= row[7] #K/L 
                        laborProductivity= row[8]
                        maxOrderProduction= row[9]
                        avgAssetsUsefulLife=row[10]  #https://www.oecd.org/sdd/productivity-stats/43734711.pdf
                        plannedMarkup=row[11]
                        orderObservationFrequency=rng.integers(row[12], row[13]+1)
                        productionType=int(row[14]) #productionType in firm-features.csv indicates the production of
                                                #investment goods if it is into the investmentGoods list in yaml
                        sectorialClass=int(self.rowNumber)
                        aFirm =Firm(k, rank, labor, capital, minOrderDuration, maxOrderDuration, recipe, laborProductivity,\
                                maxOrderProduction, avgAssetsUsefulLife, plannedMarkup, orderObservationFrequency, productionType,\
                                sectorialClass)
                        context.add(aFirm)
                        k += 1 # first element of the UID of the agents
                #if rank==0 and self.rowNumber>=0: print("last firm of sectorialClass==",sectorialClass,"=",\
                                                        #aFirm.uid,flush=True)
                self.rowNumber += 1
            self.firmCount=k #one more, here is a count, not an id
        
        
        ####################################################################################################
        ################################## CREATE CENTRAL PLANNER AGENT ####################################
        ####################################################################################################
        if rank==0: 
            self.theCentralPlanner=CentralPlanner(0,0) #local_id=0, rank=0
            context.add(self.theCentralPlanner)
            #the Central Planner is an agent that we know by its id -> context.agent((0,1,0))
            #we will create a ghost of this in the other ranks
            for aFirm in context.agents(agent_type=0):
                aFirm.theCentralPlanner = self.theCentralPlanner
                
        #else:
        #   assign the central planner ghost using cache memory by calling its uid = (0,1,0)
        #   see below in initGhosts
            
  
        ####################################################################################################
        ############################# CREATE CENTRAL PLANNER REPORTER AGENT ################################
        ####################################################################################################
        if rank!=0: 
            self.theCentralPlannerReporter=CentralPlannerReporter(0,rank) #local_id=0, rank=rank
            context.add(self.theCentralPlannerReporter)
            #the Central PlannerObserve is an agent that we know by its id -> context.agent((0,2,rank))
            #we create a ghost of this in rank 0


                
    #initialize ghosts by sending them in the ranks before starting the simulation 
    def initGhosts(self):

        if rankNum==1: return #MULTIRANK only

        ghostsToRequest = [] # list of tuples containing for each ghost the uid and its current rank;
                             # used by the requestGhosts(self) function of the model

        """ TEMPORARY NO FIRMS GHOSTS
        #ghosts of class Firm, if rank is 0; the ghosts of the Firm instances are only created there 
        if rank == 0:
            rankIds=list(range(rankNum))
            rankIds.pop(rank)
 
            for rankId in rankIds:
                for i in range(self.firmCount):
                    ghostsToRequest.append( ((i,Firm.TYPE,rankId),rankId) )
        """

        #ghost of class CentralPlanner, if rank different from 0; the CentralPlanner ghosts are only created there
        if rank != 0: ghostsToRequest.append( ((0,CentralPlanner.TYPE,0),0) )

        #ghost of class CentralPlannerReporter, if rank is 0; the CentralPlannerReporter are only if rank != 0
        if rank == 0: 
            for rankId in range(1,rankNum):
                ghostsToRequest.append( ((0,CentralPlannerReporter.TYPE,rankId),rankId) )

        ###
        ###create ghosts, pulling them
        ###
        context.request_agents(ghostsToRequest,restore_agent)

        print(rank,len(ghostsToRequest),flush=True)
        print(rank,agent_cache, flush=True)
        
        print("GHOSTS: rank ",str(rank)+" concluded the creation of the ghosts",flush=True)

        
        #the central planner as a ghost, assigned to the firms
        #workingUniqueOrMultiRank
        if rank > 0: 
            self.theCentralPlanner=agent_cache[(0,1,0)]
            for aFirm in context.agents(agent_type=0):
                aFirm.theCentralPlanner = self.theCentralPlanner

        #the central planner reporter of the rank assigned to the local central planner ghost
        if rank > 0:
            self.theCentralPlanner.theCentralPlannerReporter=self.theCentralPlannerReporter 

        #the list of central planner ghosts in rank 0 (the for cycle does not work if rankNum==1)
        if rank==0:
            for i in range(1,rankNum):
                print(agent_cache[(0,2,i)],flush=True)
                self.theCentralPlannerReporterGhostList.append(agent_cache[(0,2,i)])
        
    
    #initialize investment good prices
    def initInvestmentGoodPrices(self):
        self.investmentGoodPrices=[0]*len(params['investmentGoods'])
        
        for anInvGoodType in range(len(params['investmentGoods'])):
            count=0
            for aFirm in context.agents(agent_type=0):
                if aFirm.productionType == params['investmentGoods'][anInvGoodType]:
                    self.investmentGoodPrices[anInvGoodType]+=aFirm.estimatingInitialPricePerProdUnit()
                    count+=1
            if count != 0: self.investmentGoodPrices[anInvGoodType]/=count
        
        if not any(self.investmentGoodPrices): 
            print("\nThere are no investment goods!")
            sys.exit(0)
        
        for aFirm in context.agents(agent_type=0):
            aFirm.settingCapitalQ(self.investmentGoodPrices)
            #if aFirm.uid[0]==0: print("rank",rank,"Initial price of durable productive goods per unit",\
                                      #aFirm.priceOfDurableProductiveGoodsPerUnit, flush=True) #as an info to the user
                

    #count the cycles number
    def counter(self):
        if int(t()) % params["tickNumber.betweenChecks"] == 0 and t()>9: 
            print("rank", rank, "tick", t(), flush=True)#, \
                  #"proportionalValue",self.theCentralPlanner.proportionalValue, flush=True)
            

    def plannerPreparingActions(self): 
        #workingUniqueOrMultiRank, rules are the same
        self.theCentralPlanner.preparingActions(self) # self here is the model instance
        #step made in paraller independly in all the ranks using the infos of plannerMethods.py

        ###
        ###sinchronize ghosts
        ###
        if rankNum > 1: context.synchronize(restore_agent) #theCentralPlanner diffuse infos to its ghosts
                                                           #from rank 0 to the other ranks (currentry nothing
                                                           #interesting)
                                                           #theCentralPlannerReporter send infos to its ghost
                                                           # from rank !=0 to rank 0
        #test
        #if rank==0:
        #    for i in range(1,rankNum):
        #        print(rank, t(), self.theCentralPlannerReporterGhostList[i-1].informationTableLastCol,flush=True)

        #add data collected from central planner reporter of ranks > 0 to the central planner of rank 0 data
        if rank==0 and rankNum>1: 
            self.theCentralPlanner.mergeInformationTableData(self.theCentralPlannerReporterGhostList)
            
        
    def plannerDiffusingProductionOrders(self):

        i = t()

        while r.get("octave_to_python") is None:
            time.sleep(1)
        
        # Read Octave's data
        data = r.get("octave_to_python").decode('utf-8')
        print(f"Python received: {data}")

        # Clear the octave_to_python key to avoid stale reads
        r.delete("octave_to_python")

        # Parse the received data into Python lists
        # Extract floats and integers from the result string
        float_str = data.split("Floats: [")[1].split("], Integers: [")[0]
        int_str = data.split("Integers: [")[1].split("]")[0]

        # Convert the string of numbers into Python lists
        float_list = [float(x) for x in float_str.split(", ")]
        int_list = [int(x) for x in int_str.split(", ")]

        # Clear the octave_to_python key to avoid stale reads
        r.delete("octave_to_python")

        # Send result back to Octave (later on, this "supply side feedback" will need to be moved somewhere else in the model schedule - for now, let's keep it here)
        if 'Step: '+str(i+1) in data:
            result = f'Python {i+1}'
            print(f'Python computing step {i+1}...')
            self.theCentralPlanner.diffusingProductionOrders(float_list, int_list)
            print(f"Python sending step {i+1}")
            r.set("python_to_octave", result)
    
    def firmsProducing(self):
        self.totalProduction.append([0]*(self.rowNumber)) #for each cycle adds a sub-list of lenght number of firm class types
        self.totalCostOfProduction.append([0]*(self.rowNumber))
        self.totalCostOfUnusedFactors.append([0]*(self.rowNumber))
        self.totalInvGoodsInventories.append([0]*(self.rowNumber))
        self.totalInProgressInvGoodsInventories.append([0]*(self.rowNumber))
        self.totalConsGoodsInventories.append([0]*(self.rowNumber))
        self.totalInProgressConsGoodsInventories.append([0]*(self.rowNumber))
        self.totalLostProduction.append([0]*(self.rowNumber))
        self.totalCostOfLostProduction.append([0]*(self.rowNumber))
        self.updatedLabor.append([0]*(self.rowNumber))
        self.updatedCapital.append([0]*(self.rowNumber))
        self.totalGrossInvestmentQ.append([0]*(self.rowNumber))
        
        for aFirm in context.agents(agent_type=0): #SHUFFLE to make them acting in random order
            aFirm.produce(self) # self here is the model instance
            
    def plannerPreparingAndMakingDistributionOfInvGoods(self):

        self.theCentralPlanner.askFirmsInvGoodsDemand()
        ###
        ###sinchronize ghosts
        ###
        if rankNum > 1: context.synchronize(restore_agent)

        #test
        #if rank==0:
        #    for i in range(1,rankNum):
        #        print("from M, rank=",rank,"t=", t(), \
        #              self.theCentralPlannerReporterGhostList[i-1].invGoodsDemandList,flush=True)

        ###
        ###MULTIRANK
        ###
        #add data collected from central planner reporter of ranks > 0 to the central planner of rank 0 data
        if rank==0 and rankNum>1: 
            self.theCentralPlanner.mergeInvGoodsDemand(self.theCentralPlannerReporterGhostList)

        #determining and diffusing (if multirank) the proportionalValue to be used in the case "propotionally"
        if rank==0: self.theCentralPlanner.setProportionalValue()
        ###
        ###sinchronize ghosts
        ###
        if rankNum > 1: context.synchronize(restore_agent)
            
        self.theCentralPlanner.executeInvestmentGoodsDemandFromFirms()
    
    def firmsConcludingProduction(self):
        for aFirm in context.agents(agent_type=0):
            
            tupleOfProductionResults = aFirm.concludeProduction()

            self.totalProduction[t()][aFirm.sectorialClass] += tupleOfProductionResults[0]
            self.totalCostOfProduction[t()][aFirm.sectorialClass] += tupleOfProductionResults[1]
            self.totalCostOfUnusedFactors[t()][aFirm.sectorialClass] += tupleOfProductionResults[2]
            
            if not aFirm.productionType in params["investmentGoods"]: 
                self.totalConsGoodsInventories[t()][aFirm.sectorialClass] += tupleOfProductionResults[3]
                self.totalInProgressConsGoodsInventories[t()][aFirm.sectorialClass] += tupleOfProductionResults[4]  
            else: 
                self.totalInvGoodsInventories[t()][aFirm.sectorialClass] += tupleOfProductionResults[3]
                self.totalInProgressInvGoodsInventories[t()][aFirm.sectorialClass] += tupleOfProductionResults[4]
            
            #here we will need to separate invGoods and consGoods inventories (and in progr inventories)
            #same for revenues, to be added here to the series
            self.totalLostProduction[t()][aFirm.sectorialClass] += tupleOfProductionResults[5]
            self.totalCostOfLostProduction[t()][aFirm.sectorialClass] += tupleOfProductionResults[6]
            self.updatedLabor[t()][aFirm.sectorialClass] += tupleOfProductionResults[7]
            self.updatedCapital[t()][aFirm.sectorialClass] += tupleOfProductionResults[8]
            self.totalGrossInvestmentQ[t()][aFirm.sectorialClass] += tupleOfProductionResults[9]

            #catching key info values 
            self.keyInfoTable[t(),4]+=aFirm.capitalQ            #total capital of the firm, in Q


    def firmsMakingFinancialTransactionsRelatedToCosts(self):
        pass
    
    def plannerGeneratingDemandOrders(self):
        #currently independent processes, so also operating multirank
        self.theCentralPlanner.generateDemandOrders()
        
    def firmsMakingFinancialTransactionsRelatedToRevenues(self):
        pass
    
    def enterprisesMakingBalancesheet(self):
        self.totalInvGoodsRevenues.append([0]*(self.rowNumber))
        self.totalConsGoodsRevenues.append([0]*(self.rowNumber))
        
        for aFirm in context.agents(agent_type=0):
            aFirm.makeBalancesheet()
            self.totalConsGoodsRevenues[t()][aFirm.sectorialClass] += aFirm.myBalancesheet[t(), 3]
            self.totalInvGoodsRevenues[t()][aFirm.sectorialClass] += aFirm.myBalancesheet[t(), 4]
      
    
    #finish
    def finish(self):
        
        print("cpu time - calculating phase", Tc(), "rank", rank, flush=True)
        
        # infos for data_analysis*.ipynb
        with open('plotInfo.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow((params["log_file_root"],rankNum,\
                             context.size(agent_type_ids=[0])[0]))
        
        
        #series____________________________________________________
        
        names=["_total_production_","_total_cost_of_production_","_total_cost_of_unused_factors_",\
               "_total_inv_goods_revenues_", "_total_cons_goods_revenues_",\
               "_total_inv_goods_inventories_","_total_in_progress_inv_goods_inventories_",\
               "_total_cons_goods_inventories_","_total_in_progress_cons_goods_inventories_",\
               "_total_lost_production_","_total_cost_of_lost_production_","_updatedLabor_","_updatedCapital_",\
               "_total_grossInvestmentQ_"]
        contents=[self.totalProduction,self.totalCostOfProduction,self.totalCostOfUnusedFactors,
                  self.totalInvGoodsRevenues, self.totalConsGoodsRevenues, 
                  self.totalInvGoodsInventories,self.totalInProgressInvGoodsInventories,
                  self.totalConsGoodsInventories,self.totalInProgressConsGoodsInventories,
                  self.totalLostProduction,self.totalCostOfLostProduction,
                  self.updatedLabor,self.updatedCapital, self.totalGrossInvestmentQ]
        
        for s in range(len(names)):
            with open(params["log_file_root"]+names[s]+str(rank)+'.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for k in range(params["howManyCycles"]):
                    writer.writerow(contents[s][k])

        
        #balancesheets______________________________________________
        #via pickle
        
        #creating a dictionary of firm dataframes
        #firmData={} defined in __init__
        colNames=["firm class type", "initial inventories","total costs", "revenuesCons", "revenuesInv", "consGoods inventories",\
       "invGoods inventories",  "consGoods in progr. inventories", "invGoods in progr. inventories", "profits", \
          "added value", "total production", "cost of production", "cost of unused factors", "total lost production", \
          "total cost of lost production", "cost of labor", "cost of capital", "gross investment in Q",\
            "production type"]
        

        for aFirm in context.agents(agent_type=0):
            self.firmData[aFirm.uid]=pd.DataFrame(aFirm.myBalancesheet)
            self.firmData[aFirm.uid].columns=colNames

        pickle.dump(self.firmData, open(params["log_file_root"]+'_balancesheetDict.p', "wb"))

        #workingUniqueOrMultiRank
        np.savetxt("plannerInfo.csv", self.theCentralPlanner.informationTable, delimiter=",")
        np.savetxt("keyInfoTable.csv", self.keyInfoTable,  delimiter=",")
        print("cpu time - finishing phase", Tc(), "rank", rank, flush=True)

        ttt=datetime.datetime.now()
        lastRandom=rng.random()
        print("version",version,"execution",ttt,"last random",lastRandom,flush=True)
        with open("_signature.txt", "w") as f:
            print("version "+version+" execution "+str(ttt)+" last random ",lastRandom,file=f)
            
        print("THE END!", flush=True)
    
    def start(self):
        runner.execute()

def run(params: Dict):
    
    model = Model(params) 
    model.start()
    
run(params)