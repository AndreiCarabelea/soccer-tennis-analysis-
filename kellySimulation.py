import matplotlib.pyplot as plt
import random
import math
import numpy as np


def max_drawdown(X):
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak: 
            peak = x
        else:
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
    return mdd
    
def max_drawdownLength(X):
    mdd = 0
    peak = X[0]
    peakIndex = 0
    maxDdl = 0
    for i in range(0, len(X)):
        if X[i] > peak: 
            peak = X[i]
            peakIndex = i
        else:
            ddl = i - peakIndex
            if((peak - X[i]) / peak < 0.05):
                ddl = 0
            elif ddl > maxDdl:
                maxDdl = ddl
    return maxDdl
    
initialBA =  100 
bankAccount = 100
sums = [bankAccount]
days = 300
current_day = 0
guessValue = 0.025
parallelEvents = 5
average_edge = 0.05



 

while (current_day < days):
    current_day = current_day + 1
    #random event having 0.5 < prob < 0.9
    
    gamesPerDay = parallelEvents
    dayProfit = 0
    dayStake = 0
    
    for i in  range(gamesPerDay):
        eventProbability = random.uniform(0.3,0.8)
        eventEdge = np.random.normal(average_edge, average_edge/2, 1)
       
        eventOdds = (1 + eventEdge)/eventProbability
        eventStake =  bankAccount *  guessValue / eventOdds
        dayStake = dayStake + eventStake
       
         
        #print("Stake: " + str(round(eventStake/peak, 3) * 100) + " %")
        result = random.random()
        if( result < eventProbability):
            dayProfit = dayProfit + eventStake * ( eventOdds - 1)
        else:
            dayProfit = dayProfit - eventStake
       
    if  dayStake > bankAccount: 
         break;    
         
    bankAccount = bankAccount + dayProfit
    
    #update  last peak
    if  bankAccount > initialBA or bankAccount < initialBA - 10 * guessValue * initialBA:
        initialBA = bankAccount
        
    # sum at the end of the current day        
    sums.append(bankAccount)
    
    
            

plt.ylabel('Profit')
plt.xlabel('Number of days')
plt.plot(sums, color = 'b')
plt.show()

