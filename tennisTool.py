import sys
import math
import itertools as it
from numpy import sign
from random import random
import pickle
import os.path



def gamesToMatch(w): 

	z = 1 - w;

	P6_0 = w**6
	P6_1 = 6 * P6_0 * z;
	P6_2 = 21 * P6_0 * z**2;
	P6_3 = 56 * P6_0 * z**3;
	P6_4 = 126 * P6_0 * z**4;
	P7_5 = 252 * w**7 * z**5;
	P7_6 = 504 * w**7 * z**6;

	ps = P6_0 + P6_1 + P6_2 + P6_3 + P6_4 + P7_5 + P7_6;

	if ps > 1:
		return 1;

	return ps * ps * (3 - 2 * ps)
    
    

def compute20(pSet):
    return round(pSet*pSet + math.sqrt(pSet*pSet*100)*0.01, 2)

def validScore2sets(t):
    res = 0
    for tt in t:
        res+= sign(tt[0] - tt[1])
    return abs(res) == 2

#a 3 sets match
def validScore3setsShort(t):
    res = 0
    for tt in t[0:2]:
        res+= sign(tt[0] - tt[1])
    return abs(res) == 0

# a 5 sets match
def validScore3setsLong(t):
    res = 0
    for tt in t[0:3]:
        res+= sign(tt[0] - tt[1])
    return abs(res) == 3     

   
def validScore4sets(t):
    if validScore3setsLong(t):
        return False
    
    res = 0
    for tt in t:
        res+= sign(tt[0] - tt[1])
    return abs(res) == 2 
 
def validScore5sets(t):
    res = 0
    for tt in t[0:4]:
        res+= sign(tt[0] - tt[1])
    return abs(res) == 0

def validScoreHandicap(t, handicap):
    games1 = 0
    games2 = 0
    
    for tt in t:
        games1+= tt[0]
        games2+= tt[1]
    
    return games1 + handicap > games2 
        
def validScoreOver(t, over):
    total = 0
    
    for tt in t:
        total+= tt[0] + tt[1]
        
    return total > over 
 
  
def matchToSet(p):
    max = 1
    res = 0
    for i in range(0, 1000):
        ps = i/1000
        diff = abs(setToMatch(ps) - p)
        if(diff < max):
            max = diff
            res = ps
    return res

def setToMatch(ps):
    return ps**2*(3-2*ps)

def gameToSet(w):
    return w**6*(1+6*(1- w)+21*(1-w)**2+56*(1-w)**3+126*(1-w)**4+252*w*(1-w)**5+504*(1-w)**6*w)
    
def setToGame(ps):
    max = 1
    res = 0
    for i in range(0, 1000):
        w = i/1000
        diff = abs(gameToSet(w) - ps)
        if(diff < max):
            max = diff
            res = w
    return res

def matchToGame(p):
    ps = matchToSet(p)
    return setToGame(ps)

def  basicProbTuple(t, w):
    
    global dictComputing
    if t[0] < t[1]:
        return basicProbTuple((t[1], t[0]), 1-w)
    v1 = t[0]
    v2 = t[1]
    
    if (v1,v2,w) in dictComputing:
        return dictComputing[(v1,v2,w)]
    
    res = 0
    
    if v1 == 6 and v2 == 0:
        res = w**6
    
    if v1 == 6 and v2 == 1:
        res = w**6 * 6*(1-w)
    
    if v1 == 6 and v2 == 2:
        res = w**6 * 21*(1-w)**2
    
    if v1 == 6 and v2 == 3:
        res = w**6 * 56*(1-w)**3
    
    if v1 == 6 and v2 == 4:
        res = w**6 * 126*(1-w)**4
    
    if v1 == 7 and v2 == 5:
        res = w**7 * 252*(1-w)**5
    
    if v1 == 7 and v2 == 6:
        res = w**7* 504*(1-w)**6 
     
    dictComputing[(v1,v2,w)] = res
    
    return res  



def probHelperSets(pSet):
    p20  = compute20(pSet)
    return p20*pSet + 3*(pSet**3)*(1-pSet)
        
def probTuple(t, w):
    res = 1
    for tt in t:
        res*= basicProbTuple(tt, w)
    return res
        
def p5set(pSet): 
    p20  = compute20(pSet)
    p1 = p20*pSet
    p2 = p1 + 3*(pSet**3)*(1-pSet)
    p3 = p2 + 6*(pSet**3)*(1-pSet)**2
    return round(p3,2)
    
while True:   
    ti =   input("Parse result: ? ")
    if ti == 'n':    
        pjoc = float(input("Game probability: ? "))
    else:
        games1 = float(input("Games1: ? "))
        games2 = float(input("Games2: ? "))
        pjoc = gamesToMatch(games1/(games1+games2))
        print("3set game : " + str(round(pjoc,2)))
        
        
    odds = float(input("odds: ? "))
    nsets = float(input("Number of sets: ? "))
    pSet = matchToSet(pjoc)

    dictComputing = {} 
    if os.path.exists("setsComputing.pkl"):
        dictComputing = pickle.load(open("setsComputing.pkl", "rb"))

    #underdog by odds
    
    print("First set : " + str(round(pSet,2)))
    
    if  odds >= 1.94:
        if nsets == 3:
            cps  = round(1 - (1-pSet)**2, 2)
            print("cps : " + str(cps))
        else:
            cps = round(1 - (1-pSet)**3, 2)
            print("cps: " + str(cps))
            
            p2 = round(1 - (1-pSet)**3*(3*pSet+1),2)
            print("AH+1.5s: " + str(p2))
            
            pj = round(pSet**3 + 3*pSet**3*(1-pSet) + 6*pSet**3*(1-pSet)**2, 2)
            print("5s match: " + str(pj))
         
    #favorite by odds     
    else:
        if nsets == 3:        
            p20 = round(pSet**2, 2)
            print("2-0 : " + str(p20))
        else:
            p30 = round(pSet**3, 2)
            print("3-0 : " + str(p30))
            
            ah15 = round(p30 + 3*pSet**3*(1-pSet), 2)
            print("AH-1.5s : " + str(round(ah15, 2)))
            
            pj = round(pSet**3 + 3*pSet**3*(1-pSet) + 6*pSet**3*(1-pSet)**2, 2)
            print("5s match: " + str(pj))

    list1 = [(6,0), (6,1), (6,2), (6,3), (6,4), (7,5), (7,6), (0,6), (1,6), (2,6), (3,6), (4,6), (5,7), (6,7)]
      
    if nsets > 3:
        handicap = float(input("Handicap games: ? "))
        w = matchToGame(pjoc)
        
        list3Long = list(it.product(list1, repeat = 3))
        list3Long = list(filter(lambda x: validScore3setsLong(x), list3Long))
        list4 = list(it.product(list1, repeat = 4))
        list4 = list(filter(lambda x: validScore4sets(x), list4))
        list5 = list(it.product(list1, repeat = 5))
        list5 = list(filter(lambda x: validScore5sets(x), list5))
        
        list3LongForHandicap = list(filter(lambda x: validScoreHandicap(x, handicap), list3Long))
        list4ForHandicap = list(filter(lambda x: validScoreHandicap(x, handicap), list4))
        list5ForHandicap = list(filter(lambda x: validScoreHandicap(x, handicap), list5))
        
        probHandicap = 0
        
        for t in list3LongForHandicap + list4ForHandicap + list5ForHandicap:
            probHandicap+= probTuple(t, w)
            
        print("AH%.1f: %.2f" % (handicap, probHandicap))
        
        
        over = float(input("Over games: ? "))
        
        list3LongForOver = list(filter(lambda x: validScoreOver(x, over), list3Long))
        list4ForOver = list(filter(lambda x: validScoreOver(x, over), list4))
        list5ForOver = list(filter(lambda x: validScoreOver(x, over), list5))
        
        probOver = 0
        
        
        for t in list3LongForOver + list4ForOver + list5ForOver:
            probOver+= probTuple(t, w)
            
        print("Over %.1f: %.2f" % (over, probOver))
        
            
    if nsets == 3:

        handicap = float(input("Handicap games: ? "))
        w = matchToGame(pjoc)
        
        list2 = list(it.product(list1, repeat = 2))
        list2 = list(filter(lambda x: validScore2sets(x), list2))
        list3Short = list(it.product(list1, repeat = 3))
        list3Short = list(filter(lambda x: validScore3setsShort(x), list3Short))
        
        list2ForHandicap = list(filter(lambda x: validScoreHandicap(x, handicap), list2))
        list3ShortForHandicap = list(filter(lambda x: validScoreHandicap(x, handicap), list3Short))
        probHandicap = 0
        
        for t in list2ForHandicap + list3ShortForHandicap:
            probHandicap+= probTuple(t, w)
        
        print("AH%.1f: %.2f" % (handicap, probHandicap))
        
        over = float(input("Over games: ? "))
        
        list2ForOver = list(filter(lambda x: validScoreOver(x, over), list2))
        list3ShortForOver = list(filter(lambda x: validScoreOver(x, over), list3Short))
        
        probOver = 0
        
        for t in list2ForOver + list3ShortForOver:
            probOver+= probTuple(t, w)
        
        print("Over %.1f: %.2f" % (over, probOver))
     
    abort = input("Abort: ? ")
    
    if abort == 'Y' or abort == 'y':
        break
    

    

f = open("setsComputing.pkl","wb")
pickle.dump(dictComputing,f)
f.close()
sys.exit()
    
    


