import pandas as pd
import numpy as np 
import statsmodels.api as sm
import sys


def adjustColumn(df, col):
    dfSeries = df[col]
    m = np.mean(dfSeries.dropna())
    print(" ## " + str(m))
    df[col] = dfSeries.fillna(m)
    
    
MIN_BETS = 70
MIN_ROI  = 0.04 
#import spreadsheet 
df = pd.read_csv("Lives_24.06.2019.csv", sep = ";", nrows = 100000, decimal = ",")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print(df.columns)
print(df.head(5))




allIndvars = ['I5','I6','I7','I25','I27','I28','I31','I36','I37','I38',
'Strategy','PA','htp','atp','PmPH','PmPA','PALiveMinuts']
independent_categorical_variables = ['I5','I6','I7','I25','I27','I28','I31','I36','I37','I38','Strategy']
independent_continous_variables = list(set(allIndvars) - set(independent_categorical_variables))
print(independent_continous_variables)

depVariable = ['Balance']




#filter columns of interest
df = df.loc[: , allIndvars + depVariable]

#remove nan columns and replace Nan by values

adjustColumn(df, 'PA')
adjustColumn(df, 'PmPH')
adjustColumn(df, 'PmPA')
adjustColumn(df, 'PALiveMinuts')
adjustColumn(df, 'Balance')

del df["htp"]


for col in df.columns:
    listU = df[col].unique()
    if len(listU) == 1 and np.isnan(listU[0]):
        print("Column " + str(col) + " contains only Nans")
    if df[col].isnull().values.any():
        print("Column " + str(col) + " contains some Nans")


    
#data preview
print(df.shape[0])
print(df.columns)
print(df.head(10))




fs =  df.groupby("Strategy")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingStrategies = list(fs[fs > 0].index)
print(promissingStrategies)

fs =  df.groupby("I5")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI5 = list(fs[fs > 0].index)
print(promissingI5)

fs =  df.groupby("I6")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI6 = list(fs[fs > 0].index)
print(promissingI6)

fs =  df.groupby("I7")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI7 = list(fs[fs > 0].index)
print(promissingI7)

fs =  df.groupby("I27")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI27 = list(fs[fs > 0].index)
print(promissingI27)

fs =  df.groupby("I28")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI28 = list(fs[fs > 0].index)
print(promissingI28)

fs =  df.groupby("I31")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI31 = list(fs[fs > 0].index)
print(promissingI31)

fs =  df.groupby("I36")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI36 = list(fs[fs > 0].index)
print(promissingI36)

fs =  df.groupby("I38")["Balance"].sum().sort_values(ascending = False)
print(fs)
promissingI38 = list(fs[fs > 0].index)
print(promissingI38)


#promissingLoseLiga = [0,1]
#promissingQtyStrat = [2,3,4,5,6]

print("Start shrink the dataframe ...")

df = df.loc[df['Strategy'].isin(promissingStrategies), :]
print(df.shape[0])
# df = df.loc[df['LoseLiga'].isin(promissingLoseLiga), :]
# print(df.shape[0])
# df = df.loc[df['QtyStrat'].isin(promissingQtyStrat), :]
# print(df.shape[0])
df = df.loc[df['I5'].isin(promissingI5), :]
print(df.shape[0])
df = df.loc[df['I6'].isin(promissingI6), :]
print(df.shape[0])
df = df.loc[df['I7'].isin(promissingI7), :]
print(df.shape[0])
df = df.loc[df['I27'].isin(promissingI27), :]
print(df.shape[0])
df = df.loc[df['I28'].isin(promissingI28), :]
print(df.shape[0])
df = df.loc[df['I31'].isin(promissingI31), :]
print(df.shape[0])
df = df.loc[df['I36'].isin(promissingI36), :]
print(df.shape[0])
df = df.loc[df['I38'].isin(promissingI38), :]
print(df.shape[0])

print(df.head(9))


aggregation = {
'Balance': {
            'TP': 'sum',
            'NP': 'count'
           }
}

dfStats = df.groupby(independent_categorical_variables).agg(aggregation)
#col1 is total profit
col1 = dfStats.columns[0]

#col2 is number of bets on that strategy 
col2 = dfStats.columns[1]

#retrun of investments
dfStats["ROI"] = dfStats[col1]/(100*dfStats[col2])

#profitability index 
dfStats["PI"] = dfStats["ROI"] * np.sqrt(dfStats[col2])

#filter strategies with more than MIN_BETS
dfStats = dfStats.loc[dfStats[col2] > MIN_BETS, :]
 
#filter strategies with more than MIN_BETS
dfStats = dfStats.loc[dfStats["ROI"] > MIN_ROI,  :]


dfStats = dfStats.sort_values("PI", ascending=False)


print(dfStats.head(9))




#avoid zero result 
df['Result'] = (np.sign(df['Balance']) + 1)/2 * 0.99 + 0.01


dfInit = df

vals = [[0,0,0,0,0,0,1,0,0,0,1],
           [0,0,0,0,0,0,1,0,0,0,17],
           [0,0,0,0,0,0,0,0,0,0,17],
           [0,0,0,0,0,0,0,0,0,0,16],
           [0,0,0,0,0,0,1,0,0,0,15],
           [0,0,0,0,0,0,0,0,0,0,15],
           [0,0,0,0,0,0,0,0,0,0,22],
           [1,1,1,0,0,0,0,0,0,0,23],
           [1,0,0,0,0,0,0,0,0,0,19]
           ]
           
           
       
  
for val in vals:
    df = df [(df.I5 == val[0]) & (df.I6 == val[1]) & (df.I7 == val[2]) & (df.I25 == val[3]) & (df.I27 == val[4]) & (df.I28 == val[5]) & (df.I31 == val[6]) & (df.I36 == val[7]) & (df.I37 == val[8]) & (df.I38 == val[9]) & (df.Strategy == val[10])] 
    model = sm.GLM.from_formula("Result ~ PA +  atp + PmPH + PmPA + PALiveMinuts",
               family=sm.families.Binomial(), data=df)
    result = model.fit()
    print("Model fit quality " + str(result.deviance/df.shape[0]))
    df = dfInit
 
 


