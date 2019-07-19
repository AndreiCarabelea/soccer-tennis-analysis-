import pandas as pd
import numpy as np

from os import path


if path.exists("history.pkl"):
    saved_df = pd.read_pickle("history.pkl")
else:
    saved_df = pd.DataFrame({'player1' : [], "player2": [],  "playerx": [],  "odds": []})
    

pd.set_option('display.max_rows', 500)   
inserted = False   
src=open("history.txt","r")
fline="player1,player2,playerx,odds\n"   
oline=src.readlines()

#Here, we prepend the string we want to on first line
if fline not in oline:
    oline.insert(0,fline)
    inserted = True
src.close()


#insert header into file if not already there
if inserted == True:
    src=open("history.txt","w")
    src.writelines(oline)
    src.close()


#remove duplicates entries and read into a datframe
df = pd.read_csv("history.txt")
df = df.drop_duplicates(subset=['player1', 'player2','playerx'], keep='first')

#addn all columns
df['Result'] = -1
df['Profit'] = 0
df['Stake'] = 0




#merge this one with the saved dataframe
df = pd.concat([saved_df, df], sort = 'True', ignore_index=True)
df = df.drop_duplicates(subset=['player1', 'player2','playerx'], keep='first')




print("Update results... size %d" % df.shape[0])

#add results
for (index, row) in df.iterrows():

    if row['Result'] ==  -1:
        print("player1 %s, player2 %s, selection %s" %(row['player1'], row['player2'], row['playerx']))
            
        try:
            result = int(input("Result ? "))
        except:
            result = int(input("Result ? "))
                
        df.loc[index, "Result"] = result

#add Stake and Profit columns
df.loc[:,"Stake"] = 1/df.loc[:,"odds"]
df.loc[df["Result"] == 1, "Profit"] = df["Stake"]*(df["odds"] - 1)
df.loc[df["Result"] == 0, "Profit"] =  - df["Stake"]









print("Total number of picks %d" %df.shape[0])
print(df)

hr = np.mean(df['Result'])
roi = np.sum(df['Profit'])/np.sum(df['Stake'])
avo = (1.0 + roi)/hr

df["AbsProfit"] =  np.abs(df["Profit"])
vol = np.std(df['Profit'])/np.mean(df["AbsProfit"])
del df["AbsProfit"]
                                               
print("\nHit rate %.2f/yield %.2f/average odds %.2f/volatility %.2f" % (hr, roi, avo, vol))

#persist dataframe
df.to_pickle("history.pkl")
