import numpy as np
import pandas as pd

date=pd.read_csv("I2014.csv",sep=',')
print(date.head())
print(date.dtypes)

dateSalariati=date[date["NrSal"]>0].copy()
dateSalariati["Raport CA/NrSal"]=round(dateSalariati["TotalCifraAfaceri"]/dateSalariati["NrSal"],3)
col1=["Siruta","DenLocalitate","Raport CA/NrSal"]
dateSalariati=dateSalariati[col1]
dateSalariati=dateSalariati.sort_values(by="Raport CA/NrSal",ascending=False)
dateSalariati.to_csv("Output1.csv",index=False)

#2.
datePopulatie=pd.read_csv("PopulatieLocalitati.csv")
print(datePopulatie.head())
dateComplete=pd.merge(date,datePopulatie,on="Siruta")
cerinta2=dateComplete.groupby("Judet")[["NrSal","Populatie"]].sum().reset_index()
cerinta2["Rata Ocuparii"]=cerinta2["NrSal"]/cerinta2["Populatie"]*1000
cerinta2=cerinta2[["Judet","Rata Ocuparii"]]
cerinta2=cerinta2.sort_values(by="Rata Ocuparii",ascending=False)
cerinta2.to_csv("Output2.csv",index=False)

#3.
print(dateComplete.head())
dateComplete["Profit pe locuitor"]=dateComplete["TotalProfit"]/dateComplete["Populatie"]
cerinta3=dateComplete.sort_values(by="Profit pe locuitor",ascending=False)
cerinta3=cerinta3.drop_duplicates(subset="Judet")
cerinta3=cerinta3[["Judet","DenLocalitate"]]
cerinta3=cerinta3.rename(columns={"DenLocalitate":"Localitate"})
cerinta3.to_csv("Output3.csv",index=False)

#4.
colInformatii=["NrFirme","NrSal","TotalCifraAfaceri","TotalProfit","TotalPierdere"]
dateComplete[colInformatii]=dateComplete[colInformatii].mul(dateComplete["Populatie"],axis=0)
cerinta4=dateComplete.groupby("Judet")[colInformatii+["Populatie"]].sum()
cerinta4[colInformatii]=round(cerinta4[colInformatii].div(cerinta4['Populatie'],axis=0),3)
cerinta4=cerinta4.drop(columns="Populatie")
cerinta4.to_csv("Output4.csv")