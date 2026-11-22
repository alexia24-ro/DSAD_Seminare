import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# 1. Load hernia.csv, separate predictors vs class.
date=pd.read_csv("hernia.csv")
print(date.head())
#print(date.dtypes)
x=date.drop(columns=["No.","class"])
y=date["class"]
print (x.head())
print(y.head())

#2. Counts per class → clase_count.csv.
clase_count=y.value_counts()
print(clase_count)
clase_count.to_csv("clase_count.csv")

#3. Group means of predictors by class → medii_predictori_pe_clasa.csv.
medii=date.groupby("class")[x.columns].mean()
print(medii)
medii.to_csv("medii_predictori_pe_clasa.csv")

#4. One plot (e.g. degree_spondylolisthesis) by class.
date.boxplot(column="degree_spondylolisthesis",by="class")
plt.title("Degree spondylolisthesis by class")
plt.suptitle("")
plt.xlabel("Class")
plt.ylabel("Degree spondylolisthesis")
plt.show()

#5. train_test_split(0.3, stratify, random_state=42)
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, stratify=y, random_state=42)

#6. Fit LinearDiscriminantAnalysis on train; print classes_, priors_.
lda=LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
print("---- LDA ------")
print(lda.classes_)
print(lda.priors_)

#7. Predict on test → Predictii_test.csv.
y_pred=lda.predict(x_test)
predictii=pd.DataFrame({
    "Clasa_reala":y_test,
    "Clasa_prezisa":y_pred
})
predictii.to_csv("predictii_test.csv",index=False)

#8. Accuracy + confusion matrix → Acuratete_lda.csv, CM_lda.csv.
acuratete=accuracy_score(y_test,y_pred)
print(acuratete)
cm=confusion_matrix(y_test,y_pred)
print(cm)
pd.DataFrame({
    "Acuratete":[acuratete]
}).to_csv("acuratete_lda.csv",index=False)
pd.DataFrame(cm, index=lda.classes_, columns=lda.classes_).to_csv("CM_lda.csv")

#9. transform() on train → z.csv (Z1, Z2).
z= lda.transform(x_train)
z=pd.DataFrame(z,columns=["z1","z2"])
z.to_csv("z.csv",index=False)
print(z.shape)

#10. Scatter Z1 vs Z2, coloured by class + group centres.
z["class"]=y_train.to_numpy()

for clasa in z["class"].unique():
    date_clasa=z[z["class"]==clasa] # filtrare conditionala - pastreaza doar inregistrarile aferente clasei curente
    plt.scatter(date_clasa["z1"],date_clasa["z2"],label=clasa)
centre=z.groupby("class")[["z1","z2"]].mean()
plt.scatter(centre["z1"],centre["z2"],color="black",marker="X",s=200,label="Centre")
plt.xlabel("z1")
plt.ylabel("z2")
plt.title("Reprezentarea claselor prin LDA")
plt.legend()
plt.show()

#11. Distributions of predictors by class (as in f_distributii).
for col in x.columns:
    date.boxplot(column=col,by="class")
    plt.title(f"Distributia lui {col} pe clase")
    plt.suptitle("")
    plt.xlabel("Clasa")
    plt.ylabel(col)
    plt.show()

#12. Short interpretation: which predictor “looks” most discriminative?
# Predictorul care pare cel mai discriminativ este degree_spondylolisthesis,
# deoarece separa foarte bine clasa Spondylolisthesis de celelalte doua clase, Hernia si Normal.
# Valorile pentru Spondylolisthesis sunt mult mai mari, in timp ce pentru Hernia si Normal sunt apropiate de 0.
# Totusi, pentru a face diferenta intre clasele Hernia si Normal, sunt mai utili predictorii sacral_slope si lumbar_lordosis_angle,
# deoarece se observa diferente mai clare intre valorile mediane.
# In schimb, pelvic_radius este cel mai putin discriminativ, deoarece valorile celor trei clase se suprapun foarte mult.