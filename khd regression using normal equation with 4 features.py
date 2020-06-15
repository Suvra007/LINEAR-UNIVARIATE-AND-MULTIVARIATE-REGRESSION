import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
df=pd.read_excel("kolkata housing data.xlsx")
y=df.iloc[0:100,8].values
X=df.iloc[0:100,[1,2,3,4,6]].values
Xt=df.iloc[100:159,[1,2,3,4,6]].values
yt=df.iloc[100:159,8].values
i=0
while(i<X[:,4].size):
    if(X[i,4]=="Furnished"):
        X[i,4]=1
    elif(X[i,4]=="Semi-Furnished"):
        X[i,4]=0.5
    else:
        X[i,4]=0
    i+=1
i=0
while(i<Xt[:,4].size):
    if(Xt[i,4]=="Furnished"):
        Xt[i,4]=1
    elif(Xt[i,4]=="Semi-Furnished"):
        Xt[i,4]=0.5
    else:
        Xt[i,4]=0
    i+=1
X=X.astype(float)
Xt=Xt.astype(float)
a=X.T.dot(X)
w=np.linalg.inv(a).dot(X.T).dot(y)
h=X.dot(w)
hn=Xt.dot(w)
plt.scatter(hn,hn-yt,marker="o",c="blue",label="test values")
plt.scatter(h,h-y,marker="x",c="red",label="train values")
plt.hlines(y=0,xmin=-50,xmax=900,lw=2,color="red")
plt.xlim([-50,900])
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.legend(loc="upper left")
plt.show()
print("R^2 TRAIN")
print(r2_score(y,h))
print("R^2 TEST")
print(r2_score(yt,hn))
