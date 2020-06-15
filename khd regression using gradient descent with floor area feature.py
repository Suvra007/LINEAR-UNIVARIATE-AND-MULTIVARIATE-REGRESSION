from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("kolkata housing data.xlsx")
y=df.iloc[0:100,8].values
X1=df.iloc[0:100,3].values
Xt=df.iloc[100:159,3].values
yt=df.iloc[100:159,8].values
X01=np.ones(100)
X02=np.ones(58)
X1s=(X1-X1.mean())/X1.std()
Xts=(Xt-Xt.mean())/Xt.std()
theta1=0
theta0=0
alpha=0.1
errorgrad=0.0
errorgradz=0.0
i=0
for i in range(20):
    j=0
    for j in range(100):
        errorgrad+=((theta0*1+theta1*X1s[j])-y[j])*X1s[j]
        errorgradz+=((theta0*1+theta1*X1s[j])-y[j])
    theta0-=alpha*errorgradz/100
    theta1-=alpha*errorgrad/100
    h=theta0*X01+theta1*X1s
    hn=theta0*X02+theta1*Xts
    
theta0=54.7661928
theta1=79.32507441527557
h=theta0*X01+theta1*X1s
hn=theta0*X02+theta1*Xts
print("R^2 TRAIN")
print(r2_score(y,h))
print("R^2 TEST")
print(r2_score(yt,hn))
a=plt.subplot()
a.scatter(X1s,y,marker="x")
a.plot(X1s,h)
plt.show()
plt.scatter(hn,hn-yt,marker="o",c="blue",label="test values")
plt.scatter(h,h-y,marker="x",c="red",label="train values")
plt.hlines(y=0,xmin=-50,xmax=900,lw=2,color="red")
plt.xlim([-50,900])
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.legend(loc="upper left")
plt.show()
