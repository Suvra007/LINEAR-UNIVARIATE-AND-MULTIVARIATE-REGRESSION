import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
df=pd.read_excel("kolkata housing data.xlsx")
y=df.iloc[0:100,8].values
X1=df.iloc[0:100,[2,3,4,6]].values
Xt=df.iloc[100:159,[2,3,4,6]].values
yt=df.iloc[100:159,8].values
i=0
while(i<X1[:,3].size):
    if(X1[i,3]=="Furnished"):
        X1[i,3]=1
    elif(X1[i,3]=="Semi-Furnished"):
        X1[i,3]=0.5
    else:
        X1[i,3]=0
    i+=1
i=0
while(i<Xt[:,3].size):
    if(Xt[i,3]=="Furnished"):
        Xt[i,3]=1
    elif(Xt[i,3]=="Semi-Furnished"):
        Xt[i,3]=0.5
    else:
        Xt[i,3]=0
    i+=1
X1s=X1.copy()
Xts=Xt.copy()
X1s=(X1-X1.mean())/X1.std()
Xts=(Xt-Xt.mean())/Xt.std()
X01=np.ones(100)
X02=np.ones(58)
theta=0
thetaz=0
theta0=0
theta1=0
theta2=0
theta3=0
alpha=0.1
errorgrad0=0.0
errorgradz=0.0
errorgrad1=0.0
errorgrad2=0.0
errorgrad3=0.0
i=0
for i in range(20):
    j=0
    for j in range(100):
        
        g=thetaz*1+theta0*X1s[j,0]+theta1*X1s[j,1]+theta2*X1s[j,2]+theta3*X1s[j,3]

        errorgradz+=g-y[j]
        errorgrad0+=(g-y[j])*X1s[j,0]
        errorgrad1+=(g-y[j])*X1s[j,1]
        errorgrad2+=(g-y[j])*X1s[j,2]
        errorgrad3+=(g-y[j])*X1s[j,3]
    thetaz-=alpha*errorgradz/100
    theta0-=alpha*errorgrad0/100
    theta1-=alpha*errorgrad1/100
    theta2-=alpha*errorgrad2/100
    theta3-=alpha*errorgrad3/100
    h=thetaz*X01+theta0*X1s[:,0]+theta1*X1s[:,1]+theta2*X1s[:,2]+theta3*X1s[:,3]
    hn=thetaz*X02+theta0*Xts[:,0]+theta1*Xts[:,1]+theta2*Xts[:,2]+theta3*Xts[:,3]
   
    
thetaz=6.391199999999999
theta0= -2.791401881692439
theta1= 20.429469420084285
theta2= -2.797153026395628
theta3=-2.8236626291555535
h=thetaz*X01+theta0*X1s[:,0]+theta1*X1s[:,1]+theta2*X1s[:,2]+theta3*X1s[:,3]
hn=thetaz*X02+theta0*Xts[:,0]+theta1*Xts[:,1]+theta2*Xts[:,2]+theta3*Xts[:,3]
print("R^2 TRAIN")
print(r2_score(y,h))
print("R^2 TEST")
print(r2_score(yt,hn))
plt.scatter(hn,hn-yt,marker="o",c="blue",label="test values")
plt.scatter(h,h-y,marker="x",c="red",label="train values")
plt.hlines(y=0,xmin=-50,xmax=900,lw=2,color="red")
plt.xlim([-50,900])
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.legend(loc="upper left")
plt.show()
       
        
