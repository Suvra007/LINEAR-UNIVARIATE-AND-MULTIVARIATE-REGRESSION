from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("kolkata housing data.xlsx")
y=df.iloc[0:100,8].values
X=df.iloc[0:100,[1,3]].values
Xt=df.iloc[100:159,[1,3]].values
yt=df.iloc[100:159,8].values
a=X.T.dot(X)
w=np.linalg.inv(a).dot(X.T).dot(y)
h=X.dot(w)
hn=Xt.dot(w)
a=plt.subplot()
a.scatter(X[:,1],y,marker="x")
a.plot(X[:,1],h)
plt.show()
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

