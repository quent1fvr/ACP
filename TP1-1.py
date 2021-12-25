from os import name
from re import S
import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as scp
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler





mean= [10,20]
covariance=[[25,-12], [-12,9]]
X=np.random.multivariate_normal(mean,covariance, 100)
x=X[:,0]
y=X[:,1]

plt.scatter(x,y, c='red')
plt.show()

Y=X-np.mean(X,axis=0)


plt.scatter(Y[:,0],Y[:,1], c="blue")

print(np.mean(Y, axis=0))

v=np.dot(Y[:,0],Y[:,1])/100 
print(v)


V=np.dot(Y.transpose(),Y)/100
print(V)

VP=scp.eig(V)
VP=VP[1]
print(VP)

VaP=scp.eigvals(V)

inertie_axe_1= VaP[0]/np.sum(VaP)
inertie_axe_2=VaP[1]/np.sum(VaP)

print(inertie_axe_1*100)
print(inertie_axe_2*100)

plt.arrow(0,0,VP[0][0], VP[1][0])
plt.arrow(0,0,VP[0][1], VP[1][1])
          
plt.show()







acp = PCA(svd_solver='full')
coord = acp.fit_transform(Y)

#print(acp.n_components_)

#print(acp.explained_variance_)

print(acp.components_)
print(acp.explained_variance_ratio_)


Dataframe=pd.read_csv("/Users/quent1/Desktop/activites.txt", sep='\t')
Dataframe.info()
Dataframe.shape
Dataframe.head()
Dataframe.describe()

dfc=Dataframe.iloc[:,0]
dfc1=np.array(dfc)
#print(dfc)


df= Dataframe.drop(["CIV","PAY","SEX","ACT","POP"], axis="columns")
#print(df)

name_column=df.columns
#print(name_column)e

#print(df.corr())

##pd.plotting.scatter_matrix(df)
##plt.show()


##print(pd.plotting(df))

#sommeil tele moins corrélés 
##prof transport plus corrélés

X= df.to_numpy()
scaler = StandardScaler()
##print(X)

Z=scaler.fit_transform(X)
##print(Z)
##print(np.mean(Z))
##print(np.var(Z))


acp2 = PCA(svd_solver='full')

datacp = acp2.fit_transform(Z)


#print(acp2.explained_variance_)
##print(acp2.explained_variance_ratio_)
##print(acp2.n_components_)
print(np.cumsum(acp2))
##plt.plot(acp2.explained_variance_)
##plt.show()
##plt.plot(np.cumsum(datacp))


##Qualité globale de la représentation de 90%
##print(datacp)
x=datacp[:,0]
y=datacp[:,1]

plt.scatter(x,y)
plt.title("2 premiers axes principaux")

##plt.show()

x1=datacp[:,2]
y1=datacp[:,3]

plt.scatter(x1,y1)

#plt.text(x1,y1)

plt.title("2 derniers axes principaux")

##plt.show()
##print(dfc.shape)