#!/usr/bin/env python3
import numpy as np
import pandas as pd

data = pd.read_csv("latents_sd1.csv")
# avg colors
y = data.iloc[:,:3].values.astype(float)
y = y*2-1
# latents
x = data.iloc[:,3:].values.astype(float)

X = np.hstack([x,np.ones((x.shape[0],1))])
y = y.reshape(-1,3)

W, _, _, _ = np.linalg.lstsq(X,y)

P = W[:-1,:] 
b = W[-1,:]

print(P)
print(b)