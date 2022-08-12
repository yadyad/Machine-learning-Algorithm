import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read data from csv
def readData(filename):
    df = pd.read_csv(filename)
    data = np.asarray(df)
    return data

def inhompolynomialFunction(x,y):
    #Polynomial kernel is being used
    #K(x,y) = (x,y)^d here let d = 2
    return np.dot(x,y)**2

def gaussian_kernel(x1, x2):
    return np.exp(-np.linalg.norm(x1-x2)**2/2*1.**2)

def findKernel(data):
    K = np.ndarray([data.shape[0],data.shape[0]])
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            K[i,j] = gaussian_kernel(data[i],data[j])
    return K

def centerKernel(kernel,N):
    one = np.ones(kernel.shape)
    one = one/N
    kOne = np.matmul(kernel,one)
    oneK = np.matmul(one,kernel)
    oneKOne = np.matmul(oneK,one)
    return kernel - oneK - kOne + oneKOne
    
    

data = readData("KernelPCA/Circledata.sec")
# color points in original classes
color_dict = {0: "red", 1: "blue"}
color_list = [color_dict[int(label)] for label in data[:,2]]
print(color_list)

N = data.shape[0]
print(N)
K = findKernel(data)
kCentered = centerKernel(K,N)
u,v = np.linalg.eig(kCentered)

#Removing imaginary part
u = np.real(u)
v = np.real(v)

u = np.flip(u[u.argsort()])
v = np.flip(v[:,u.argsort()],axis=1)

#Fetching the first two principle components
u = u[0:2]
v = v[:,[0,2]]

#Normalize eigenvectors
for i in range(2):
    scale = np.sqrt(1/(u[i]*N*v[:,i]@ v[:,i]))
    v[:,i] = v[:,i] * scale

a = []
for i in range(N):
    a.append(np.array([sum(v[k, j] * gaussian_kernel(data[k], data[i]) for k in range(N)) for j in range(2)]))

z = np.array(a)
plt.scatter(z[:, 0], z[:, 1], color=color_list)
plt.show()
