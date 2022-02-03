
from math import *
import numpy as np
import scipy.linalg as nlin
import scipy.sparse as sp
import scipy.sparse.linalg as lin
from joblib import Parallel, delayed
import time

start = time.time()

#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

#plt.close("all")

#This function does Binary search for x in i-th row.

def binarySearch(alist, item):
    first= 0
    last = len(alist) - 1
    found = False
    
    while first<=last and not found:
        midpoint = int((first + last)/2)
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1
                
#    print(found)            
    if (found == True):
        return midpoint
    else:
        return -1
   
def calculateTag(vector, M):
    tag = 0
    for i in range(0,M):
        P_i = 100*i + 3
        tag += np.sqrt(P_i)*vector[i]
    return tag

def adaggera(vector, i, j):
    
    if (i == j):
        val = vector[i]
    else:
        val = np.sqrt((vector[i]+1)*vector[j])
    return val


def bh_diag(N, M, Mx, My, U_int, mu, Jx, Jy, Vx, Vy):
    
    ########## Construct Basis Vectors ##########
    D = np.math.factorial(N + M - 1) / (np.math.factorial(N)*np.math.factorial(M - 1))
    #D = M(M+1)/2
    basis_vectors = (np.zeros((D,M))).astype(int)
    basis_vectors[0,:] = N
    prev_basis = 0
    new_basis = 0
    for i in range(1,M):
        basis_vectors[0,i] = 0
    
    for i in range(1, D):
            
        prev_basis = np.copy(basis_vectors[i-1])
        n_k = np.amax(np.nonzero(prev_basis[0:M-1]))
        new_basis = np.copy(prev_basis).astype(int)
        
        if(n_k>0):
            new_basis[0:n_k-1] = prev_basis[0:n_k-1]
            
        new_basis[n_k] = prev_basis[n_k] - 1
        
        if(n_k+1 <= M-1):
            new_basis[n_k+1] = N - np.sum(new_basis[0:n_k+1])
        
        if(n_k+2 <= M-1):
            new_basis[n_k+2:] = 0       
    
        basis_vectors[i,:] = np.copy(new_basis)

    curr_vector = np.reshape(basis_vectors, (D, Mx, My))
        
#    print("Basis vectors created")
    ########## Hashing The Basis Vectors ##########
    
    
    T = np.zeros(D)
    for i in range(0, D):
        T[i] = calculateTag(basis_vectors[i,:], M)
    
    indices = np.argsort(T)

    TSorted = np.sort(T, kind='quicksort')
    
    T = []
    
#    print("Basis vectors hashed")
    
    # Calculates hopping for all occupied sites and the corresponding vector that results from hopping.
    phi = pi
    data1 = []
    data2 = []
    row1 = []
    col1 = []
    coord1 = []
    row2 = []
    col2 = []
    coord2 = []
    data3 = []
    data4 = []
    row3 = []
    col3 = []
    coord3 = []
    row4 = []
    col4 = []
    coord4 = []
    tag = 0
    location = 0
    u = 0
   
    

    #two for loop outter over all basis vectors inner over neighbour sites

    for i in range(0, D):

        vector = basis_vectors[i,:]

        for jy in range(0, My-1):

            for jx in range(0, Mx-1):

                if(curr_vector[i][jx][jy]>0 and (jx-1)<0):
                    hopping_term_x = -Jx*np.sqrt((curr_vector[i][Mx-1][jy]+1)*curr_vector[i][0][jy])
                    new_basis = np.copy(vector).astype(int)
                    new_basis[0] -= 1
                    new_basis[M-1] += 1
                    tag = calculateTag(new_basis, M)
                    location = binarySearch(TSorted, tag)
                    u = indices[location]
                    data1.append(hopping_term_x)
                    coord1.append([i,u])

                if (curr_vector[i][jx][jy]>0):
                    hopping_term_x = -Jx * np.sqrt((curr_vector[i][jx+1][jy]+1)*curr_vector[i][jx][jy])*np.exp(1j*phi*curr_vector[i][jx+1][jy])
                    for j in range(0, M-1):
                        new_basis = np.copy(vector).astype(int)
                        new_basis[j] -= 1
                        new_basis[j+1] += 1
                        tag = calculateTag(new_basis, M)
                        location = binarySearch(TSorted, tag)
                        u = indices[location]
                        data1.append(hopping_term_x)
                        coord1.append([i,u])

        for jx in range(0, Mx-1):

            for jy in range(0, My-1):

                if(curr_vector[i][jx][jy]>0 and (jy-1)<0):
                    hopping_term_y = -Jy*np.sqrt((curr_vector[i][jx][My-1] + 1)*curr_vector[i][jx][0])
                    new_basis = np.copy(vector).astype(int)
                    new_basis[0] -= 1
                    new_basis[M-1] += 1
                    tag = calculateTag(new_basis, M)
                    location = binarySearch(TSorted, tag)
                    u = indices[location]
                    data2.append(hopping_term_y)
                    coord2.append([i,u])

                if (curr_vector[i][jx][jy]>0):
                    hopping_term_y = -Jy * np.sqrt((curr_vector[i][jx][jy+1]+1)*curr_vector[i][jx][jy])*np.exp(1j*phi*curr_vector[i][jx][jy+1])
                    for j in range(0, M-1):
                        new_basis = np.copy(vector).astype(int)
                        new_basis[j] -= 1
                        new_basis[j+1] += 1
                        tag = calculateTag(new_basis, M)
                        location = binarySearch(TSorted, tag)
                        u = indices[location]
                        data2.append(hopping_term_y)
                        coord2.append([i,u])

            U = 0.5*U_int*vector[j]*(vector[j]-1)
            C = mu*vector[j]*vector[j]
            if((U-C)!=0):
                data2.append((U-C))
                coord2.append([i,i])

        for jy in range(0, My-1):

            for jx in range(0, Mx-1):

                if(curr_vector[i][jx][jy]>0 and curr_vector[i][jx+1][jy]>0 and (jx-1)<0):
                    extended_term_x = Vx*curr_vector[i][Mx-1][jy]*curr_vector[i][0][jy]
                    new_basis = np.copy(vector).astype(int)
                    tag = calculateTag(new_basis, M)
                    location = binarySearch(TSorted, tag)
                    u = indices[location]
                    data3.append(extended_term_x)
                    coord3.append([i,u])

                if (curr_vector[i][jx][jy]>0 and curr_vector[i][jx+1][jy]>0):
                    extended_term_x = Vx*curr_vector[i][jx+1][jy]+1*curr_vector[i][jx][jy]

                    for j in range(0, M-1):
                        new_basis = np.copy(vector).astype(int)
                        tag = calculateTag(new_basis, M)
                        location = binarySearch(TSorted, tag)
                        u = indices[location]
                        data3.append(extended_term_x)
                        coord3.append([i,u])

        for jx in range(0, Mx-1):

            for jy in range(0, My-1):

                if(curr_vector[i][jx][jy]>0 and curr_vector[i][jx][jy+1]>0 and (jy-1)<0):
                    extended_term_y = Vy*curr_vector[i][jx][My-1]*curr_vector[i][jx][0]
                    new_basis = np.copy(vector).astype(int)
                    tag = calculateTag(new_basis, M)
                    location = binarySearch(TSorted, tag)
                    u = indices[location]
                    data4.append(extended_term_y)
                    coord4.append([i,u])

                if (curr_vector[i][jx][jy]>0 and curr_vector[i][jx][jy+1]>0):
                    extended_term_y = Vy*curr_vector[i][jx][jy+1]*curr_vector[i][jx][jy]

                    for j in range(0, M-1):
                        new_basis = np.copy(vector).astype(int)
                        tag = calculateTag(new_basis, M)
                        location = binarySearch(TSorted, tag)
                        u = indices[location]
                        data4.append(extended_term_y)
                        coord4.append([i,u])
            

    coord1 = np.reshape(coord1, (len(data1),2))
    row1 = coord1[:,0]
    col1 = coord1[:,1]

    coord2 = np.reshape(coord2, (len(data2),2))
    row2 = coord2[:,0]
    col2 = coord2[:,1]

    coord3 = np.reshape(coord3, (len(data3),2))
    row3 = coord3[:,0]
    col3 = coord3[:,1]

    coord4 = np.reshape(coord4, (len(data4),2))
    row4 = coord4[:,0]
    col4 = coord4[:,1]


    H = sp.csr_matrix((data1, (row1, col1)), (D,D))+sp.csr_matrix((data2, (row2, col2)), (D,D))+sp.csr_matrix((data3, (row3, col3)), (D,D))+sp.csr_matrix((data4, (row4, col4)), (D,D))
    

    D2 = sp.triu(H, k=1, format='csr')

    H = H + sp.csr_matrix.transpose(D2, copy = True)

    eVals, eVectors = lin.eigs(H, k=1, which='SR', maxiter = 1000000)
#    print("Eigenvectors and values calculated")
    eVals = np.real(eVals)
    eVectors = np.real(eVectors)
    groundState = eVectors[:,0]
    normalized_v = groundState / np.sqrt(np.sum(groundState**2))




    #Calculate Single Particle Density Matrix <G|a^dagger a |G>
    
    rho = np.zeros((M,M))
    
    
    for i in range(0,M):
        for j in range(0,M):
            data2 = []
            coord2 = []
            for k in range(0,D):
                
                vec = np.copy(basis_vectors[k,:])
                hopVal = adaggera(vec, i, j)
                newVec = np.copy(vec)
                newVec[i] += 1
                newVec[j] -= 1
                tag = calculateTag(newVec, M)
                location = binarySearch(TSorted, tag)
                u = indices[location]
                if hopVal !=0:
                    data2.append(hopVal)
                    coord2.append([k,u])
            
        
            coord2 = np.reshape(coord2, (len(data2),2))
            row2 = coord2[:,0]
            col2 = coord2[:,1]
            aa12 = sp.csr_matrix((data2, (row2, col2)), (D,D)) 
            newState = aa12.dot(normalized_v)
            conjtrans = np.conj(np.transpose(normalized_v))        
            rho[i,j] = conjtrans.dot(newState)
            
    
    lmbda, eVectorsRho = lin.eigs(rho, k=1, which='LR', maxiter = 10000000)


    rr = np.sum(lmbda)

    return rr


N = 3

M = 32

Mx = 8

My = 4

U = 1

Vx = 0.375

Vy = 0.375

m = np.linspace(0, 10, 60)

Jx = 0.05

Jy = 0.05


rr = np.zeros(len(m))


#rr = Parallel(n_jobs=-2, backend="multiprocessing")(delayed(bh_diag)(N[j], M, Mx, My, U, m[i], Jx, Jy, Vx, Vy) for j in range(len(N)) for i in range(0, len(m)) )

rr = Parallel(n_jobs=-2, backend="multiprocessing")(delayed(bh_diag)(N, M, Mx, My, U, m[i], Jx, Jy, Vx, Vy) for i in range(0, len(m)) )

    


print((np.real(rr))/M)
    


print 'It took', time.time()-start, 'seconds.'




    






