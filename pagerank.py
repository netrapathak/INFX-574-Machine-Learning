import os
import numpy as np
import copy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy as sc
import time


def matrix_modification(adj_mat): # function to modify matrix and calculate dangling vector
    # set diagonal to zero
    np.fill_diagonal(adj_mat,0)

    dang_nodes = np.zeros(len(adj_mat))
    H = copy.deepcopy(adj_mat)

    # normalize columns in the matrix
    for idx in range(len(adj_mat)):
        if np.sum(adj_mat[:,idx], axis=0) == 0:
            # print (idx, adj_mat[:,idx])
            dang_nodes[idx] = 1 # calculate dangling node
        else:
            H[:, idx] = adj_mat[:,idx]/np.sum(adj_mat[:,idx], axis=0)

#     print ("Column Normalized Matrix H:\n",H)
#     print ("\n Dangling Node Vector: ",dang_nodes)

    return H, dang_nodes


# calculate article vector
def article_vector_(total_published):
    article_vector = np.asarray(total_published).astype('float')
    A_tot = np.sum(article_vector)
    article_vector /= A_tot
    return article_vector


# function to calculate influence vector
def influence_vector(alpha, epsilon, H, dang_nodes, article_vector):
    N = len(H) # length of adjacency matrix
    
    # initializing start vector, used in iterating the influence vector
    pi_0=np.empty(N); 
    pi_0.fill(1/N) 
    
    curr_iter = pi_0
    
    # initializing maximum value for l1 norm of residual, which is difference between 2 iterations
    max_diff_iter = np.inf 
    
    num_iter = 0 # initializing number of iterations 
    
    while(max_diff_iter > epsilon):
        next_iter = (alpha * np.dot(H,curr_iter)) + ((alpha * np.dot(dang_nodes,curr_iter)) + (1-alpha))*article_vector
        max_diff_iter = np.sum(abs(next_iter-curr_iter)) # l1 norm of residual
        curr_iter = next_iter
        num_iter+=1 #incrementing for every iteration

    assert np.sum(curr_iter) == 1 # sanity check that influence vector i.e. current iteration sums to 1
    return curr_iter, num_iter # return influence vector and number of iterations



def eigen_factor_vector(H,influence_vector):
    eigen_factor = 100 * np.dot(H,influence_vector) / np.sum(np.absolute(np.dot(H,influence_vector)))
    return eigen_factor



if __name__ == "__main__": 
    alpha = 0.85
    epsilon = 0.00001
       
    # adj_mat = np.asarray([[1,0,2,0,4,3], [3,0,1,1,0,0], [2,0,4,0,1,0], [0,0,1,0,0,1], [8,0,3,0,5,2], [0,0,0,0,0,0]]).astype('float') # sample matrix
    
    N = 10748
    adj_mat = np.zeros((N,N))
    start_time = time.time()
    
    with open ("links.txt","r") as data:
        for lines in data.readlines():
            lines = lines.replace(" ","")
            source_citing = int(lines.split(",")[0])
            source_cited = int(lines.split(",")[1])
            citations = int(lines.split(",")[2])
            adj_mat[source_cited,source_citing] = citations
    
    H, dang_nodes = matrix_modification(adj_mat)
    article_vector = article_vector_(np.ones(N))
    curr_iter, num_iter = influence_vector(alpha, epsilon, H, dang_nodes, article_vector)
    eigen_factor = eigen_factor_vector(H,curr_iter)
    end_time = time.time()

    sorted_ = sorted(enumerate(eigen_factor), key=lambda x: x[1], reverse=True)[:20] # top 20 scores
    print ("Total number of iterations =",num_iter)
    print ("\nTime taken for computation =", end_time-start_time)
    rank = 1
    print ("\nTop 20 Journal Indexes and their EF Scores:")
    for journal_idx, score in sorted_:
        print("Rank {0} -> Journal Index {1} : Score {2}".format(rank,journal_idx,score))
        rank += 1

'''
a)Top 20 Journals and Their EF Scores:

Rank 1 -> Journal Index 4408 : Score 1.4481186906767658
Rank 2 -> Journal Index 4801 : Score 1.4127186417103392
Rank 3 -> Journal Index 6610 : Score 1.2350345744039224
Rank 4 -> Journal Index 2056 : Score 0.6795023571614873
Rank 5 -> Journal Index 6919 : Score 0.6648791185697114
Rank 6 -> Journal Index 6667 : Score 0.634634841504729
Rank 7 -> Journal Index 4024 : Score 0.5772329716737369
Rank 8 -> Journal Index 6523 : Score 0.48081511644794395
Rank 9 -> Journal Index 8930 : Score 0.47777264655981166
Rank 10 -> Journal Index 6857 : Score 0.43973480229889855
Rank 11 -> Journal Index 5966 : Score 0.4297177536469513
Rank 12 -> Journal Index 1995 : Score 0.38620652068869915
Rank 13 -> Journal Index 1935 : Score 0.38512026339956595
Rank 14 -> Journal Index 3480 : Score 0.3795776033159692
Rank 15 -> Journal Index 4598 : Score 0.37278900869129533
Rank 16 -> Journal Index 2880 : Score 0.3303062827175172
Rank 17 -> Journal Index 3314 : Score 0.32750789522300333
Rank 18 -> Journal Index 6569 : Score 0.31927166890567205
Rank 19 -> Journal Index 5035 : Score 0.3167790348824175
Rank 20 -> Journal Index 1212 : Score 0.3112570455380745

b)Time taken to run the code = 15.9 seconds

c)Number of iterations = 32

'''

