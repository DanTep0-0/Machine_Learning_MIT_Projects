import numpy as np
import kmeans
import common
import naive_em
import em
import em_older_version


X = np.loadtxt("netflix_incomplete.txt") # (1)  (2) "test_incomplete.txt" (3)  / "netflix_complete.txt"
#add_nulls = np.random.randint(0, 2, X.shape)
#X = np.multiply(X, add_nulls)

#print(add_nulls)
#print(X)
#breakpoint()
# print("X: " + str(X))

### K-Means with Gaussian distance function ###

K = [12]
seed = [1]

costs = np.ones((len(K), len(seed)))
plts = np.zeros((len(K), len(seed)))
bics = np.zeros((len(K), len(seed)))

for i, k in enumerate(K):
    for j, current_seed in enumerate(seed):
        mixture, post = common.init(X, k, current_seed)
        #mixture, post, cost = kmeans.run(X, mixture, post) # K-means
        #mixture, post, cost = naive_em.run(X, mixture, post) # EM Naive Approach
        mixture, post, cost = em.run(X, mixture, post) # EM for missing and big data
        #mixture, post, cost = em_older_version.run(X, mixture, post) # EM_older version for missing and big data
        costs[i, j] = cost
        bics[i, j] = common.bic(X, mixture, cost)

#mins = np.argmin(costs, axis=1) # Kmeans
mins = np.argmax(costs, axis=1) # EM
best_seeds = np.max(bics, axis=1)
best_K = K[np.argmax(best_seeds)]
best_seed = seed[np.argmax(bics[np.argmax(best_seeds)])]


# for i, s in enumerate(mins):
#     mixture, post = common.init(X, K[i], seed[s])
#     #mixture, post, cost = kmeans.run(X, mixture, post) # K-means
#     #mixture, post, cost = naive_em.run(X, mixture, post) # EM Naive Approach
#     mixture, post, cost = em.run(X, mixture, post) # EM for missing and big data
#     plt = common.plot(X, mixture, post, "Title")
#     plt.savefig('plt_mixture_k_' + str(K[i]) + '_seed_' + str(seed[s]))

print(costs)
print("Best K = " + str(best_K) + ", best K bic = " + str(np.max(best_seeds)) + ", best seed = " + str(best_seed))


### Comparing EM-results and actual values ###
X_gold = np.loadtxt("netflix_complete.txt")
mixture, post = common.init(X, best_K, best_seed)
mixture, post, cost = em.run(X, mixture, post)

X_predicted = em.fill_matrix(X, mixture, way="soft")
squared_error = common.rmse(X_gold, X_predicted)

X_predicted = em.fill_matrix(X, mixture, way="int")
mask_0 = X == 0
prozent_error = common.prozent_error(X_gold, X_predicted, mask_0)
print("Squared Error of prediction = " + str(squared_error))
print("Prozent Error of prediction = " + str(prozent_error))
#print("X_predicted[:2, : 10]: " + str(X_predicted[:2, : 10]))