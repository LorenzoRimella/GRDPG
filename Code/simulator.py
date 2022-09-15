import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class GRDPG:

    def __init__(self, p, q, N):

        self.p = p
        self.q = q
        self.I      = tf.concat((tf.ones(p), -tf.ones(q)), axis = 0)
        self.N      = N

    def sample_Beta(self, beta_a, beta_b):

        X      = tfp.distributions.Beta(beta_a, beta_b).sample((self.N, self.p+self.q))
        Xminus = X*tf.reshape(self.I, (1, self.p+self.q))

        M = tf.einsum("ik,jk->ij", Xminus, X)

        M = tf.reduce_min(tf.concat((tf.reshape(M, (self.N, self.N, 1)), tf.ones( (self.N, self.N, 1))), axis =2), axis =2)
        M = tf.reduce_max(tf.concat((tf.reshape(M, (self.N, self.N, 1)), tf.zeros((self.N, self.N, 1))), axis =2), axis =2)

        A = tfp.distributions.Binomial(1, probs = tf.experimental.numpy.triu(M, k = 1)).sample()

        return A + tf.transpose(tf.experimental.numpy.triu(A, k = 1))

    def sample(self, X):

        Xminus = X*tf.reshape(self.I, (1, self.p+self.q))

        M = tf.einsum("ik,jk->ij", Xminus, X)

        M = tf.reduce_min(tf.concat((tf.reshape(M, (self.N, self.N, 1)), tf.ones( (self.N, self.N, 1))), axis =2), axis =2)
        M = tf.reduce_max(tf.concat((tf.reshape(M, (self.N, self.N, 1)), tf.zeros((self.N, self.N, 1))), axis =2), axis =2)

        A = tfp.distributions.Binomial(1, probs = tf.experimental.numpy.triu(M, k = 1)).sample()

        return A + tf.transpose(tf.experimental.numpy.triu(A, k = 1))


class stochasticBlockModel:
    
    def __init__(self, B):

        self.B = B

    def pop_sample(self, initial_distribution, N):

        Erv        = tfp.distributions.OneHotCategorical(probs = initial_distribution)

        return tf.cast(Erv.sample(N), dtype = tf.float32)

    def sample(self, E): 

        M = tf.einsum("nj,kj->nk", tf.einsum("nk,kj->nj", E, self.B), E)

        A = tfp.distributions.Binomial(1, probs = tf.experimental.numpy.triu(M, k = 1)).sample()

        return A + tf.transpose(tf.experimental.numpy.triu(A, k = 1))

    def to_GRDPG(self, E): 
        eigenvalues, eigenvectors = tf.linalg.eig(self.B)

        eigenvalues  = tf.math.real(eigenvalues )
        eigenvectors = tf.math.real(eigenvectors)

        eigenvalues_sort    = tf.sort(   eigenvalues, direction='DESCENDING')
        eigenvalues_argsort = tf.argsort(eigenvalues, direction='DESCENDING')
        eigenvectors_sort   = tf.gather(eigenvectors, eigenvalues_argsort, axis = 1)

        q = sum(eigenvalues_sort.numpy()<0)
        p = eigenvalues_sort.shape[0] - q

        eigenvect_eigenval = tf.einsum("ij,j->ij", eigenvectors_sort, tf.sqrt(tf.abs(eigenvalues_sort)))

        #E_sort = tf.gather(E, eigenvalues_argsort, axis = 1)          

        return p, q, eigenvect_eigenval, tf.einsum("nj,jk->nk", E, eigenvect_eigenval) #, eigenvalues_argsort


# class BlockModel:
        
#     def __init__(self, B):

#         self.B = B

#     def pop_sample(self, initial_distribution, N): #, beta_a, beta_b):

#         # Correction = tfp.distributions.Beta(beta_a, beta_b)
#         Erv        = tfp.distributions.OneHotCategorical(probs = initial_distribution)

#         return tf.cast(Erv.sample(N), dtype = tf.float32)#, Correction.sample(initial_distribution.shape)

#     def sample(self, E): #, Correction):

#         # M = tf.einsum("nj,kj->nk", tf.einsum("nk,kj->nj", tf.einsum("ni,i->ni", E, Correction), self.B), tf.einsum("ni,i->ni", E, Correction))
#         M = tf.einsum("nj,kj->nk", tf.einsum("nk,kj->nj", E, self.B), E)

#         A = tfp.distributions.Binomial(1, probs = tf.experimental.numpy.triu(M, k = 1)).sample()

#         return A + tf.transpose(tf.experimental.numpy.triu(A, k = 1))

#     def to_GRDPG(self, E): #, Correction):

#         eigenvalues, eigenvectors = tf.linalg.eig(self.B)

#         eigenvalues  = tf.math.real(eigenvalues )
#         eigenvectors = tf.math.real(eigenvectors)

#         q = sum(eigenvalues.numpy()<0)
#         p = eigenvalues.shape[0] - q

#         eigenvect_eigenval = tf.einsum("ij,j->ij", eigenvectors, tf.sqrt(tf.abs(eigenvalues)))

#         return p, q, tf.einsum("nj,jk->nk", E, eigenvect_eigenval)

