import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class GRDPG:

    def __init__(self, p, q, N):

        self.p = p
        self.q = q
        self.I      = tf.concat((tf.ones(p), tf.ones(q)), axis = 0)
        self.N      = N

    def sample_Beta(self, beta_a, beta_b):

        X      = tfp.distributions.Beta(beta_a, beta_b).sample((self.N, self.p+self.q))
        Xminus = X*tf.reshape(self.I, (1, self.p+self.q))

        M = tf.einsum("ik,jk->ij", Xminus, X)

        M = tf.reduce_min(tf.concat((tf.reshape(M, (self.N, self.N, 1)), tf.ones( (self.N, self.N, 1))), axis =2), axis =2)
        M = tf.reduce_max(tf.concat((tf.reshape(M, (self.N, self.N, 1)), tf.zeros((self.N, self.N, 1))), axis =2), axis =2)

        A = tfp.distributions.Binomial(1, probs = np.triu(M, k = 1)).sample()

        return A + tf.transpose(tf.experimental.numpy.triu(A, k = 1))


class BlockModel:
    
    def __init__(self, B):

        self.B = B

    def pop_sample(self, initial_distribution, N, beta_a, beta_b):

        Correction = tfp.distributions.Beta(beta_a, beta_b)
        Erv        = tfp.distributions.OneHotCategorical(probs = initial_distribution)

        return Erv.sample(N), Correction.sample(initial_distribution.shape)

    def sample(self, E, Correction):

        M = np.einsum("nj,kj->nk", np.einsum("nk,kj->nj", np.einsum("ni,i->ni", E, Correction), self.B), np.einsum("ni,i->ni", E, Correction))

        A = tfp.distributions.Binomial(1, probs = np.triu(M, k = 1)).sample()

        return A + tf.transpose(tf.experimental.numpy.triu(A, k = 1))

