# GRDPG

## Methodology Week 2, initial project overview by Patrick Rubin-Delanchy

Large dynamic networks occur in many areas of cyber-security, and analysts seek to understand network topology and connectivity behaviour for various purposes (anomaly detection, network auditing, etc). One of the more popular applied approaches to network analysis is graph embedding, which refers to the task of representing the nodes as points in space, which can then be used for e.g. exploratory data analysis or as feature vectors in downstream machine-learning tasks. 

The most popular graph embedding algorithms, such as node2vec or spectral embedding, were not originally based on explicit statistical models. However, statistical models have helped understand and improve them. For example, by considering the stochastic block model, it becomes clear that negative as well as positive eigenvalues should be used for spectral embedding, contrary to many earlier recommendations. Similarly, asymptotic analysis of spectral embedding under the random dot product graph (RDPG) indicates that Gaussian clustering, rather K-means, should be used to find network communities. Finally, most relevant to the present proposal, recent work has shown that optimising the RDPG likelihood explicitly, rather than implicitly through spectral embedding, has concrete statistical advantages including reduced asymptotic variance, and the authors later also proposed a Bayesian solution. 

The RDPG lacks generality and in particular is inadequate for several cyber-security networks because it essentially assumes homophilic connectivity (a friend of a friend is a friend) which is seemingly rare in computer networks. node2vec seems to make similar, homophilic assumptions, although the model it is implicitly fitting is less obvious. The generalised random dot product graph and the graph root distribution, its infinite-dimensional counterpart, address this problem, but no Bayesian solution exists. From a few preliminary studies, it seems that the “shape of Bayesian uncertainty” about the latent positions in heterophilic, sparse graphs could be interesting. 

The objectives of this work are two-fold: 
- develop a Bayesian approach to fitting the generalised random dot product graph or even the graph root distribution, ideally scalable to large, sparse graphs; 
- theoretically investigate the geometry of Bayesian uncertainty about the latent positions under heterophilic, sparse graphs, and determine the extent to which this is captured by the Bayesian approach above.

Several cyber-security datasets are available for this project.
