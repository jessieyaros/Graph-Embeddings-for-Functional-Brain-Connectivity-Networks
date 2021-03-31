# Graph-Embeddings-for-Functional-Brain-Connectivity-Networks
This script generates graph embeddings for brain connectivity networks.
Data not included as it is protected.

Script takes as input symmetrical adjacency matrices. They must be undirected. Weigthed is okay.

The adjacency matrices are converted into NetworkX graphs, and graph embeddings are created using the GL2vec technique implemented in the karateclub library.

Becuase my graphs correspond to different subjects and fmri conditions, those aspects of the code will not be applicable to others. 

References
https://link.springer.com/chapter/10.1007/978-3-030-36718-3_1
https://karateclub.readthedocs.io/en/latest/index.html
