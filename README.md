# Graph-Embeddings-for-Functional-Brain-Connectivity-Networks
These scripts generate graph embeddings for brain connectivity networks.
Data is not included as it is protected (IRB).

Both generate_graph_embeddings.py and  generate_graph_embeddings_reduced_memory_usage.py take as input symmetrical adjacency matrices in csv format. Matrixes must be undirected, but weightings are okay. For reference, for my use I read in 352 sparse adjacency matrices. They were thresholded at a cost of 25%. Any element below that was zeroed out. (I.e. edges were removed). I kept the remaining edges with weights, but they could easily be binarized (i.e. set to 1).

The adjacency matrices are converted into NetworkX graphs, and graph embeddings are created using the GL2vec technique implemented in the karateclub library.

generate_graph_embeddings_reduced_memory_usage.py breaks up the generation of embeddings into 16 parts, in order to reduce overtaxing memory. Once each batch of embeddigns are gerated they are stored in a dataframe.

Script exports all embeddings to a csv, formatted for ease of use in ML projects. Ie each embedding is a vector of length 128. Each element of the vector is assigned to its own cell.

One thing I'm not positive about yet is whether whole-graph embeddings are comparable on an element-wise level. Ie, can each column of an embedding matrix be used as a feature vector, becaue they describe some analagous underlying component across graphs? I'm assuming this is the case, but should confirm.

Becuase my graphs correspond to different subjects and fmri conditions, those aspects of the code will not be applicable to others. 

References
https://link.springer.com/chapter/10.1007/978-3-030-36718-3_1

https://karateclub.readthedocs.io/en/latest/index.html
