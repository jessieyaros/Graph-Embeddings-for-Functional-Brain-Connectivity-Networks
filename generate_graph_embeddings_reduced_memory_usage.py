# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:01:25 2021

@author: jyaros
"""

from datetime import datetime
STARTTIME = datetime.now()
import os
import numpy as np
import pandas as pd
import networkx as nx
import karateclub as kc 
import warnings

print ("\n***Begin***\n")
workingDir = os.getcwd()


#INPUT_FILE_DIR = os.path.abspath(os.path.join(workingDir, os.pardir, 
#                                       'e_adjacency_matrices_reducedCosts'))

#testing with fewer files
INPUT_FILE_DIR = os.path.abspath(os.path.join(workingDir, os.pardir, 
                                        'test'))

INPUT_FILES = os.listdir(INPUT_FILE_DIR)


# This function creates a df that organizes file locations for the matrices 
#   alongside the subject and conditions for each matrix
def define_matrix_conditions(input_file_dir = INPUT_FILE_DIR, input_files = INPUT_FILES):
    #list files in directory
    input_files = os.listdir(input_file_dir)  
    #only import the lowest sparstiy files, which are still sparse matrices at 
    #a cost of 25%
    matrix_files = [file for file in input_files if file.endswith('0.25_cost.csv')]
    
    #create list of files as well as subject and condition information to 
    #create df with
    matrix_details = []
    
    #for each file create dictionary of file descriptors and append to 
    #matrix list
    for file in matrix_files:
        d = {}
        d['condition'] = file.split('_')[2]
        d['subject'] = file.split('_')[1]
        d['file_name'] = file
        matrix_details.append(d)  
    
    #create df with row describing each matrix file        
    df_matrix_details = pd.DataFrame(matrix_details)
    #organize table by condition, and then subject for viewing purposes
    df_matrix_details = df_matrix_details.sort_values(["condition", "subject"],
                                                      ignore_index = True)
    return df_matrix_details

# Call function to create table pointing to file locations for graphs
df =  define_matrix_conditions()

# This function creates a dictionary of networkx graph objects nested under  
# their corresponding subjects and conditions as dictionary keys
def create_graphs(df, input_file_dir = INPUT_FILE_DIR):

    #create dictionary to store embeddings alognside subject & condition variables
    d = {}
    #step throug eahc condition
    for cond in df['condition'].unique():
        #create dictionary for condition
        d[cond] = {}
        df_subset = df[df['condition'] == cond]
        #print(cond)
        # For each subject, load in their adjacency matrix for the current 
        # condition. Create a graph 
        for subj in df['subject'].unique():
            matrix_filename = df_subset['file_name'].loc\
                [df_subset['subject'] == subj].values[0]
            #load in matrix from csv    
            adjacency_matrix =  np.genfromtxt((os.path.join\
                            (input_file_dir,matrix_filename)), delimiter=',')
            
            #create undierected graph from the symmetrical adjacency matrix
            G = nx.from_numpy_matrix(adjacency_matrix)
            #print ("Is Graph Directed?")
            #print(nx.is_directed(G))
            
            # Assign graph to dictionary, under corresonding condition and 
            # subject keys.
            d[cond][subj] = G     

    return d

#Call function to create dictionary of graphs for each condition and subject
d = create_graphs(df)   

# This function takes in the dictionary of graph objects, converts them to a 
#   list and fits a model to learn graph embeddings. Outputs embeddings 
#   as a matrix
def generate_graph_embeddings(d, startTime = STARTTIME):
    
    # Create df from dictioabary of graphs. Columns are conditions and indices
    #    are subjects
    graph_df = pd.DataFrame.from_dict(d)
    
    # Create structure for df that will be exported to csv, with labeled 
    #   graph embeddings
    
    output_df = pd.DataFrame(graph_df.copy().stack())
    # Reset subject and condition indices to be values in columns, and rename
    output_df.reset_index(inplace = True)
    output_df.rename(columns = {'level_0':'subject', 'level_1':'condition'}, 
                                                            inplace = True)
    # Drop column containing graph objects. No longer needed now that we have
    #   the embeddings
    output_df.drop(0, axis=1, inplace = True)
 
    #Since we are creating embeddings by condition, structure table to be 
    #   sorted and organized by condition, then subject
    output_df.sort_values(["condition", "subject"],ignore_index = True,
                                                          inplace = True)
    cols = ['condition','subject']
    output_df = output_df[cols]
    
    for cond in output_df['condition'].unique():
        print(cond)
        
        # Take graphs for one condition at a time, otherwise script requires 
        #   too much memory         
        df_subset = graph_df.loc[:,graph_df.columns == cond]
        
        # Create list of graph objects from the column of graph objects
        graph_list = df_subset.values.flatten().tolist()
        
        # karateclub pagage uses soem deprectated phrases that will be removed 
        #   in python 4. This corresponds to a parameter called 'iter' that is 
        #   being renamed "epochs". For now, ignore to clean up printe outputs
        warnings.filterwarnings("ignore", category=UserWarning)

        
        # Inititate GL2vec (Graph Embedding Enriched by Line Graphs with Edge Features)
        model = kc.GL2Vec()
    
        # Learn the emebedding for each graph
        model.fit(graph_list)
    
        # Return embedding vectors in matrix with order corresponding to input list
        X = model.get_embedding()
        
        # Add embeddings (vectors) as a row of features corresponding to the 
        #   correct subject and condition labels
        
        #convert embeddings matrix to df
        df_embeddings =  pd.DataFrame(X)
        
        #subset output_df for condition/subject rows corresponding to embeddings
        output_subset = output_df.loc[output_df['condition'] == cond]
        
        # Reset df_embeddings indices to match the output_subset indices. This
        #   is crucial in order to re-associate (ie join) embeddings on the 
        #   correct subject and condition columns
        df_embeddings.index = output_subset.index
        print('index is', df_embeddings.index)
        
        #append embeddings to the proper rows in the output_df (on index)
        output_df = output_df.join(df_embeddings)
        
        print("Runtime:", datetime.now() - startTime)
        
    return output_df      
output_df = generate_graph_embeddings(d)       


#Call function to construct df of embeddings and their labels
#table_embeddings = assign_embeddings_to_conditions(embeddings, graph_series)
#Export to csv
#table_embeddings.to_csv("graph_embeddings.csv")

print("\nTotal Runtime:", datetime.now() - STARTTIME)