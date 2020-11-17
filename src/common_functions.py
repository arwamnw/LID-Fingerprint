#! /afs/cad/linux/anaconda-2.1.0/anaconda/bin/python


# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import sys
import math

__author__ = "arwawali"
__date__ = "$Mar 12, 2015 12:57:41 PM$"

# find unique labels of classes and clusters
def uniques_lables(labels):
    unique_lables = []
    count = 0
    while count != len(labels):
        if(labels[count] not in unique_lables):
            unique_lables.append(labels[count])
        count = count + 1
    
    return unique_lables


# function to count the pointts in each cluster and classes
def unique_clusters_classes(unique_true,unique_pred, true_lables, predec_labels):
    true_length=len(unique_true)
    pred_length=len(unique_pred)
    
    classes_labels_count=[]
    clusters_labels_count=[]
    for j in range (0,true_length,1 ):
        classes_labels_count.append(0)
    for k in range (0,pred_length,1 ):
         clusters_labels_count.append(0)
         
    for j in range (0,len(true_lables),1):
        classes_labels_count[unique_true.index(true_lables[j])]=classes_labels_count[unique_true.index(true_lables[j])]+1
    
    
    for k in range (0,len(predec_labels),1 ):
        clusters_labels_count[unique_pred.index(predec_labels[k])]=clusters_labels_count[unique_pred.index(predec_labels[k])]+1
    
    #print unique_true
    #print classes_labels_count
    ##print
    
    #print unique_pred
    #print clusters_labels_count
    return (classes_labels_count,clusters_labels_count )


def calculate_joint_labels(true_lables, predec_labels,unique_true, unique_pred ):
    
    # calculate I(Cluster,Classes)
    true_length=len(unique_true)
    #print true_length
    
    pred_length=len(unique_pred)
    #print pred_length
    joint_labels=[]   # matrix of length  pred_length*true_length
    for i in range (0,pred_length,1 ):
        temp=[]
        for j in range (0, true_length,1):
            
            temp.append(0.0)
            
        joint_labels.append(temp)
    
    
    for k in range (0,pred_length,1 ):
        for j in range (0,true_length,1 ):
            for i in range (0,len(true_lables),1 ):
                #print str(predec_labels[i]) + "  " + str(true_lables[i])
                
                if( predec_labels[i]==unique_pred[k] and true_lables[i]==unique_true[j]):
                    #print "I am here"
                    joint_labels[k][j]=float(joint_labels[k][j])+1.0
                    #print joint_labels[k][j]
    #print  joint_labels          
    return joint_labels


def readdataFile(datafile):
    f=open(datafile)
    data=[]
    i=0


    ############
    ### reading data
    ##############

    # read the number of rows and columns
    l=f.readline()
    a=l.split()
    rows=int(a[0])
    cols=int(a[1])

    # define the distance matrix
    #matrix=[rows][rows]
    l=f.readline()
    while(l !=''):
        a=l.split()
        l2=[]
        for j in range(0, len(a), 1):
            l2.append(float(a[j]))
        data.append(l2)
        l=f.readline()


    f.close()
    return data

def readlabelsfile(labelsfile):
    # Read labels file

    
    f=open(labelsfile)
    trainlabels=[]

    l=f.readline()
    l=f.readline()
    while(l!=''):
	trainlabels.append(l.rstrip('\n'))
	l=f.readline()
        
    return (trainlabels)


def calculate_distance_matrices(new_eval_matrix,old_eval_matrix):
    matrix_distances=0.0
    
    for i in range(0,len(new_eval_matrix),1):
        for j in range (0,len(new_eval_matrix[0]),1 ):
            matrix_distances= matrix_distances+(new_eval_matrix[i][j]-old_eval_matrix[i][j])**2
            
    
    matrix_distances=math.sqrt(matrix_distances)
    return matrix_distances

def get_rows_cols(datafile):
    f=open(datafile)
    data=[]
    i=0


    ############
    ### reading data
    ##############

    

    # define the distance matrix
    #matrix=[rows][rows]
    l=f.readline()
    while(l !=''):
        a=l.split()
        l2=[]
        for j in range(0, len(a), 1):
            l2.append(float(a[j]))
        data.append(l2)
        l=f.readline()

    rows=len(data)
    cols=len(data[0])
    f.close()
    return (rows,cols)

if __name__ == "__main__":
    datafile=sys.argv[1]
    
    
    (rows,cols)=get_rows_cols(datafile)
    
    print str(rows) +"  " + str(cols)
    '''
    old_values=sys.argv[1]
    new_values=sys.argv[2]
    
    old_matrix=[]
    new_matrix=[]
    
    f=open(old_values)
    l=f.readline()
    l=f.readline()
    while(l !=''):
        a=l.split()
        l2=[]
        for j in range(2, 7, 1):
            l2.append(float(a[j]))
        old_matrix.append(l2)
        l=f.readline()
        
    f.close()
    
    f=open(new_values)
    l=f.readline()
    l=f.readline()
    while(l !=''):
        a=l.split()
        l2=[]
        for j in range(2, 7, 1):
            l2.append(float(a[j]))
        new_matrix.append(l2)
        l=f.readline()
        
    f.close()
    
    print calculate_distance_matrices(old_matrix,new_matrix)'''
    
    