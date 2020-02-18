
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:45:29 2019

@author: txuslopez
"""

import os
os.system("%matplotlib inline")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data import DataStream
from sklearn.svm import SVC
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow import lazy
from skgarden import MondrianForestClassifier,MondrianTreeClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB
from texttable import Texttable
from CA_VonNeumann_estimator import CA_VonNeumann_Classifier
from CA_Moore_estimator import CA_Moore_Classifier
from CA_Gradient_estimator import CA_Gradient_Classifier
from IPython.display import HTML
from sklearn.neighbors import KNeighborsClassifier
from numpy import ndarray
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.base import BaseEstimator, ClassifierMixin
from skmultiflow.bayes import NaiveBayes
from sklearn.base import clone
from collections import deque

import matplotlib.animation as animat; animat.writers.list()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import scipy.io as sio
    
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 22})

#==============================================================================
# FUNCTIONS
#==============================================================================

def empties(b):
    invB = np.flip(b, axis=0)
    empty = []
    for b in invB:
        build = deepcopy(empty)
        empty = []
        for i in range(0,b):
            empty.append(build)

    return np.array(empty).tolist()

def hyperparametertuning_CA(ca,scoring,cv,X,y):

    grid1 = {'dimensions': [[3,3]],
               'cells': [empties([3,3])],
               'bins': [[]]
               }

    grid2 = {'dimensions': [[5,5]],
               'cells': [empties([5,5])],
               'bins': [[]]
               }

    grid3 = {'dimensions': [[7,7]],
               'cells': [empties([7,7])],
               'bins': [[]]
               }

    grid4 = {'dimensions': [[10,10]],
               'cells': [empties([10,10])],
               'bins': [[]]
               }
    
    lst_dicts_params=[grid1,grid2,grid3,grid4]
                
    if ca.__class__.__name__=='CA_VonNeumann_Classifier':

        grid_cv = GridSearchCV(ca, lst_dicts_params, cv=cv,scoring=scoring)
        grid_cv.fit(X.as_matrix(), y.as_matrix().ravel())     
             
    elif ca.__class__.__name__=='CA_Moore_Classifier':
    
        grid_cv = GridSearchCV(ca, lst_dicts_params, cv=cv,scoring=scoring)
        grid_cv.fit(X.as_matrix(), y.as_matrix().ravel())     

    elif ca.__class__.__name__=='CA_Gradient_Classifier':
    
        grid_cv = GridSearchCV(CA_Gradient_Classifier(), lst_dicts_params, cv=cv,scoring=scoring)
        grid_cv.fit(X.as_matrix(), y.as_matrix().ravel())     
            
    return grid_cv.best_estimator_            

def hyperparametertuning_classifiers(classifiers,scoring,cv,X_init,y_init,max_iter,knn_max_window_size):
    
    knn_optimized_params=[]
    
    for cl in range(len(classifiers)):
    
        cl_name=classifiers[cl].__class__.__name__                                                
        print (cl_name)

        if cl_name=='PassiveAggressiveClassifier':

            PAC_grid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                        'max_iter': [max_iter,100,200,500]            
            }                            
            
            grid_cv_PAC = RandomizedSearchCV(classifiers[cl], PAC_grid, cv=cv,scoring=scoring)
            grid_cv_PAC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_PAC.best_estimator_                      
    
        elif cl_name=='SGDClassifier':

            SGDC_grid = {
                'alpha': 10.0 ** -np.arange(1, 7),
                'loss': ['perceptron','hinge', 'log', 'modified_huber', 'squared_hinge'],
                'learning_rate':['constant','optimal','invscaling','adaptive'],
                'eta0':[0.1,0.5,1.0],
                'penalty': [None, 'l2', 'l1', 'elasticnet'],
                'max_iter': [max_iter,100,200,500]
            }              
            
            grid_cv_SGDC = RandomizedSearchCV(classifiers[cl], SGDC_grid, cv=cv,scoring=scoring)
            grid_cv_SGDC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_SGDC.best_estimator_                      

        elif cl_name=='MLPClassifier':

            MLPC_grid = {'hidden_layer_sizes': [(50, ), (100,), (50,50), (100,100)],
                          'activation': ['identity', 'logistic', 'tanh', 'relu'],
                          'solver': ['sgd','adam'],
                          'learning_rate': ['constant','invscaling','adaptive'],
                          'learning_rate_init': [0.0005,0.001,0.005],
                          'alpha': 10.0 ** -np.arange(1, 10),
                          'batch_size': [1,'auto'],
                          'max_iter': [1,100,200,500]                          
                          }        
            
            grid_cv_MLPC = RandomizedSearchCV(classifiers[cl], MLPC_grid, cv=cv,scoring=scoring)
            grid_cv_MLPC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_MLPC.best_estimator_                        
    
        elif cl_name=='MondrianTreeClassifier':

            MTC_grid = {'max_depth': [None,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                          'min_samples_split': [2, 5, 10]
                          }        
            
            grid_cv_MTC = RandomizedSearchCV(classifiers[cl], MTC_grid, cv=cv,scoring=scoring)
            grid_cv_MTC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_MTC.best_estimator_                        

        elif cl_name=='MondrianForestClassifier':

            MFR_grid = {'n_estimators': [5,10,25,50,100],
                          'max_depth': [None,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                          'min_samples_split': [2, 5, 10]
                          }        
            
            grid_cv_MFC = RandomizedSearchCV(classifiers[cl], MFR_grid, cv=cv,scoring=scoring)
            grid_cv_MFC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_MFC.best_estimator_

        elif cl_name=='KNN':

            KNN_grid = {'n_neighbors': [5,10,15,25,50],
#                          'max_window_size': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                          'leaf_size': [5, 10,20,30],
                          'algorithm':['auto'],
                          'weights':['uniform','distance']
                          }        

            grid_cv_KNN = RandomizedSearchCV(KNeighborsClassifier(), KNN_grid, cv=cv,scoring=scoring)
            grid_cv_KNN.fit(X_init,y_init)                
#            print('grid_cv_KNN.best_params_: ',grid_cv_KNN.best_params_)
            n_neighbors=grid_cv_KNN.best_params_['n_neighbors']
            leaf_size=grid_cv_KNN.best_params_['leaf_size']

            knn_optimized_params=[n_neighbors,knn_max_window_size,leaf_size]
            
            classifiers[cl]=lazy.KNN(n_neighbors=n_neighbors, max_window_size=knn_max_window_size, leaf_size=leaf_size)

        elif cl_name=='VFDR':
            print (cl_name, ' No tuning yet! ')

#            VFDR_grid = {'ordered_rules': [True,False],
#                          'rule_prediction': ['first_hit','weighted_max','weighted_sum'],
#                          'max_rules': [5,10,20,30,50],
#                          'drift_detector': [None],
#                          'expand_criterion': ['info_gain','hellinger','foil_gain']
#                          }        
            
#            classifiers[cl]=VFDR()
            
        elif cl_name=='HoeffdingTree':
            print (cl_name,' No tuning yet! ')
            classifiers[cl]=HoeffdingTree()

        elif cl_name=='GaussianNB':
            classifiers[cl]=GaussianNB()

        elif cl_name=='GaussianNB':
            classifiers[cl]=GaussianNB()
                            
    return classifiers,knn_optimized_params

def get_classifiers_names(classifiers):
    
    classifiers_names=[]            
    for r in range(len(classifiers)):
        reg_name=classifiers[r].__class__.__name__
        
        if reg_name=='PassiveAggressiveClassifier':
            classifiers_names.append('PAC')                    
        elif reg_name=='SGDClassifier':
            classifiers_names.append('SGDC')                
        elif reg_name=='MLPClassifier':
            classifiers_names.append('MLPC')
        elif reg_name=='HoeffdingTree':
            classifiers_names.append('HTC')
        elif reg_name=='MondrianForestClassifier':
            classifiers_names.append('MFC')
        elif reg_name=='MondrianTreeClassifier':
            classifiers_names.append('MTC') 
        elif reg_name=='GaussianNB':
            classifiers_names.append('GNBC') 
        elif reg_name=='KNN':
            classifiers_names.append('KNN') 
            
    return classifiers_names

def get_paired_learners_names(paired_learners):
    
    paired_learners_names=[]            
    for r in range(len(paired_learners)):
        reg_name=paired_learners[r].stable_learner.__class__.__name__
        
        if reg_name=='PassiveAggressiveClassifier':
            paired_learners_names.append('PAC')                    
        elif reg_name=='SGDClassifier':
            paired_learners_names.append('SGDC')                
        elif reg_name=='MLPClassifier':
            paired_learners_names.append('MLPC')
        elif reg_name=='HoeffdingTree':
            paired_learners_names.append('HTC')
        elif reg_name=='MondrianForestClassifier':
            paired_learners_names.append('MFC')
        elif reg_name=='MondrianTreeClassifier':
            paired_learners_names.append('MTC') 
        elif reg_name=='GaussianNB':
            paired_learners_names.append('GNBC') 
        elif reg_name=='KNN':
            paired_learners_names.append('KNN') 
        else:
            paired_learners_names.append('Unnamed') 
        
    return paired_learners_names

def plot_CA_boundaries_stream(cellular_automatas,buchaquer_X,buchaquer_y,X_columns,y_columns,sample,mutaciones,ca_names):
            
    idxs=[[]]*len(cellular_automatas)

    for ca in range(len(cellular_automatas)):
    
        if paired_automatas[ca].__class__.__name__=='AutomataPairedLearners':
            cel_automat=deepcopy(cellular_automatas[ca].stable_learner)
        else:                        
            cel_automat=deepcopy(cellular_automatas[ca])
        
        dim=cel_automat.dimensions
    
        # Create image arrays
        img = deepcopy(empties(dim))
    
        # Set variables to model results
        cells=cel_automat.cells
    
        for j, c in enumerate(sample[0]):
            idxs.append(np.argmax(c <= cel_automat.bins[j]))            
        
        for i in range(0, len(cells)):
            for j in range(0, len(cells)):
    
                if cells[i][j]:
                    
                    if i==idxs[0] and j==idxs[1]:
                        if mutaciones[ca]:
                            img[i][j]=[255,51,255]
                        else:
                            img[i][j]=[0,0,0]
                    else:                
                        s = cells[i][j][0].species
                        rgb = (np.zeros(3)).tolist()
                        rgb[int(s)] = 255
                        img[i][j] = rgb
                else:
                    img[i][j] = [255,255,255]
                

    
        # Convert image arrays to appropriate data types
        img = np.array(img, dtype='uint8')

        # Show the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
        buch_pd_X=pd.DataFrame(buchaquer_X)
        buch_pd_X.columns=X_columns
        buch_pd_y=pd.DataFrame(buchaquer_y)
        buch_pd_y.columns=[y_columns]
        
        todo=pd.concat([buch_pd_X,buch_pd_y],axis=1)
        
        X1=todo[todo[y_columns]==0]
        X2=todo[todo[y_columns]==1]
    #    X3=todo[todo['class']==2]
        
        # Data Subplot
        ax1.plot(X1.iloc[:,0], X1.iloc[:,1], 'r.')
        ax1.plot(X2.iloc[:,0], X2.iloc[:,1], 'g.')
    #    ax1.plot(X3.iloc[:,0], X3.iloc[:,1], 'b.')
        ax1.title.set_text('Dataset')
    
        # Method 1 Subplot
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.title.set_text(ca_names[ca])
        ax2.imshow(img)
        
        plt.show()
  

def plot_CA_boundaries(cellular_automatas,buch_X,buch_y,X_columns,y_columns,ca_names):


    for ca in range(len(cellular_automatas)):
        
        if paired_automatas[ca].__class__.__name__=='AutomataPairedLearners':
            cel_automat=deepcopy(cellular_automatas[ca].stable_learner)
        else:                        
            cel_automat=deepcopy(cellular_automatas[ca])
        
        dim=cel_automat.dimensions
    
        # Create image arrays
        img = deepcopy(empties(dim))
    
        # Set variables to model results
        cells = cel_automat.cells
    
        for i in range(0, len(cells)):
            for j in range(0, len(cells)):
                                
                if cells[i][j]:                                              
                    s = cells[i][j][0].species
                    rgb = (np.zeros(3)).tolist()
                    rgb[int(s)] = 255
                    img[i][j] = rgb
                else:
                    img[i][j] = [255,255,255]
    
        # Convert image arrays to appropriate data types
        img = np.array(img, dtype='uint8')
    
        # Show the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        buch_pd_X=pd.DataFrame(buch_X)
        buch_pd_X.columns=X_columns
        buch_pd_y=pd.DataFrame(buch_y)
        buch_pd_y.columns=[y_columns]
        
        todo=pd.concat([buch_pd_X,buch_pd_y],axis=1)
        
        X1=todo[todo[y_columns]==0]
        X2=todo[todo[y_columns]==1]
    #    X3=todo[todo['class']==2]
        
        # Data Subplot
        ax1.plot(X1.iloc[:,0], X1.iloc[:,1], 'r.')
        ax1.plot(X2.iloc[:,0], X2.iloc[:,1], 'g.')
    #    ax1.plot(X3.iloc[:,0], X3.iloc[:,1], 'b.')
        ax1.title.set_text('Dataset')
        
        # Method 1 Subplot
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.title.set_text(ca_names[ca])
        ax2.imshow(img)

        plt.show() 

#def prediction_score(predicciones,c,y_test_then_train,s):
#
#    return accuracy_score(predicciones[c], y_test_then_train.iloc[:s+1,:])

def prequential_acc(predicted_class,Y_tst,PREQ_ACCS,t,f):

    #Prequential accuracy
    pred=0
    if predicted_class==Y_tst:    
        pred=1
    else:
        pred=0

    if s==0:
        preqAcc=1
    else:        
        preqAcc=(PREQ_ACCS[-1]+float((pred-PREQ_ACCS[-1])/(t-f+1)))

    return preqAcc

def cellular_automatas_naming(cellular_automatas,columnas):
    
    ca_names=[str()]*len(cellular_automatas)
    for ca in range(len(cellular_automatas)):
        if cellular_automatas[ca].__class__.__name__=='CA_VonNeumann_Classifier':            
            ca_names[ca]='LUNAR'
            
    
    return ca_names

def streamers_Texttable(classifiers,class_names,pac_res_mean,pac_res_std,sgdc_res_mean,sgdc_res_std,htc_res_mean,htc_res_std,knn_res_mean,knn_res_std):
    
    t_streamers = Texttable()
    
    for c in range(len(classifiers)):

        if class_names[c]=='PAC':
            m=np.round(np.mean(pac_res_mean),3)
            std=np.round(np.mean(pac_res_std),3)
                             
        elif class_names[c]=='SGDC':
            m=np.round(np.mean(sgdc_res_mean),3)
            std=np.round(np.mean(sgdc_res_std),3)
            
        elif class_names[c]=='HTC':
            m=np.round(np.mean(htc_res_mean),3)
            std=np.round(np.mean(htc_res_std),3)
            
        elif class_names[c]=='KNN':
            m=np.round(np.mean(knn_res_mean),3)
            std=np.round(np.mean(knn_res_std),3)

        t_streamers.add_rows([['STREAMERS', 'Accuracy'],[class_names[c],str(m)+str('+-')+str(std)]])

    print (t_streamers.draw())  
    
def automatas_Texttable(cellular_automatas,paired_automatas_names,automatas_res_mean,automatas_res_std,X_init):    
    
    t_automatas = Texttable()
    
    automatas_mean=[[]]*len(cellular_automatas)
    automatas_std=[[]]*len(cellular_automatas)
    for h in range(len(cellular_automatas)):
        automatas_mean[h]=np.round(np.mean(automatas_res_mean[h][X_init.shape[0]:]),3)
        automatas_std[h]=np.round(np.mean(automatas_res_std[h][X_init.shape[0]:]),3)

    for h in range(len(cellular_automatas)):
        t_automatas.add_rows([['AUTOMATAS', 'Accuracy'],[str(paired_automatas_names[h]),str(automatas_mean[h])+str('+-')+str(automatas_std[h])]])
    
    print (t_automatas.draw())    

def plot_automatas_results(size_X,size_Y,colors,font_size,title,XT,cellular_automatas,automatas_res_mean,automatas_res_std,ca_names,X_init,rolling_w):
    
    fig=plt.figure(figsize=(size_X,size_Y))
#    plt.title(title,size=font_size)
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('Prequential accuracy',size=font_size)
    plt.ylim(0.0,1.0)
    plt.xlim(0,XT.shape[0])
                
    for p in range(len(cellular_automatas)):
        df_m=pd.DataFrame(automatas_res_mean[p])
        
        plt.plot(df_m.rolling(window=rolling_w).mean(),color=colors[p],label=ca_names[p],linestyle='--')
        
    plt.axvspan(0, X_init.shape[0], alpha=0.5, color='green')    

    plt.legend(prop={'size': font_size},loc='lower center',fancybox=True, shadow=True,ncol=5,bbox_to_anchor=(0.5, 1.0))
    plt.show()  
    
def plot_streamers_results(size_X,size_Y,colors,font_size,title,XT,pac_res_mean,pac_res_std,sgdc_res_mean,sgdc_res_std,htc_res_mean,htc_res_std,gnbc_res_mean,gnbc_res_std,knn_res_mean,knn_res_std,X_init,rolling_w):
    
    fig=plt.figure(figsize=(size_X,size_Y))
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('Prequential accuracy',size=font_size)
    plt.ylim(0.0,1.0)
    plt.xlim(0,XT.shape[0])  
    
    df_pac_res_mean=pd.DataFrame(pac_res_mean)
    plt.plot(df_pac_res_mean.rolling(window=rolling_w).mean(),color=colors[3],label='PAC')

    df_sgdc_res_mean=pd.DataFrame(sgdc_res_mean)
    plt.plot(df_sgdc_res_mean.rolling(window=rolling_w).mean(),color=colors[4],label='SGDC')

    df_htc_res_mean=pd.DataFrame(htc_res_mean)
    plt.plot(df_htc_res_mean.rolling(window=rolling_w).mean(),color=colors[7],label='HTC')

    df_gnbc_res_mean=pd.DataFrame(gnbc_res_mean)
    plt.plot(df_gnbc_res_mean.rolling(window=rolling_w).mean(),color=colors[8],label='GNBC')

    df_knn_res_mean=pd.DataFrame(knn_res_mean)
    plt.plot(df_knn_res_mean.rolling(window=rolling_w).mean(),color=colors[9],label='KNN')
    
    plt.axvspan(0, X_init.shape[0], alpha=0.5, color='green')    

    plt.legend(prop={'size': font_size},loc='upper right',fancybox=True, shadow=True,ncol=4,bbox_to_anchor=(0.80, 1.18))   
    
    plt.show()
    
def plot_results(size_X,size_Y,colors,font_size,title,caso,num_cel_automatas,automatas_res_mean,automatas_res_std,ca_names,pac_res_mean,pac_res_std,sgdc_res_mean,sgdc_res_std,htc_res_mean,htc_res_std,knn_res_mean,knn_res_std,X_init,rolling_w,data_type):
    
    fig=plt.figure(figsize=(size_X,size_Y))
#    plt.title(title,size=font_size)
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('Prequential accuracy',size=font_size)
    plt.ylim(0.0,1.0)
    plt.xlim(0,caso.shape[0])  
    
    df_pac_res_mean=pd.DataFrame(pac_res_mean)
    plt.plot(df_pac_res_mean.rolling(window=rolling_w).mean(),color=colors[3],label='PAC')
    plt.fill_between(range(len(pac_res_mean)), pac_res_mean-pac_res_std, pac_res_mean+pac_res_std,facecolor=colors[3], alpha=0.1)    

    df_sgdc_res_mean=pd.DataFrame(sgdc_res_mean)
    plt.plot(df_sgdc_res_mean.rolling(window=rolling_w).mean(),color=colors[4],label='SGDC')
    plt.fill_between(range(len(sgdc_res_mean)), sgdc_res_mean-sgdc_res_std, sgdc_res_mean+sgdc_res_std,facecolor=colors[4], alpha=0.1)    

    df_htc_res_mean=pd.DataFrame(htc_res_mean)
    plt.plot(df_htc_res_mean.rolling(window=rolling_w).mean(),color=colors[7],label='HTC')
    plt.fill_between(range(len(htc_res_mean)), htc_res_mean-htc_res_std, htc_res_mean+htc_res_std,facecolor=colors[7], alpha=0.1)    

    df_knn_res_mean=pd.DataFrame(knn_res_mean)
    plt.plot(df_knn_res_mean.rolling(window=rolling_w).mean(),color=colors[9],label='KNN')
    plt.fill_between(range(len(knn_res_mean)), knn_res_mean-knn_res_std, knn_res_mean+knn_res_std,facecolor=colors[9], alpha=0.1)    
    
    for p in range(num_cel_automatas):
        df_m=pd.DataFrame(automatas_res_mean[p])
        
        plt.plot(df_m.rolling(window=rolling_w).mean(),color=colors[p],label=ca_names[p],linestyle='--')
        plt.fill_between(range(len(automatas_res_mean[p])), automatas_res_mean[p]-automatas_res_std[p], automatas_res_mean[p]+automatas_res_std[p],facecolor=colors[p], alpha=0.1)    
    
    plt.axvspan(0, X_init.shape[0], alpha=0.5, color='#FEE88A')    

    plt.legend(prop={'size': font_size},loc='upper left',fancybox=True, shadow=True,ncol=1)#bbox_to_anchor=(0.80, 1.18)   
    
    plt.savefig('results_real_'+str(data_type)+'.svg')
    
    plt.show()    
          
def save_data(output_pickle,SC_pac,SC_sgdc,SC_htc,SC_gnbc,SC_knn,SC_ca_automatas,data_type):
    
    output = open(output_pickle+'SC_pac_'+str(data_type)+'.pkl', 'wb')
    pickle.dump(SC_pac, output)
    output.close()
    sio.savemat(output_pickle+'SC_pac_'+str(data_type)+'.mat', {'SC_pac_'+str(data_type):SC_pac})

    output = open(output_pickle+'SC_sgdc_'+str(data_type)+'.pkl', 'wb')
    pickle.dump(SC_sgdc, output)
    output.close()
    sio.savemat(output_pickle+'SC_sgdc_'+str(data_type)+'.mat', {'SC_sgdc_'+str(data_type):SC_sgdc})

    output = open(output_pickle+'SC_htc_'+str(data_type)+'.pkl', 'wb')
    pickle.dump(SC_htc, output)
    output.close()
    sio.savemat(output_pickle+'SC_htc_'+str(data_type)+'.mat', {'SC_htc_'+str(data_type):SC_htc})

    output = open(output_pickle+'SC_gnbc_'+str(data_type)+'.pkl', 'wb')
    pickle.dump(SC_gnbc, output)
    output.close()
    sio.savemat(output_pickle+'SC_gnbc_'+str(data_type)+'.mat', {'SC_gnbc_'+str(data_type):SC_gnbc})

    output = open(output_pickle+'SC_knn_'+str(data_type)+'.pkl', 'wb')
    pickle.dump(SC_knn, output)
    output.close()
    sio.savemat(output_pickle+'SC_knn_'+str(data_type)+'.mat', {'SC_knn_'+str(data_type):SC_knn})

    output = open(output_pickle+'SC_ca_automatas_'+str(data_type)+'.pkl', 'wb')
    pickle.dump(SC_ca_automatas, output)
    output.close()
    sio.savemat(output_pickle+'SC_ca_automatas_'+str(data_type)+'.mat', {'SC_ca_automatas_'+str(data_type):SC_ca_automatas})
    
def load_data(output_pickle,data_type):
    
    fil = open(output_pickle+'SC_pac_'+str(data_type)+'.pkl','rb')
    SC_pac = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'SC_sgdc_'+str(data_type)+'.pkl','rb')
    SC_sgdc = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'SC_htc_'+str(data_type)+'.pkl','rb')
    SC_htc = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'SC_gnbc_'+str(data_type)+'.pkl','rb')
    SC_gnbc = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'SC_knn_'+str(data_type)+'.pkl','rb')
    SC_knn = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'SC_ca_automatas_'+str(data_type)+'.pkl','rb')
    SC_ca_automatas = pickle.load(fil)
    fil.close()

    return SC_pac,SC_sgdc,SC_htc,SC_gnbc,SC_knn,SC_ca_automatas
    
#==============================================================================
# CLASSES
#==============================================================================
        
class PairedLearners (BaseEstimator, ClassifierMixin):  
    
    def __init__(self, stable_learner=NaiveBayes(), reactive_learner=NaiveBayes(), window_size=50, threshold=0.1):
        self.stable_learner = stable_learner
        self.reactive_learner = reactive_learner
        self.window_size=window_size
        self.threshold=threshold
        self.C=deque(maxlen=window_size)
        self.theta=self.window_size*self.threshold
        self.changeDetected=0
        self.n_errors=0

    def fit(self, X, y):

        self.stable_learner.fit(X,y)
        self.reactive_learner.fit(X,y)

        return self

    def partial_fit(self, X, y, Xs, ys,clss,knn_optimized_params):
        
        #Se testea
        pred_stable_learner=self.stable_learner.predict(X)
        pred_reactive_learner=self.reactive_learner.predict(X)
        
        #Se mira si difieren en sus predicciones
        if pred_stable_learner!=y and pred_reactive_learner==y:            
            self.C.appendleft(1)
        else:
            self.C.appendleft(0)
                                                        
        #Se entrenan el stable learner
        self.stable_learner.partial_fit(X,y)

        #Se entrenan el reactive learner con la ventana actual. Hay que reiniciarlo        
        if self.reactive_learner.__class__.__name__=='HoeffdingTree':#No es estimator de sklearn
            new_reactive=HoeffdingTree()
        elif self.reactive_learner.__class__.__name__=='KNN':#No es estimator de sklearn
            new_reactive=lazy.KNN(n_neighbors=knn_optimized_params[0], max_window_size=knn_optimized_params[1], leaf_size=knn_optimized_params[2])
        else:
            new_reactive=clone(self.reactive_learner)#Clone does a deep copy of the model in an estimator without actually copying attached data. It yields a new estimator with the same parameters that has not been fit on any data.
        
        #Ahora se entrena
        if self.reactive_learner.__class__.__name__=='KNN':    
            new_reactive.fit(Xs,ys,clss)
        else:
            for i in range(len(Xs)):
                sample=Xs[i]
                lab=ys[i]
                new_reactive.partial_fit([sample],[lab],clss)
            
        self.reactive_learner=new_reactive
            
        #Se mira si hay drift
        if self.theta<sum(self.C):
            self.changeDetected+=1
                 
        return self

    def predict(self, X):

        return self.stable_learner.predict(X)
    
    def driftAdaptation(self,Xs,ys,clss):
        
        self.C=deque(maxlen=window_size)
        self.changeDetected=0
        self.stable_learner=deepcopy(self.reactive_learner)#Aqui si queremos mantener el entrenamiento y los parametros
        
        #Se entrenan el reactive learner con la ventana actual. Hay que reiniciarlo        
        if self.reactive_learner.__class__.__name__=='HoeffdingTree':#No es estimator de sklearn
            new_reactive=HoeffdingTree()
            new_reactive.partial_fit(Xs.as_matrix(),ys.as_matrix().ravel(),clss)
        elif self.reactive_learner.__class__.__name__=='KNN':#No es estimator de sklearn
            new_reactive=lazy.KNN(n_neighbors=knn_optimized_params[0], max_window_size=knn_optimized_params[1], leaf_size=knn_optimized_params[2])
            new_reactive.fit(Xs.as_matrix(),ys.as_matrix().ravel(),clss)
        else:
            new_reactive=clone(self.reactive_learner)#Clone does a deep copy of the model in an estimator without actually copying attached data. It yields a new estimator with the same parameters that has not been fit on any data.
            for i in range(len(Xs)):
                sample=Xs.iloc[i]
                lab=ys.iloc[i]
                new_reactive.partial_fit([sample],[lab],clss)
                    
        self.reactive_learner=new_reactive        
        
        return self     

    def score(self, X,window_size, y):
        # counts number of values bigger than mean
        return print('Scoring should be implemented in case of need')
        
    def get_params(self, deep=True):
        return {'stable_learner': self.stable_learner, 'reactive_learner': self.reactive_learner, 'window_size': self.window_size, 'threshold': self.threshold, 'C': self.C, 'theta': self.theta}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self  
    
class AutomataPairedLearners (BaseEstimator, ClassifierMixin):  
    
    def __init__(self, stable_learner=CA_VonNeumann_Classifier(bins=[],bins_margin=0.1,dimensions=[5,5], cells=empties([5,5])), reactive_learner=CA_VonNeumann_Classifier(bins=[],bins_margin=0.1,dimensions=[5,5], cells=empties([5,5])), window_size=50, threshold=0.1):
        self.stable_learner = stable_learner
        self.reactive_learner = reactive_learner
        self.window_size=window_size
        self.threshold=threshold
        self.C=deque(maxlen=window_size)
        self.theta=self.window_size*self.threshold
        self.changeDetected=0
        self.n_errors=0

    def fit(self, X, y):

        self.stable_learner,stable_limits=self.stable_learner.fit(X,y)
        self.reactive_learner,reactive_limits=self.reactive_learner.fit(X,y)
        
        return self,stable_limits,reactive_limits

    def partial_fit(self, X, y, Xs, ys,clss,stable_limits,reactive_limits):
        
        #Se testea
        pred_stable_learner=self.stable_learner.predict(X)
        pred_reactive_learner=self.reactive_learner.predict(X)
        
        #Se mira si difieren en sus predicciones
        if pred_stable_learner!=y and pred_reactive_learner==y:
            self.C.appendleft(1)
        else:
            self.C.appendleft(0)
                                                        
        #Se entrenan el stable learner
        self.stable_learner,stable_limits,muta_stable,stable_indexes=self.stable_learner.partial_fit(X,y,None,stable_limits)
        new_reactive=clone(self.reactive_learner)#Clone does a deep copy of the model in an estimator without actually copying attached data. It yields a new estimator with the same parameters that has not been fit on any data.
        
        #Se actualizan los bins y limites del reactivo
        w_limits=[]
        w_bins=[]
        n=len(self.reactive_learner.dimensions)
        dims = np.array(self.reactive_learner.dimensions)

        for j in range(0,n):
            min_dat = np.min(Xs[:, j]) - self.reactive_learner.bins_margin*(np.max(Xs[:, j])-np.min(Xs[:, j]))
            max_dat = np.max(Xs[:, j]) + self.reactive_learner.bins_margin*(np.max(Xs[:, j])-np.min(Xs[:, j]))
            delta = (max_dat-min_dat)/dims[j]

            w_bins.append(np.arange(min_dat, max_dat, delta)+delta)            
            w_limits.append([np.min(Xs[:, j]),np.max(Xs[:, j])])
            
        reactive_limits=w_limits
        self.reactive_learner.bins=w_bins
        
        #Ahora se entrena el reactivo
        for i in range(len(Xs)):
            sample=Xs[i]
            lab=ys[i]
            new_reactive,reactive_limits,muta_reactive,reactive_indexes=new_reactive.partial_fit([sample],[lab],None,reactive_limits)
            
        self.reactive_learner=new_reactive
            
        #Se mira si hay drift
        if self.theta<sum(self.C):
            self.changeDetected+=1
                 
        return self,stable_limits,muta_stable,reactive_limits,muta_reactive

    def predict(self, X):

        return self.stable_learner.predict(X)
    
    def driftAdaptation(self,Xs,ys,clss,reactive_limits,reactive_mutations):
        
        self.C=deque(maxlen=window_size)
        self.changeDetected=0
        self.stable_learner=deepcopy(self.reactive_learner)#Aqui si queremos mantener el entrenamiento y los parametros
        stable_lims=reactive_limits
        stable_muts=reactive_mutations
        
        #Se entrenan el reactive learner con la ventana actual. Hay que reiniciarlo        
        new_reactive=clone(self.reactive_learner)#Clone does a deep copy of the model in an estimator without actually copying attached data. It yields a new estimator with the same parameters that has not been fit on any data.

        new_reactive,reactive_limits_automat=new_reactive.fit(Xs,ys)
                    
        self.reactive_learner=new_reactive  
                
        return self,stable_lims,stable_muts,reactive_limits_automat     

    def score(self, X,window_size, y):
        # counts number of values bigger than mean
        return print('Scoring should be implemented in case of need')
        
    def get_params(self, deep=True):
        return {'stable_learner': self.stable_learner, 'reactive_learner': self.reactive_learner, 'window_size': self.window_size, 'threshold': self.threshold, 'C': self.C, 'theta': self.theta}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self  
    
#==============================================================================
# DATASETS
#==============================================================================
'''
#ELEC2
data_type='elec2'
path='yourpath//elecNormNew.csv'
caso = pd.read_csv(path, sep=',', header=0)
caso.drop('date', axis=1, inplace=True)#Se borra la columna date
caso.drop('nswprice', axis=1, inplace=True)#Se borra la columna nswprice
caso.drop('vicprice', axis=1, inplace=True)#Se borra la columna vicprice

caso.drop('day', axis=1, inplace=True)#Se borra la columna day
columns=caso.iloc[:,0:4].columns

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(caso)
caso = pd.DataFrame(np_scaled)
caso=caso[20000:40000]#Cogemos 20k instancias

#DATA
test_then_train_per=0.75
window_size=50#50

#PAIRED LEARNERS PARAMETERS
threshold=0.05#0.01
class_thresholds=[0.8,0.1,0.8,0.1,0.1]
ca_dims=3#len(columns)
'''

'''
#WEATHER
data_type='weather'
path='yourpath/weather.csv'
caso = pd.read_csv(path, sep=',', header=0)

x = caso.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
caso = pd.DataFrame(x_scaled)
columns=caso.iloc[:,0:8].columns

#DATA
test_then_train_per=0.5
window_size=50#100

#PAIRED LEARNERS PARAMETERS
threshold=0.5#0.1
class_thresholds=[0.8,0.1,0.8,0.1,0.1]
ca_dims=3#len(columns)
'''


#GMSC
data_type='gmsc'
path='yourpath//cs-training_Amazon_def.csv'
caso = pd.read_csv(path, sep=',', header=0)

caso = caso.drop('Unnamed: 0', 1)#Quitamos la primera columna
caso=caso.dropna(how='any')#Se quitan las filas con Nan
caso=caso[0:20000]#Limitar datos a 20k samples    
caso.columns=['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents', 'class']


x = caso.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
caso = pd.DataFrame(x_scaled)
columns=caso.iloc[:,0:10].columns

#DATA
test_then_train_per=0.75
window_size=250#100

#PAIRED LEARNERS PARAMETERS
threshold=0.01
class_thresholds=[0.001,0.0001,0.001,0.01,0.001]
ca_dims=3#5,7,10


'''
#POKER
data_type='poker'
path='yourpath//norm.csv'
caso = pd.read_csv(path, sep=',', header=None)
caso=caso.iloc[np.random.permutation(len(caso))]
caso=caso.iloc[:20000]

#x = caso.iloc[:,0:10].values
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)

feats = pd.DataFrame(caso.iloc[:,0:10])
columns=feats.columns

#DATA
test_then_train_per=0.75
window_size=250#100

#PAIRED LEARNERS PARAMETERS
threshold=0.001
class_thresholds=[0.1,0.01,0.1,0.01,0.001]
ca_dims=3##5,7,10
'''

#==============================================================================
# VARIABLES
#==============================================================================

#Global
preparatory_per=1-test_then_train_per
scoring='accuracy'#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
cv=10
max_iter=1
knn_max_window_size=window_size

knn_optimized_params=[5,knn_max_window_size,30]#Solo se usa si se comenta la funcion de hyperparameter tuning

#CA
runs=1#5,10
bins_margin=0.1#0.1

#==============================================================================
# DATA SLICING
#==============================================================================

if data_type=='elec2':
    XT=caso.iloc[:,0:4]
    YT=caso.iloc[:,4]
elif data_type=='weather':     
    XT=caso.iloc[:,0:8]
    YT=caso.iloc[:,8]
elif data_type=='gmsc':     
    XT=caso.iloc[:,0:10]
    YT=caso.iloc[:,10] 
elif data_type=='poker':     
    XT=feats
    YT=caso.iloc[:,10]      


#Data
features=pd.DataFrame(XT)
labels=pd.DataFrame(YT)
features.columns=columns            
labels.columns=['class']

#Data slicing
X_init=features.iloc[0:math.ceil(features.shape[0]*preparatory_per),:]
y_init=labels.iloc[0:math.ceil(labels.shape[0]*preparatory_per),:]
X_test_then_train=features.iloc[math.ceil(features.shape[0]*preparatory_per):,:]
y_test_then_train=labels.iloc[math.ceil(labels.shape[0]*preparatory_per):,:]

#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    
    # Ignore warnings
    import warnings
    warnings.simplefilter("ignore")
            
    SCORES_ca_automatas=[]
    
    SCORES_sgdc=[]
    SCORES_htc=[]
    SCORES_mtc=[]
    SCORES_pac=[]
    SCORES_gnbc=[]
    SCORES_knn=[]
    SCORES_mlpc=[]

    for ru in range(runs):

        print ('-RUN='+str(ru))
                
        SGDC=SGDClassifier()
        HTC=HoeffdingTree()
        MTC=MondrianTreeClassifier()
        PAC=PassiveAggressiveClassifier()
        GNBC=GaussianNB()
        KNN=lazy.KNN()
        MLPC=MLPClassifier()
                    
        classifiers=[SGDC,HTC,PAC,KNN]
                        
        #Stream classifiers hyper-parameter tuning        
        print ('STREAMERS HYPERPARAMETER TUNING ...')        
        classifiers,knn_optimized_params=hyperparametertuning_classifiers(classifiers,scoring,cv,X_init,y_init,max_iter,knn_max_window_size)
        class_names=get_classifiers_names(classifiers)
        
        #Stream paired_learners pre-training        
        print ('STREAM PAIRED LEARNERS PRE-TRAINING ...')
        
        online_accuracies=[]        

        for c in range(len(classifiers)):
#            predicciones.append([])
            online_accuracies.append([])
        
        for c in range(len(classifiers)):                            
            #Pre-training
            print(class_names[c],' ...')

            if class_names[c]=='KNN':                    
                classifiers[c].fit(X_init.as_matrix(),y_init.as_matrix().ravel())
                
                for x in range(X_init.shape[0]):
                    online_accuracies[c].append(0)
            else:
            
                for x in range(X_init.shape[0]):
                    sample=np.array(X_init.iloc[x,:]).reshape(1, -1)
                    lab=np.array(y_init.iloc[x,:])
                
                    classifiers[c].partial_fit(sample,lab,np.unique(y_init))
                    online_accuracies[c].append(0)
                                        
        print('CREATING PAIRED STREAM LEARNERS')
        paired_learners=[]
        #Defining paired learners
        for c in range(len(classifiers)):            
            paired=PairedLearners(stable_learner=classifiers[c], reactive_learner=classifiers[c], window_size=window_size, threshold=class_thresholds[c])
            paired_learners.append(paired)
            
        paired_learners_names=get_paired_learners_names(paired_learners)
        
        stream_detections=[[] for _ in range(len(paired_learners))]
                                        
        ######################## CELLULAR AUTOMATAS ########################

        # Initializin models
        
        print ('INITIALIZING AUTOMATAS ...')
        num_dims=[ca_dims]*X_init.shape[1]
        
        von_neumann_5=CA_VonNeumann_Classifier(bins=[],bins_margin=bins_margin,dimensions=num_dims, cells=empties(num_dims))        
        cellular_automatas=[von_neumann_5]
                
        print('CREATING PAIRED AUTOMATAS')
        paired_automatas=[]
        automat_thresholds=[threshold]*len(cellular_automatas)

        paired_automatas_names=cellular_automatas_naming(cellular_automatas,columns)
        
        #Defining paired learners
        for c in range(len(cellular_automatas)):   
            print(paired_automatas_names[c],' ...')
            paired=AutomataPairedLearners(stable_learner=cellular_automatas[c], reactive_learner=cellular_automatas[c], window_size=window_size, threshold=automat_thresholds[c])
            paired_automatas.append(paired)
                    
        automatas_detections=[[] for _ in range(len(paired_automatas))]
        automatas_stable_limits=[[] for _ in range(len(paired_automatas))]
        automatas_reactive_limits=[[] for _ in range(len(paired_automatas))]
        automatas_stable_mutations=[[] for _ in range(len(paired_automatas))]
        automatas_reactive_mutations=[[] for _ in range(len(paired_automatas))]

        # Initialize model data
        print ('CELLULAR AUTOMATAS PRE-TRAINING ...')
                
        for pa in range(len(paired_automatas)):
#            paired_automat=deepcopy(paired_automatas[pa])
            print(paired_automatas_names[pa],' ...')
            paired_automat,stable_limits_automat,reactive_limits_automat=paired_automatas[pa].fit(X_init.as_matrix(), y_init.as_matrix().ravel())
            paired_automatas[pa]=paired_automat
            automatas_stable_limits[pa]=stable_limits_automat
            automatas_reactive_limits[pa]=reactive_limits_automat
            
        print('FIN CELLULAR AUTOMATAS PRE-TRAINING')
            
        ca_scores=[[]]*len(paired_automatas)
        ca_mutaciones=[[]]*len(paired_automatas)
            
        for s in range(len(ca_scores)): 
            ca_scores[s]=list(np.zeros(X_init.shape[0]))
            ca_mutaciones[s]=list(np.zeros(X_init.shape[0]))
                                                        
        ######################## TEST-THEN-TRAIN PROCESSING ########################
        print ('TEST-THEN-TRAIN PROCESS ...')

        f_streamers=[1]*len(paired_learners)
        f_automatas=[1]*len(paired_automatas)

        buchaquer_X=list(X_init.as_matrix())
        buchaquer_y=list(y_init.as_matrix())
                     
        clss=np.unique(y_test_then_train)
        for s in range(X_test_then_train.shape[0]):                        
                         
            if s%500==0:
                print('run=',str(ru),' - s=',s)
            
            sample=np.array(X_test_then_train.iloc[s,:]).reshape(1, -1)
            lab=np.array(y_test_then_train.iloc[s,:])

            buchaquer_X.append(sample[0])
            buchaquer_y.append(lab)
            
            muta=[False]*len(paired_automatas)            
                        
            #window of samples
            if window_size>s:
                for c in range(len(paired_learners)):
                    online_accuracies[c].append(0)
                    
                for pa in range(len(paired_automatas)):
                    ca_scores[pa].append(0)
                    
            elif window_size==s:
                w_X_test_then_train=X_test_then_train.iloc[s-window_size:s]
                w_y_test_then_train=y_test_then_train.iloc[s-window_size:s]

                for c in range(len(paired_learners)):
                    #Training                    
                    if paired_learners_names[c]=='KNN':                    
                        paired_learners[c].fit(w_X_test_then_train.as_matrix(),w_y_test_then_train.as_matrix().ravel())
                                                    
                    else:
                    
                        for x in range(w_X_test_then_train.shape[0]):
                            sam=np.array(w_X_test_then_train.iloc[x,:]).reshape(1, -1)
                            la=np.array(w_y_test_then_train.iloc[x,:])
                        
                            paired_learners[c].partial_fit(sam,la,w_X_test_then_train.as_matrix(),w_y_test_then_train.as_matrix().ravel(),clss,knn_optimized_params)                            
                            
                    online_accuracies[c].append(0)                                            

                for pa in range(len(paired_automatas)):
                    #Training   
                    stable_limits_automat=automatas_stable_limits[pa]
                    reactive_limits_automat=automatas_reactive_limits[pa]
                    muta_stable=automatas_stable_mutations[pa]
                    muta_react=automatas_reactive_mutations[pa]
                                        
                    paired_automatas[pa],limits_stable,mut_stab,limits_reactive,mut_react=paired_automatas[pa].partial_fit(sample,lab,w_X_test_then_train.as_matrix(),w_y_test_then_train.as_matrix().ravel(),clss,stable_limits_automat,reactive_limits_automat)
                    automatas_stable_limits[pa]=limits_stable
                    automatas_reactive_limits[pa]=limits_reactive
                    automatas_stable_mutations[pa]=mut_stab
                    automatas_reactive_mutations[pa]=mut_react
                    
                    if muta[pa]:
                        ca_mutaciones[pa].append(1)
                    else:
                        ca_mutaciones[pa].append(0)
                    
                
            else:
                w_X_test_then_train=X_test_then_train.iloc[s-window_size:s]
                w_y_test_then_train=y_test_then_train.iloc[s-window_size:s]

                #STREAMERS
                for c in range(len(paired_learners)):
                                        
                    #Testing
                    pred=paired_learners[c].predict(sample)
                    
                    #Scoring
                    preqAcc=prequential_acc(pred,lab,online_accuracies[c],s,f_streamers[c])
                    online_accuracies[c].append(preqAcc)
                    
                    #Training
                    paired_learners[c].partial_fit(sample,lab,w_X_test_then_train.as_matrix(),w_y_test_then_train.as_matrix().ravel(),clss,knn_optimized_params)
                                        
                    #Detection and adaptation
                    if paired_learners[c].changeDetected>0:
                        print('DRIFT de ',paired_learners_names[c],' en t=',s)
                        stream_detections[c].append(s)
                        f_streamers[c]=s+1
                        
                        #Adaptation
                        paired_learners[c].driftAdaptation(w_X_test_then_train,w_y_test_then_train,clss)

                #CELLULAR AUTOMATAS 
                for pa in range(len(paired_automatas)):
                    #Testing
                    pred=paired_automatas[pa].predict(sample)
                
                    #Scoring
                    preqAcc=prequential_acc(pred,lab,ca_scores[pa],s,f_automatas[pa])
                    ca_scores[pa].append(preqAcc)
                
                    #Training   
                    stable_limits_automat=automatas_stable_limits[pa]
                    reactive_limits_automat=automatas_reactive_limits[pa]
                    muta_stable=automatas_stable_mutations[pa]
                    muta_react=automatas_reactive_mutations[pa]
                                        
                    paired_automatas[pa],limits_stable,mut_stab,limits_reactive,mut_react=paired_automatas[pa].partial_fit(sample,lab,w_X_test_then_train.as_matrix(),w_y_test_then_train.as_matrix().ravel(),clss,stable_limits_automat,reactive_limits_automat)
                    automatas_stable_limits[pa]=limits_stable
                    automatas_reactive_limits[pa]=limits_reactive
                    automatas_stable_mutations[pa]=mut_stab
                    automatas_reactive_mutations[pa]=mut_react
                    
                    if muta[pa]:
                        ca_mutaciones[pa].append(1)
                    else:
                        ca_mutaciones[pa].append(0)

                    #Detection and adaptation
                    if paired_automatas[pa].changeDetected>0:
                        print('CA detection en ',paired_automatas_names[pa])
                        automatas_detections[pa].append(s)
                        f_automatas[pa]=s+1

                        #Se hace la adaptacion                                                
                        paired_automatas[pa],stable_limits_automat,muta_stable,reactive_limits_automat=paired_automatas[pa].driftAdaptation(w_X_test_then_train.as_matrix(),w_y_test_then_train.as_matrix().ravel(),clss,reactive_limits_automat,muta_react)                        
                        automatas_reactive_limits[pa]=reactive_limits_automat
                        automatas_stable_limits[pa]=stable_limits_automat
                        automatas_stable_mutations[pa]=muta_stable

        ######################## ONLINE ACCURACY ########################        
        for c in range(len(paired_learners)):
            name=paired_learners_names[c]
            if name=='PAC':
                SCORES_pac.append(online_accuracies[c])
            elif name=='SGDC':
                SCORES_sgdc.append(online_accuracies[c])
            elif name=='MTC':
                SCORES_mtc.append(online_accuracies[c])
            elif name=='HTC':
                SCORES_htc.append(online_accuracies[c])
            elif name=='GNBC':
                SCORES_gnbc.append(online_accuracies[c])
            elif name=='KNN':
                SCORES_knn.append(online_accuracies[c])
            elif name=='MLPC':
                SCORES_mlpc.append(online_accuracies[c])

        SCORES_ca_automatas.append(ca_scores)

    ######################## SAVE RESULTS ########################    
    ruta_pickle='/home/txuslopez/Dropbox/jlopezlobo/Publicaciones/INFORMATION_SCIENCES_2019/PY/results/'+str(data_type)+'/'
    save_data(ruta_pickle,SCORES_pac,SCORES_sgdc,SCORES_htc,SCORES_gnbc,SCORES_knn,SCORES_ca_automatas,data_type)

    ######################## LOAD RESULTS ########################    
    SCORES_pac,SCORES_sgdc,SCORES_htc,SCORES_gnbc,SCORES_knn,SCORES_ca_automatas=load_data(ruta_pickle,data_type)

    ######################## RESULTS ########################  
    pac_results_mean=np.mean(SCORES_pac,axis=0)
    pac_results_std=np.std(SCORES_pac,axis=0)

    sgdc_results_mean=np.mean(SCORES_sgdc,axis=0)
    sgdc_results_std=np.std(SCORES_sgdc,axis=0)

    htc_results_mean=np.mean(SCORES_htc,axis=0)
    htc_results_std=np.std(SCORES_htc,axis=0)

    knn_results_mean=np.mean(SCORES_knn,axis=0)
    knn_results_std=np.std(SCORES_knn,axis=0)
    
    automatas_results_mean=np.mean(SCORES_ca_automatas,axis=0)
    automatas_results_std=np.std(SCORES_ca_automatas,axis=0)
                
    #STREAMERS Texttable
    pac_r_m=pac_results_mean[X_init.shape[0]:]
    pac_r_std=pac_results_std[X_init.shape[0]:]
    sgdc_r_m=sgdc_results_mean[X_init.shape[0]:]
    sgdc_r_std=sgdc_results_std[X_init.shape[0]:]
    htc_r_m=htc_results_mean[X_init.shape[0]:]
    htc_r_std=htc_results_std[X_init.shape[0]:]
    knn_r_m=knn_results_mean[X_init.shape[0]:]
    knn_r_std=knn_results_std[X_init.shape[0]:]
    
    streamers_Texttable(classifiers,class_names,pac_r_m,pac_r_std,sgdc_r_m,sgdc_r_std,htc_r_m,htc_r_std,knn_r_m,knn_r_std)                        

    #CELLULAR AUTOMATAS Texttable  
    automatas_Texttable(paired_automatas,paired_automatas_names,automatas_results_mean,automatas_results_std,X_init)

    ######################## PLOTS ########################    

    size_X=15
    size_Y=7
    colors=['b','g','r','y','m','c','pink','k','orange','palegreen','gold'] 
    font_size=20
    title=''

    rolling_w=500    
    num_cel_automatas=len(paired_automatas)
    plot_results(size_X,size_Y,colors,font_size,title,caso,num_cel_automatas,automatas_results_mean,automatas_results_std,paired_automatas_names,pac_results_mean,pac_results_std,sgdc_results_mean,sgdc_results_std,htc_results_mean,htc_results_std,knn_results_mean,knn_results_std,X_init,rolling_w,data_type)
    
        