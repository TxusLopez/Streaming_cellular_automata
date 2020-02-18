
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
#from skmultiflow.lazy.knn import KNN
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
#from skmultiflow.rules import VFDR
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.base import clone

import matplotlib as mpl
import matplotlib.animation as animat; animat.writers.list()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpngw
import matplotlib.style as style

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 22})
#mpl.rcParams['lines.linewidth'] = 2.0

#style.use('seaborn-dark') #sets the size of the charts
#style.use('ggplot')

#==============================================================================
# CLASSES
#==============================================================================
        
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

    grid5 = {'dimensions': [[15,15]],
               'cells': [empties([15,15])],
               'bins': [[]]
               }

    grid6 = {'dimensions': [[20,20]],
               'cells': [empties([20,20])],
               'bins': [[]]
               }

    grid7 = {'dimensions': [[25,25]],
               'cells': [empties([25,25])],
               'bins': [[]]
               }

    grid8 = {'dimensions': [[30,30]],
               'cells': [empties([30,30])],
               'bins': [[]]
               }

    grid9 = {'dimensions': [[40,40]],
               'cells': [empties([40,40])],
               'bins': [[]]
               }

    grid10 = {'dimensions': [[50,50]],
               'cells': [empties([50,50])],
               'bins': [[]]
               }
    
    lst_dicts_params=[grid1,grid2,grid3,grid4,grid5,grid6,grid7,grid8,grid9,grid10]
                
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
    
    for cl in range(len(classifiers)):
    
        cl_name=classifiers[cl].__class__.__name__                                                

        if cl_name=='PassiveAggressiveClassifier':
#            print (cl_name,' tuning ...')

            PAC_grid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                        'max_iter': [max_iter,100,200,500]            
            }                            
            
            grid_cv_PAC = RandomizedSearchCV(classifiers[cl], PAC_grid, cv=cv,scoring=scoring)
            grid_cv_PAC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_PAC.best_estimator_                      
    
        elif cl_name=='SGDClassifier':
#            print (cl_name,' tuning ...')

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
#            print (cl_name,' tuning ...')

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
#            print (cl_name,' tuning ...')

            MTC_grid = {'max_depth': [None,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                          'min_samples_split': [2, 5, 10]
                          }        
            
            grid_cv_MTC = RandomizedSearchCV(classifiers[cl], MTC_grid, cv=cv,scoring=scoring)
            grid_cv_MTC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_MTC.best_estimator_                        

        elif cl_name=='MondrianForestClassifier':
#            print (cl_name,' tuning ...')

            MFR_grid = {'n_estimators': [5,10,25,50,100],
                          'max_depth': [None,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                          'min_samples_split': [2, 5, 10]
                          }        
            
            grid_cv_MFC = RandomizedSearchCV(classifiers[cl], MFR_grid, cv=cv,scoring=scoring)
            grid_cv_MFC.fit(X_init,y_init)                
            classifiers[cl]=grid_cv_MFC.best_estimator_

        elif cl_name=='KNN':
#            print (cl_name, ' No tuning yet! ')

            KNN_grid = {'n_neighbors': [5,10,15,20],
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

            classifiers[cl]=lazy.KNN(n_neighbors=n_neighbors, max_window_size=knn_max_window_size, leaf_size=leaf_size)

#        elif cl_name=='VFDR':
#            print (cl_name, ' No tuning yet! ')

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
                            
    return classifiers

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

def plot_CA_boundaries_stream(cellular_automatas,buchaquer_X,buchaquer_y,X_columns,y_columns,sample,mutaciones,ca_names):
            
    idxs=[[]]*len(cellular_automatas)

    for ca in range(len(cellular_automatas)):
        
        dim=cellular_automatas[ca].dimensions
    
        # Create image arrays
        img = deepcopy(empties(dim))
    
        # Set variables to model results
        cells=cellular_automatas[ca].cells
    
        for j, c in enumerate(sample[0]):
            idxs.append(np.argmax(c <= cellular_automatas[ca].bins[j]))            
        
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
  

def plot_CA_boundaries(cellular_aut,buch_X,buch_y,X_columns,y_columns,ca_names,punto):

    images=[]
    
    for ca in range(len(cellular_aut)):
                                
        dim=cellular_aut[ca].dimensions
        # Create image arrays
        img = deepcopy(empties(dim))
        # Set variables to model results
        cells = cellular_aut[ca].cells
            
        for i in range(0, len(cells)):
            for j in range(0, len(cells)):
                                
                if cells[i][j]:                                              
                    s = cells[i][j][0].species
#                    rgb = (np.zeros(3)).tolist()

#                    rgb = [102,178,255]
#                    rgb[int(s)] = 50

#                    if int(s)==0:
#                        rgb = [1,155,196]
#                    else:
#                        rgb = [37,52,148]                        

#                    if int(s)==0:
#                        rgb = [216,179,101]
#                    else:
#                        rgb = [90,180,172]                        

                    if int(s)==0:
                        rgb = [254,232,138]
                    else:
                        rgb = [196,121,0]                        

                    img[i][j] = rgb
                else:
                    img[i][j] = [255,255,255]
    
        # Convert image arrays to appropriate data types
        rotated_img= np.rot90(img, 1)
        img = np.array(rotated_img, dtype='uint8')
        images.append(img)
    
    # Show the results
#    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(14, 7))
    
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1,4,1, aspect=1.0)
    ax2 = fig.add_subplot(1,4,2)
    ax3 = fig.add_subplot(1,4,3)
    ax4 = fig.add_subplot(1,4,4)
    
    
    buch_pd_X=pd.DataFrame(buch_X)
    buch_pd_X.columns=X_columns
    buch_pd_y=pd.DataFrame(buch_y)
    buch_pd_y.columns=[y_columns]
    
    todo=pd.concat([buch_pd_X,buch_pd_y],axis=1)
    
    X1=todo[todo[y_columns]==0]
    X2=todo[todo[y_columns]==1]
#    X3=todo[todo['class']==2]
    
    # Data Subplot
#    ax1.set_yticklabels([])
#    ax1.set_xticklabels([])
    ax1.set_xlabel('$x_1$',fontsize=22)
    ax1.set_ylabel('$x_2$',fontsize=22)    
    ax1.title.set_text('Learned instances')
    ax1.scatter(X1.iloc[:,0], X1.iloc[:,1], color='#FEE88A', marker='.',edgecolors='k',linewidths=0.0, s=200)#FEE88A
    ax1.scatter(X2.iloc[:,0], X2.iloc[:,1], color='#C47900', marker='.',edgecolors='k',linewidths=0.0, s=200)#C47900
    
    # Method 1 Subplot
    ax2.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    
    ax2.title.set_text(ca_names[0])
    ax2.imshow(images[0])

    # Method 2 Subplot
    ax3.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    
    ax3.title.set_text(ca_names[1])
    ax3.imshow(images[1])

    # Method 3 Subplot
    ax4.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    
    ax4.title.set_text(ca_names[2])
    ax4.imshow(images[2])
    
    fig.tight_layout()    

    plt.savefig('current_image_'+str(punto)+'.svg')
    
    plt.show() 
        
def plot_CA(ca):

    dim=ca.dimensions
    # Create image arrays
    img = deepcopy(empties(dim))
    # Set variables to model results
    cells = deepcopy(ca.cells)
        
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Method 1 Subplot
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)

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

def cellular_automatas_naming(cellular_automatas,title):
    
    ca_names=[str()]*len(cellular_automatas)
    for ca in range(len(cellular_automatas)):
        if cellular_automatas[ca].__class__.__name__=='CA_VonNeumann_Classifier':            
            if ca==0:
#                ca_names[ca]='VN_5x5_'+title
                ca_names[ca]=r'\texttt{sCA} $5\times5$'
            elif ca==1:
#                ca_names[ca]='VN_10x10_'+title
                ca_names[ca]=r'\texttt{sCA} $10\times10$'
            elif ca==2:
#                ca_names[ca]='VN_20x20_'+title
                ca_names[ca]=r'\texttt{sCA} $20\times20$'

#        elif cellular_automatas[ca].__class__.__name__=='CA_Moore_Classifier':            
#            if ca==3:
#                ca_names[ca]='M_5x5_'+title
#            elif ca==4:
#                ca_names[ca]='M_10x10_'+title
#            elif ca==5:
#                ca_names[ca]='M_20x20_'+title

#        elif cellular_automatas[ca].__class__.__name__=='CA_Gradient_Classifier':            
#            if ca==6:
#                ca_names[ca]='G_5x5'
#            elif ca==7:
#                ca_names[ca]='G_10x10'
#            elif ca==8:
#                ca_names[ca]='G_20x20'
    
    return ca_names

def streamers_Texttable(classifiers,class_names,pac_results_mean,pac_results_std,sgdc_results_mean,sgdc_results_std,mtc_results_mean,mtc_results_std,htc_results_mean,htc_results_std,gnbc_results_mean,gnbc_results_std,mlpc_results_mean,mlpc_results_std,bd,ad,drift_position,measure_position_after_drift,t_streamers):
    
    for c in range(len(classifiers)):

        if class_names[c]=='PAC':

            m_bd=np.round((pac_results_mean[bd]),3)
            std_bd=np.round((pac_results_std[bd]),3)
                             
            m_d=np.round((pac_results_mean[measure_position_after_drift]),3)                
            std_d=np.round((pac_results_std[measure_position_after_drift]),3)

            m_ad=np.round((pac_results_mean[ad]),3)                 
            std_ad=np.round((pac_results_std[ad]),3)
            
        elif class_names[c]=='SGDC':

            m_bd=np.round((sgdc_results_mean[bd]),3)
            std_bd=np.round((sgdc_results_std[bd]),3)
                             
            m_d=np.round((sgdc_results_mean[measure_position_after_drift]),3)                
            std_d=np.round((sgdc_results_std[measure_position_after_drift]),3)

            m_ad=np.round((sgdc_results_mean[ad]),3)                 
            std_ad=np.round((sgdc_results_std[ad]),3)
            
        elif class_names[c]=='MTC':

            m_bd=np.round((mtc_results_mean[bd]),3)
            std_bd=np.round((mtc_results_std[bd]),3)
                             
            m_d=np.round((mtc_results_mean[measure_position_after_drift]),3)                
            std_d=np.round((mtc_results_std[measure_position_after_drift]),3)

            m_ad=np.round((mtc_results_mean[ad]),3)                 
            std_ad=np.round((mtc_results_std[ad]),3)
            
        elif class_names[c]=='HTC':

            m_bd=np.round((htc_results_mean[bd]),3)
            std_bd=np.round((htc_results_std[bd]),3)
                             
            m_d=np.round((htc_results_mean[measure_position_after_drift]),3)                
            std_d=np.round((htc_results_std[measure_position_after_drift]),3)

            m_ad=np.round((htc_results_mean[ad]),3)                 
            std_ad=np.round((htc_results_std[ad]),3)
            
        elif class_names[c]=='GNBC':

            m_bd=np.round((gnbc_results_mean[bd]),3)
            std_bd=np.round((gnbc_results_std[bd]),3)
                             
            m_d=np.round((gnbc_results_mean[measure_position_after_drift]),3)                
            std_d=np.round((gnbc_results_std[measure_position_after_drift]),3)

            m_ad=np.round((gnbc_results_mean[ad]),3)                 
            std_ad=np.round((gnbc_results_std[ad]),3)

        elif class_names[c]=='KNN':

            m_bd=np.round((knn_results_mean[bd]),3)
            std_bd=np.round((knn_results_std[bd]),3)
                             
            m_d=np.round((knn_results_mean[measure_position_after_drift]),3)                
            std_d=np.round((knn_results_std[measure_position_after_drift]),3)

            m_ad=np.round((knn_results_mean[ad]),3)                 
            std_ad=np.round((knn_results_std[ad]),3)

        elif class_names[c]=='MLPC':

            m_bd=np.round((mlpc_results_mean[bd]),3)
            std_bd=np.round((mlpc_results_std[bd]),3)
                             
            m_d=np.round((mlpc_results_mean[measure_position_after_drift]),3)                
            std_d=np.round((mlpc_results_std[measure_position_after_drift]),3)

            m_ad=np.round((mlpc_results_mean[ad]),3)                 
            std_ad=np.round((mlpc_results_std[ad]),3)
            
        t_streamers.add_rows([['STREAMERS', 'Accuracy BD','Accuracy D','Accuracy AD'],[class_names[c],str(m_bd)+str('+-')+str(std_bd),str(m_d)+str('+-')+str(std_d),str(m_ad)+str('+-')+str(std_ad)]])

    print (t_streamers.draw())  
    
def automatas_Texttable(cellular_automatas,automatas_results_mean,automatas_results_std,bd,ad,drift_position,measure_position_after_drift,t_automatas,title,names):    
    
    bd_automatas_mean=[[]]*len(cellular_automatas)
    bd_automatas_std=[[]]*len(cellular_automatas)
    for h in range(len(cellular_automatas)):
        bd_automatas_mean[h]=np.round((automatas_results_mean[h][bd]),3)
        bd_automatas_std[h]=np.round((automatas_results_std[h][bd]),3)

    d_automatas_mean=[[]]*len(cellular_automatas)
    d_automatas_std=[[]]*len(cellular_automatas)
    for h in range(len(cellular_automatas)):
        d_automatas_mean[h]=np.round((automatas_results_mean[h][measure_position_after_drift]),3)
        d_automatas_std[h]=np.round((automatas_results_std[h][measure_position_after_drift]),3)

    ad_automatas_mean=[[]]*len(cellular_automatas)
    ad_automatas_std=[[]]*len(cellular_automatas)
    for h in range(len(cellular_automatas)):
        ad_automatas_mean[h]=np.round((automatas_results_mean[h][ad]),3)
        ad_automatas_std[h]=np.round((automatas_results_std[h][ad]),3)

    for h in range(len(cellular_automatas)):
        t_automatas.add_rows([['AUTOMATAS_'+title, 'Accuracy BD','Accuracy D','Accuracy AD'],[str(names[h]),str(bd_automatas_mean[h])+str('+-')+str(bd_automatas_std[h]),str(d_automatas_mean[h])+str('+-')+str(d_automatas_std[h]),str(ad_automatas_mean[h])+str('+-')+str(ad_automatas_std[h])]])
    
    print (t_automatas.draw())    

#def plot_automatas_results(size_X,size_Y,colors,font_size,title,XT,cellular_automatas,automatas_results_mean,automatas_results_std,ca_names,drift_position,measure_position_after_drift,X_init,drift_per,detection_point):
def plot_automatas_results(size_X,size_Y,colors,font_size,title,XT,cellular_automatas_con,automatas_con_results_mean,automatas_con_results_std,ca_names_con,cellular_automatas_sin,automatas_sin_results_mean,automatas_sin_results_std,ca_names_sin,drift_position,measure_position_after_drift,X_init,drift_per,detection_point):
    
#    fig=plt.figure(figsize=(size_X,size_Y))
    fig, axes = plt.subplots(1,1,figsize=(size_X,size_Y))
    
#    plt.title(title,size=font_size)
    axes.set_xlabel(r't',size=font_size)
    axes.set_ylabel(r'Prequential accuracy (t)',size=font_size)
#    plt.ylim(0.5,1.0)
    axes.set_xlim(0,XT.shape[0])
            
    for p in range(len(cellular_automatas_sin)):        
        axes.plot(automatas_sin_results_mean[p],color=colors[p],label=ca_names_sin[p],linestyle='-')
        axes.fill_between(range(len(automatas_sin_results_mean[p])), automatas_sin_results_mean[p]-automatas_sin_results_std[p], automatas_sin_results_mean[p]+automatas_sin_results_std[p],facecolor=colors[p], alpha=0.1)    

    for p in range(len(cellular_automatas_con)):        
        axes.plot(automatas_con_results_mean[p],color=colors[p],label=ca_names_con[p],linestyle='--')
        axes.fill_between(range(len(automatas_con_results_mean[p])), automatas_con_results_mean[p]-automatas_con_results_std[p], automatas_con_results_mean[p]+automatas_con_results_std[p],facecolor=colors[p], alpha=0.1)    

    lines = axes.get_lines()
    legend_sin = axes.legend([lines[i] for i in [0,1,2]], ca_names_sin, prop={'size': font_size},loc='lower left',fancybox=True, shadow=True,ncol=1,bbox_to_anchor=(0.12, 0.01))
    legend_con = axes.legend([lines[i] for i in [3,4,5]], ca_names_con, prop={'size': font_size},loc='lower right',fancybox=True, shadow=True,ncol=1,bbox_to_anchor=(0.88, 0.01))

        
    axes.axvspan(drift_position, drift_position+drift_per, alpha=0.5, color='#FEE88A')    

    axes.axvspan(0, X_init.shape[0], alpha=0.5, color='#C47900')    

    axes.axvline(x=drift_position,color='k', linestyle='-')
    axes.axvline(x=drift_position+detection_point,color='r', linestyle='-.')
    axes.axvline(x=measure_position_after_drift,color='k', linestyle='dotted')
    

#    plt.legend(prop={'size': font_size},loc='lower center',fancybox=True, shadow=True,ncol=2,bbox_to_anchor=(0.275, 0.01))
    axes.add_artist(legend_sin)
    axes.add_artist(legend_con)
    
    plt.savefig('results_synth_'+str(title)+'.svg')
    
    plt.show()  
    
def plot_streamers_results(size_X,size_Y,colors,font_size,title,XT,pac_results_mean,pac_results_std,sgdc_results_mean,sgdc_results_std,mtc_results_mean,mtc_results_std,htc_results_mean,htc_results_std,gnbc_results_mean,gnbc_results_std,knn_results_mean,knn_results_std,mlpc_results_mean,mlpc_results_std,drift_position):
    
    fig=plt.figure(figsize=(size_X,size_Y))
#    plt.title(title,size=font_size)
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('Prequential accuracy',size=font_size)
    plt.ylim(0.0,1.0)
    plt.xlim(0,XT.shape[0])      
        
    plt.plot(pac_results_mean,color=colors[3],label='PAC')
    plt.fill_between(range(len(pac_results_mean)), pac_results_mean-pac_results_std, pac_results_mean+pac_results_std,facecolor=colors[3], alpha=0.1)    

    plt.plot(sgdc_results_mean,color=colors[4],label='SGDC')
    plt.fill_between(range(len(sgdc_results_mean)), sgdc_results_mean-sgdc_results_std, sgdc_results_mean+sgdc_results_std,facecolor=colors[4], alpha=0.1)    

#    plt.plot(mtc_results_mean,color=colors[5],label='MTC')
#    plt.fill_between(range(len(mtc_results_mean)), mtc_results_mean-mtc_results_std, mtc_results_mean+mtc_results_std,facecolor=colors[5], alpha=0.1)    

    plt.plot(htc_results_mean,color=colors[7],label='HTC')
    plt.fill_between(range(len(htc_results_mean)), htc_results_mean-htc_results_std, htc_results_mean+htc_results_std,facecolor=colors[7], alpha=0.1)    

    plt.plot(gnbc_results_mean,color=colors[8],label='GNBC')
    plt.fill_between(range(len(gnbc_results_mean)), gnbc_results_mean-gnbc_results_std, gnbc_results_mean+gnbc_results_std,facecolor=colors[8], alpha=0.1)    

    plt.plot(knn_results_mean,color=colors[9],label='KNN')
    plt.fill_between(range(len(knn_results_mean)), knn_results_mean-knn_results_std, knn_results_mean+knn_results_std,facecolor=colors[9], alpha=0.1)    
    
#    plt.plot(mlpc_results_mean,color=colors[10],label='MLPC')
#    plt.fill_between(range(len(mlpc_results_mean)), mlpc_results_mean-mlpc_results_std, mlpc_results_mean+mlpc_results_std,facecolor=colors[10], alpha=0.1)    

#    plt.axvspan(drift_position, drift_position+periodo_drift, alpha=0.5, color='yellow')    

    plt.axvspan(0, X_init.shape[0], alpha=0.5, color='green')    

    plt.axvline(x=1000,color='k', linestyle='--')
    plt.axvline(x=measure_position_after_drift,color='k', linestyle='dotted')

#    plt.legend(prop={'size': font_size},loc='upper right',fancybox=True, shadow=True,ncol=4,bbox_to_anchor=(0.80, 1.18))   
    plt.legend(prop={'size': font_size},loc='upper right',fancybox=True, shadow=True)   
    
    plt.show()
          

#==============================================================================
# DATASETS
#==============================================================================

data_type='minku'
severity=3
speed=3
problem_name='sineH'
problem_name2='SineH'
path='yourpath//ArtificialDataSets//'    
fil=problem_name+'//data'+problem_name2+'Sev'+str(severity)+'Sp'+str(speed)+'Train.csv'
columns=['x1','x2']

raw_data= pd.read_csv(path + fil, sep=',',header=None)
caso=raw_data[raw_data.columns[0:3]]
caso.columns=['X1','X2','class']

#==============================================================================
# VARIABLES
#==============================================================================

#Global
test_then_train_per=0.95
preparatory_per=1-test_then_train_per
scoring='accuracy'#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
cv=10
max_iter=1
knn_max_window_size=100

#CA
runs=1#25
bins_margin=0.1#0.1

#CONCEPT DRIFT
window_size=0
detection_point=0
drift_position=1000

drift_period=0
if speed==1:
    drift_period=1
    window_size=25
    detection_point=drift_period+window_size

elif speed==2:
    drift_period=249
    window_size=50
    detection_point=drift_period+window_size

elif speed==3:
    drift_period=499
    window_size=100
    detection_point=drift_period+window_size
    
#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    
    # Ignore warnings
    import warnings
    warnings.simplefilter("ignore")
    
    XT=caso.iloc[:,0:2]
    YT=caso.iloc[:,2]
    
    SCORES_ca_automatas_sin=[]
    SCORES_ca_automatas_con=[]
    
    SCORES_sgdc=[]
    SCORES_htc=[]
    SCORES_mtc=[]
    SCORES_pac=[]
    SCORES_gnbc=[]
    SCORES_knn=[]
    SCORES_mlpc=[]

    for ru in range(runs):

        print ('-RUN='+str(ru))
                
        #Defining streamers
        SGDC=SGDClassifier()
        HTC=HoeffdingTree()
        MTC=MondrianTreeClassifier()
        PAC=PassiveAggressiveClassifier()
        GNBC=GaussianNB()
        KNN=lazy.KNN()
        MLPC=MLPClassifier()
                    
        classifiers=[SGDC,HTC,PAC,KNN]
        
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
        
#        measure_position_after_drift=drift_position+(2*window_size)
        measure_position_after_drift=drift_position+detection_point+(window_size)
        
        #Drift measures
        bd=drift_position-1
        ad=caso.shape[0]-1
        
        #Stream classifiers hyper-parameter tuning        
        print ('STREAMERS HYPERPARAMETER TUNING ...')
        
        classifiers=hyperparametertuning_classifiers(classifiers,scoring,cv,X_init,y_init,max_iter,knn_max_window_size)
        class_names=get_classifiers_names(classifiers)

        #Stream classifiers pre-training        
        print ('STREAMERS PRE-TRAINING ...')
        
        online_accuracies=[]        

        for c in range(len(classifiers)):
            online_accuracies.append([])
        
        for c in range(len(classifiers)):                            
            #Pre-training
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
                    
        ######################## CELLULAR AUTOMATAS ########################
        # Initializin models
        print ('INITIALIZING AUTOMATAS ...')
        von_neumann_5=CA_VonNeumann_Classifier(bins=[],bins_margin=bins_margin,dimensions=[5,5], cells=empties([5,5]))
        von_neumann_10=CA_VonNeumann_Classifier(bins=[],bins_margin=bins_margin,dimensions=[10,10], cells=empties([10,10]))
        von_neumann_20=CA_VonNeumann_Classifier(bins=[],bins_margin=bins_margin,dimensions=[20,20], cells=empties([20,20]))
        
        cellular_automatas_sin=[von_neumann_5,von_neumann_10,von_neumann_20]
        n_automatas=len(cellular_automatas_sin)
        limits_automatas_sin=list(np.zeros(len(cellular_automatas_sin)))

        cellular_automatas_con=[von_neumann_5,von_neumann_10,von_neumann_20]
        limits_automatas_con=list(np.zeros(len(cellular_automatas_con)))

        #CAs names
        ca_names_sin=cellular_automatas_naming(cellular_automatas_sin,'sin')
        ca_names_con=cellular_automatas_naming(cellular_automatas_con,'con')
                        
        # Initialize model data
        print ('CELLULAR AUTOMATAS PRE-TRAINING ...')
                        
        for ca in range(len(cellular_automatas_sin)):
            cel_automat_sin=deepcopy(cellular_automatas_sin[ca])
            lim_automat_sin=deepcopy(limits_automatas_sin[ca])
            cel_automat_sin,limits_automatas_sin[ca]=cel_automat_sin.fit(X_init.as_matrix(), y_init.as_matrix().ravel())
            cellular_automatas_sin[ca]=deepcopy(cel_automat_sin)

        for ca in range(len(cellular_automatas_con)):
            cel_automat_con=deepcopy(cellular_automatas_con[ca])
            lim_automat_con=deepcopy(limits_automatas_con[ca])
            cellular_automatas_con[ca],limits_automatas_con[ca]=cel_automat_con.fit(X_init.as_matrix(), y_init.as_matrix().ravel())
            
        ca_scores_sin=[[]]*len(cellular_automatas_sin)
        ca_mutaciones_sin=[[]]*len(cellular_automatas_sin)

        ca_scores_con=[[]]*len(cellular_automatas_con)
        ca_mutaciones_con=[[]]*len(cellular_automatas_con)
            
        for s in range(len(ca_scores_sin)): 
            ca_scores_sin[s]=list(np.zeros(X_init.shape[0]))
            ca_mutaciones_sin[s]=list(np.zeros(X_init.shape[0]))

        for s in range(len(ca_scores_con)): 
            ca_scores_con[s]=list(np.zeros(X_init.shape[0]))
            ca_mutaciones_con[s]=list(np.zeros(X_init.shape[0]))
                        
        print ('Inicializacion SIN')            
        plot_CA_boundaries(cellular_automatas_sin,X_init,y_init,columns,'class',ca_names_sin,'ini_sin')
        print ('Inicializacion CON')            
        plot_CA_boundaries(cellular_automatas_con,X_init,y_init,columns,'class',ca_names_con,'ini_con')
                                        
        ######################## TEST-THEN-TRAIN PROCESSING ########################
        print ('TEST-THEN-TRAIN PROCESS ...')

        f=1

        buchaquer_X_sin=list(X_init.as_matrix())
        buchaquer_y_sin=list(y_init.as_matrix())

        buchaquer_X_con=list(X_init.as_matrix())
        buchaquer_y_con=list(y_init.as_matrix())
                      
        for s in range(X_test_then_train.shape[0]):                        
                        
            sample=np.array(X_test_then_train.iloc[s,:]).reshape(1, -1)
            lab=np.array(y_test_then_train.iloc[s,:])
            
                
            ############################################## STREAMERS
            for c in range(len(classifiers)):
                #Testing
                pred=classifiers[c].predict(sample)
                #Scoring
                preqAcc=prequential_acc(pred,lab,online_accuracies[c],s,f)
                online_accuracies[c].append(preqAcc)
                #Training
                classifiers[c].partial_fit(sample,lab)

            ############################################## CELLULAR AUTOMATAS 
            muta_sin=[False]*len(cellular_automatas_sin)
            buchaquer_X_sin.append(sample[0])
            buchaquer_y_sin.append(lab)

            muta_con=[False]*len(cellular_automatas_con)
            buchaquer_X_con.append(sample[0])
            buchaquer_y_con.append(lab)
                           
            for ca in range(n_automatas):
                                
                #######WITHOUT
                cA_sin=deepcopy(cellular_automatas_sin[ca])
                lim_automat_sin=deepcopy(limits_automatas_sin[ca])
                #Testing
                pred=cA_sin.predict(sample)                
                #Scoring
                preqAcc=prequential_acc(pred,lab,ca_scores_sin[ca],s,f)
                ca_scores_sin[ca].append(preqAcc)                                
                #Training   
                cA_sin,limits_automatas_sin[ca],muta_sin[ca]=cA_sin.partial_fit(sample,lab,s,lim_automat_sin,muta_sin[ca])
                cellular_automatas_sin[ca]=deepcopy(cA_sin)
                #Mutations
                if muta_sin[ca]:
                    ca_mutaciones_sin[ca].append(1)
                else:
                    ca_mutaciones_sin[ca].append(0)

                #######WITH
                cel_automat_con=deepcopy(cellular_automatas_con[ca])
                lim_automat_con=deepcopy(limits_automatas_con[ca])
                #Testing
                pred=cel_automat_con.predict(sample)                
                #Scoring
                preqAcc=prequential_acc(pred,lab,ca_scores_con[ca],s,f)
                ca_scores_con[ca].append(preqAcc)                                
                #Training   
                cellular_automatas_con[ca],limits_automatas_con[ca],muta_con[ca]=cel_automat_con.partial_fit(sample,lab,s,lim_automat_con,muta_con[ca])
                #Mutations                
                if muta_con[ca]:
                    ca_mutaciones_con[ca].append(1)
                else:
                    ca_mutaciones_con[ca].append(0)
                        

            #CELLULAR ADAPTATION
            if s==drift_position-X_init.shape[0]+detection_point:
                                
                print ('Punto de adaptación en: ',s)
                                
                w_X_test_then_train=X_test_then_train.iloc[s-window_size:s]
                w_y_test_then_train=y_test_then_train.iloc[s-window_size:s]                
                                
                print ('Pre Adaptation WITHOUT')            
                plot_CA_boundaries(cellular_automatas_sin,buchaquer_X_sin,buchaquer_y_sin,columns,'class',ca_names_sin,'beforeadapt_sin')
                print ('Pre Adaptation WITH')            
                plot_CA_boundaries(cellular_automatas_con,buchaquer_X_con,buchaquer_y_con,columns,'class',ca_names_con,'beforeadapt_con')
                                
                #WITH
                for ca in range(len(cellular_automatas_con)):
                                            
                    #Training with adaptation
                    cel_con=deepcopy(cellular_automatas_con[ca])
                    dims=cel_con.dimensions
                    celdas=empties([dims[0],dims[0]])

                    new_automat=CA_VonNeumann_Classifier(bins=[],bins_margin=bins_margin,dimensions=dims,cells=celdas)

                    cellular_automatas_con[ca],limits_automatas_con[ca]=new_automat.fit(w_X_test_then_train.as_matrix(),w_y_test_then_train.as_matrix().ravel())                                                    

                buchaquer_X_con=list(w_X_test_then_train.as_matrix())
                buchaquer_y_con=list(w_y_test_then_train.as_matrix())
                
                print ('Post Adaptation WITHOUT')            
                plot_CA_boundaries(cellular_automatas_sin,buchaquer_X_sin,buchaquer_y_sin,columns,'class',ca_names_sin,'afteradapt_sin')
                print ('Post Adaptation WITH')            
                plot_CA_boundaries(cellular_automatas_con,buchaquer_X_con,buchaquer_y_con,columns,'class',ca_names_con,'afteradapt_con')


                        
            if s%100==0:
                print('s: ',s)

        print ('Final WITHOUT')            
        plot_CA_boundaries(cellular_automatas_sin,buchaquer_X_sin,buchaquer_y_sin,columns,'class',ca_names_sin,'final_sin')
        print ('Final WITH')            
        plot_CA_boundaries(cellular_automatas_con,buchaquer_X_con,buchaquer_y_con,columns,'class',ca_names_con,'final_con')
                
        ######################## ONLINE ACCURACY ########################        
        for c in range(len(classifiers)):
            if class_names[c]=='PAC':
                SCORES_pac.append(online_accuracies[c])
            elif class_names[c]=='SGDC':
                SCORES_sgdc.append(online_accuracies[c])
#            elif class_names[c]=='MTC':
#                SCORES_mtc.append(online_accuracies[c])
            elif class_names[c]=='HTC':
                SCORES_htc.append(online_accuracies[c])
#            elif class_names[c]=='GNBC':
#                SCORES_gnbc.append(online_accuracies[c])
            elif class_names[c]=='KNN':
                SCORES_knn.append(online_accuracies[c])
#            elif class_names[c]=='MLPC':
#                SCORES_mlpc.append(online_accuracies[c])

        SCORES_ca_automatas_sin.append(ca_scores_sin)
        SCORES_ca_automatas_con.append(ca_scores_con)


    ######################## RESULTS ########################  
    pac_results_mean=np.mean(SCORES_pac,axis=0)
    pac_results_std=np.std(SCORES_pac,axis=0)

    sgdc_results_mean=np.mean(SCORES_sgdc,axis=0)
    sgdc_results_std=np.std(SCORES_sgdc,axis=0)

#    mtc_results_mean=np.mean(SCORES_mtc,axis=0)
#    mtc_results_std=np.std(SCORES_mtc,axis=0)

    htc_results_mean=np.mean(SCORES_htc,axis=0)
    htc_results_std=np.std(SCORES_htc,axis=0)

#    gnbc_results_mean=np.mean(SCORES_gnbc,axis=0)
#    gnbc_results_std=np.std(SCORES_gnbc,axis=0)

    knn_results_mean=np.mean(SCORES_knn,axis=0)
    knn_results_std=np.std(SCORES_knn,axis=0)

#    mlpc_results_mean=np.mean(SCORES_mlpc,axis=0)
#    mlpc_results_std=np.std(SCORES_mlpc,axis=0)
    
    automatas_sin_results_mean=np.mean(SCORES_ca_automatas_sin,axis=0)
    automatas_sin_results_std=np.std(SCORES_ca_automatas_sin,axis=0)

    automatas_con_results_mean=np.mean(SCORES_ca_automatas_con,axis=0)
    automatas_con_results_std=np.std(SCORES_ca_automatas_con,axis=0)
        
    #STREAMERS Texttable
#    t_streamers = Texttable()
#    streamers_Texttable(classifiers,class_names,pac_results_mean,pac_results_std,sgdc_results_mean,sgdc_results_std,mtc_results_mean,mtc_results_std,htc_results_mean,htc_results_std,gnbc_results_mean,gnbc_results_std,mlpc_results_mean,mlpc_results_std,bd,ad,drift_position,measure_position_after_drift,t_streamers)

    #CELLULAR AUTOMATAS WITHOUT ADAPTATION Texttable  
    t_automatas_sin = Texttable()
    title='No adaptation'
    automatas_Texttable(cellular_automatas_sin,automatas_sin_results_mean,automatas_sin_results_std,bd,ad,drift_position,measure_position_after_drift,t_automatas_sin,title,ca_names_sin)

    #CELLULAR AUTOMATAS WITH ADAPTATION Texttable  
    t_automatas_con = Texttable()
    title='With adaptation'
    automatas_Texttable(cellular_automatas_con,automatas_con_results_mean,automatas_con_results_std,bd,ad,drift_position,measure_position_after_drift,t_automatas_con,title,ca_names_con)

    ######################## PLOTS ########################    

    size_X=15
    size_Y=7
    colors=['#FF6666','#FF9933','gold'] 
    font_size=22
    title=str(problem_name)+str(severity)+str(speed)


    #PLOT CELLULARS
    plot_automatas_results(size_X,size_Y,colors,font_size,title,XT,cellular_automatas_con,automatas_con_results_mean,automatas_con_results_std,ca_names_con,cellular_automatas_sin,automatas_sin_results_mean,automatas_sin_results_std,ca_names_sin,drift_position,measure_position_after_drift,X_init,drift_period,detection_point)
