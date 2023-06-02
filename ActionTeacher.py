import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def hot2idx(inp):
    return [idx for idx,item in enumerate(inp) if item > 0]

def idx2hot(inp,num_features):
    return [1 if i in inp else 0 for i in range(num_features)]


  
def KBestAdvisor(pre_action_list, cur_action_list, X_train, Y_train):
    N_feature = X_train.shape[1]
    cur_select_index = hot2idx(cur_action_list)
    
    pre_select_index = hot2idx(pre_action_list)
    pre_deselect_index = [i for i in range(N_feature) if i not in pre_select_index]
    
    select_select_index = set(cur_select_index) & set(pre_select_index)
    deselect_select_index = set(cur_select_index) & set(pre_deselect_index)
    
    k = int(len(select_select_index)/2) + len(deselect_select_index)
    k_action_list =  SelectKBest(mutual_info_classif,k).fit(X_train.iloc[:,cur_action_list==1],Y_train).get_support(indices=False)
    k_index = [cur_select_index[i] for i,item in enumerate(k_action_list) if item]
    
    k_assure_index = [i for i in deselect_select_index if i in k_index]
    advise_index = list(select_select_index) + k_assure_index
    
    return np.array(idx2hot(advise_index,N_feature))



# DT Advisor
def DTAdvisor(pre_action_list, cur_action_list, X_train, Y_train):
    N_feature = X_train.shape[1]
    cur_select_index = hot2idx(cur_action_list)
    
    pre_select_index = hot2idx(pre_action_list)
    pre_deselect_index = [i for i in range(N_feature) if i not in pre_select_index]
    
    select_select_index = set(cur_select_index) & set(pre_select_index)
    deselect_select_index = set(cur_select_index) & set(pre_deselect_index)
    
    clf = DecisionTreeClassifier()
    f_imp = clf.fit(X_train.iloc[:,cur_action_list==1],Y_train).feature_importances_
    
    imp_idx = [(x,cur_select_index[i]) for i,x in enumerate(f_imp)]
    sorted_result = sorted(imp_idx,key=lambda x:x[0],reverse=True)
    
    k = int(len(select_select_index)/2) + len(deselect_select_index)
    top_k_idx = [ x[1] for x in sorted_result[:k]]
    
    advise_index =  list(set(top_k_idx + list(select_select_index)))
    
    return np.array(idx2hot(advise_index,N_feature))

# HybridTeaching
def MIXAdvisor(pre_action_list, cur_action_list, X_train, Y_train, step, MEMORY_UNIT):
    if step <= MEMORY_UNIT/2:
        return KBestAdvisor(pre_action_list, cur_action_list, X_train, Y_train)
    else:
        return DTAdvisor(pre_action_list, cur_action_list, X_train, Y_train)

    
