import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import argparse
from GCN import *
from DQN import *
from ActionTeacher import *
from data_process import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_name', type=str, default='', help='train_dataset')
parser.add_argument('-o','--output_dir', type=str, help='output file')
parser.add_argument('-s','--step', default=2000, type=int, help='run steps')
parser.add_argument('-beta','--beta',default=10,type=int,help='weight of CORR when calculating r_list')

parser.add_argument('-imp','--importance',action='store_true', default=False)

parser.add_argument('-model','--model', default='DT', type=str, help='downstream task')

parser.add_argument('-teacher','--teacher',default='',type=str,help='select_teacher')

args = parser.parse_args()

   
# pretrain data
dataset = ReadData(args.data_name)
r, c = dataset.shape
array = dataset.values
X = dataset.iloc[:,0:(c-1)]
Y = dataset.iloc[:,(c-1)]
X_train, X_val, Y_train, Y_val = model_selection .train_test_split(X, Y, test_size=0.1, random_state=0)


model = DecisionTreeClassifier()

if args.teacher == 'KB': # KBT
    action_teacher = KBestAdvisor
elif args.teacher == 'DT': # DTT
    action_teacher = DTAdvisor
elif args.teacher == 'MIX': # Hybrid Teaching
    action_teacher = MIXAdvisor
else:
    action_teacher = None  

    
print(model)
print(action_teacher) 
print(args.importance)



# feature number
N_feature = X_train.shape[1] 
# feature length,i.e., sample number
N_sample = X_train.shape[0] 
# DQN params
BATCH_SIZE = 16
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = int(args.step/2)
N_ACTIONS = 2
N_STATES = len(X_train)
BETA = args.beta

# initialize
np.random.seed(0)
action_list = np.random.randint(2, size= N_feature)
i = 0
while sum(action_list) < 2:
    np.random.seed(i)
    action_list = np.random.randint(2, size= N_feature)
    i += 1

X_selected = X_train.iloc[:, action_list == 1]
model.fit(X_train.iloc[:, action_list == 1], Y_train)
s = Feature_GCN(X_selected)

accuracy = model.score(X_val.iloc[:, action_list == 1], Y_val)
ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
r_list = (accuracy - BETA * ave_corr) / sum(action_list) * action_list

action_list_p = action_list
dqn_list = []
for agent in range(N_feature):
    dqn_list.append(DQN(N_STATES, N_ACTIONS, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER,  MEMORY_CAPACITY))


RImp = np.ones(N_feature)
np.random.seed(0)

write_advice_name = args.output_dir.replace('log','advice').split('/')
write_advice_name.insert(-1,'tmp')
write_advice_name = '/'.join(write_advice_name)

write_imp_name = args.output_dir.replace('log','imp').split('/')
write_imp_name.insert(-1,'tmp')
write_imp_name = '/'.join(write_imp_name)


for i in range(args.step):
    if i % 100 == 0:
        print(i)
    
    action_list = np.zeros(N_feature)
    for agent, dqn in enumerate(dqn_list):
        action_list[agent] = dqn.choose_action(s)
    while sum(action_list) < 2:
        np.random.seed(i+1)
        action_list = np.random.randint(2, size=N_feature)
        i += 1
    
    if action_teacher is not None and i%2 == 0 and i < MEMORY_CAPACITY + 1:
        if args.teacher != 'MIX':
            temp = action_teacher(action_list_p,action_list,X_train,Y_train)
        else:
            temp = action_teacher(action_list_p,action_list,X_train,Y_train,i,MEMORY_CAPACITY)
            
        print('Advise! Advise Number: ' + str(N_feature - np.sum(temp==action_list))) 
        f = open(write_advice_name,'a+')
        f.write('Advise Number:' + str(N_feature - np.sum(temp==action_list))+'\n')
        f.close()
        action_list = temp
    
    if args.importance and i > (MEMORY_CAPACITY + 1):
        RImp += action_list
        f = open(write_imp_name,'a+')
        f.write(str(RImp) + '\n')
        f.close()
    
    
    X_selected = X_train.iloc[:, action_list == 1]
    model.fit(X_train.iloc[:, action_list == 1], Y_train)
    s_ = Feature_GCN(X_selected)


    accuracy = model.score(X_val.iloc[:, action_list == 1], Y_val)
    #ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
    ave_corr = X_val.iloc[:, action_list == 1].corr().abs().sum().sum() / (X_val.iloc[:, action_list == 1].shape[0] * X_val.iloc[:, action_list == 1].shape[1])
    # action_list_change = np.array([x or y for (x,y) in zip(action_list_p, action_list)])
    r_list = (accuracy - BETA * ave_corr) / sum(action_list) * action_list
    #print(accuracy,ave_corr)
    
    if args.importance and i > (MEMORY_CAPACITY + 1):
        r_list = RImp /np.sum(RImp * action_list) * r_list * np.sum(action_list)
        
    for agent, dqn in enumerate(dqn_list):
        dqn.store_transition(s, action_list[agent], r_list[agent], s_)

    if dqn_list[0].memory_counter > MEMORY_CAPACITY:
        for dqn in dqn_list:
            dqn.learn()
        f = open(args.output_dir, 'a+')
        f.write(str(sum(r_list)) + ' ' + str(accuracy) + '\n')
        f.write(str(model.predict(X_val.iloc[:,action_list==1])) + '\n')
        f.close()
        print(sum(r_list), accuracy)

    s = s_
    action_list_p = action_list



