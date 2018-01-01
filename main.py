#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:29:53 2017

@author: ben
"""

import numpy as np


###########################################
###########################################    
################## Preprocessing ##########
###########################################

root='/Users/ben/Documents/datasets/rong360/data/'

browse_txt=root+'train/browse_history_train.txt'
bank_txt=root+'train/bank_detail_train.txt'
overdue_txt=root+'train/overdue_train.txt'
bill_txt=root+'train/bill_detail_train.txt'
uinfo_txt=root+'train/user_info_train.txt'
loantime_txt=root+'train/loan_time_train.txt'
bill_npy=root+'train/bill.npy'
bank_npy=root+'train/bank_detail.npy'
browse_npy=root+'train/browse.npy'
bill_sorted=root+'train/bill_sorted.npy'
bank_sorted=root+'train/bank_sorted.npy'
browse_sorted=root+'train/browse_sorted.npy'

tbrowse_txt=root+'test/browse_history_test.txt'
tbank_txt=root+'test/bank_detail_test.txt'
toverdue_txt=root+'test/overdue_test.txt'
tuinfo_txt=root+'test/user_info_test.txt'
tloantime_txt=root+'test/loan_time_test.txt'
tbill_txt=root+'test/bill_detail_test.txt'
tbill_sorted=root+'test/bill_sorted.npy'
tbank_sorted=root+'test/bank_sorted.npy'
tbrowse_sorted=root+'test/browse_sorted.npy'

sub_path='/Users/ben/OneDrive/coding/projects/rong360/submissions/'


def load_bill(path=bill_sorted):
    bill=np.load(path)
    field_names=['f'+str(i+1) for i in range(15)]
    field_names[0]='id'
    field_names[14]='overdue'
    field_names[5]='credit limit'
    field_names[12]='balance'
    field_names[13]='cash balance'
    field_names[8]='number of payment'
    bill.dtype.names=field_names
    return bill

#bill=load_bill(bill_sorted)
#bank=np.load(bank_sorted)

# browse 游览历史
# 用户id,时间戳,浏览行为数据,浏览子行为编号 
# 34724,5926003545,172,     1 
# 34724,5926003545,163,     4 
# 67215,5932800403,163,     4 
#browse=np.load(browse_sorted)    


# missing value:
    # 1) fill in with averages
    # 2) delete it


#tbank=np.genfromtxt(tbank_txt,delimiter=',',dtype=[int,int,int,float,int])
#tbill=np.genfromtxt(tbill_txt,delimiter=',',
#                   dtype=[int,int,int, float,float,float,float,float,
#                          int, float,float,float,float,float, int])

    
###########################################
###########################################    
################## Features ###############
###########################################
        
def part_vectorize_bill(b):
    usr,idx,count=np.unique(b['id'],return_index=True,return_counts=True)
    n=len(usr)
    idx=np.append(idx,len(b))
    od=[b['overdue'][idx[i]:idx[i+1]].sum() for i in range(n)]
    od_rate=np.array(od)/count.astype(float)
    bal_mean=np.array([b['balance'][idx[i]:idx[i+1]].mean() for i in range(n)])
    cbal_mean=np.array([b['cash balance'][idx[i]:idx[i+1]].mean() for i in range(n)])
    cl_mean=np.array([b['credit limit'][idx[i]:idx[i+1]].mean() for i in range(n)])
    np_mean=np.array([b['number of payment'][idx[i]:idx[i+1]].mean() for i in range(n)])
    return usr,idx,np.array([od_rate,bal_mean,cbal_mean,cl_mean,np_mean]).T

#time_scale=10.**6
#time_scale=1

# KS score is stable, regardless of features nor classifiers, automize it
# compute all the possible vectorizations based on statistics: sum, max, 
# min, mean, median, and test them seperately against KS

def vectorize(b):
    # stats: sum,max,min,median,variance
    usr,idx,count=np.unique(b['id'],return_index=True,return_counts=True)
    n=len(usr)
    idx=np.append(idx,len(b))
    l=len(b.dtype)
#    vec=np.zeros((n,l))
    vec=np.zeros((n,2*l-1))
#    vec[:,0]=count
    vec[:,0]=0
    i=0
    for nm in b.dtype.names:
        for j in range(n):
            t=b[idx[j]:idx[j+1]][nm]
            #time rescaling: f2 is time for both bill and bank    
            if nm=='f2': 
                vec[j,i]=t.max()  
#                vec[j,i]=0
            elif not(nm=='id'):
                vec[j,i]=t.max()
#                vec[j,i]=0
#                vec[j,l+i-1]=b[idx[j]:idx[j+1]][nm].var()
        i+=1
    return usr,vec
    
def vecz_brs(br):
    u,idx=np.unique(br[:,0],return_index=True)
    n=len(u)
    idx=np.append(idx,len(br))
    #number of actions ranges from 1 to 216
    step=10
    #step from 10 to 1 does not seem to have obvious improvement
    v=np.zeros((n,1+216/step),dtype=float)
    bs=np.arange(1,218,step)
    for i in range(n):
        v[i,0]=br[idx[i]:idx[i+1]][:,1].max()
        v[i,1:]=np.histogram(br[idx[i]:idx[i+1]][:,2],bins=bs)[0]
    return u,v

n_train_usr=55596
    
def fill_vec(usr,vec,n=n_train_usr):
    # fill data of missing users
    m=vec.shape[1]
    v=np.zeros((n,m))
    for i in range(m):
        v[:,i]=vec[:,i].mean()
    if n==n_train_usr:
        v[usr-1]=vec
    else:
        v[usr-n_train_usr-1]=vec
    return v
    
def norm_time(t0):
    t=t0.copy()
    arg=(t>0.1)     # excluding exceptional values
    m=t[arg].mean()
    t[arg]-=m
    return t

overdue=np.genfromtxt(overdue_txt,delimiter=',',dtype=int)
od=overdue[:,1]

def get_od_color(tar=od):
    color=np.zeros(len(tar),dtype='|S5')
    color[tar==0]='y'
    color[tar==1]='red'
    return color
    

#u0,i0,v0=part_vectorize_bill()
#u_bl,v_bl=vectorize(bill)
#u_bk,v_bk=vectorize(bank)
#u_br,v_br=vecz_brs(browse)



#v1=fill_vec(u_bl,v_bl)
#v1[:,1]=norm_time(v1[:,1])
#v2=fill_vec(u_br,v_br)
v_info=np.genfromtxt(uinfo_txt,delimiter=',',dtype=int)[:,1:]
v_loantime=np.genfromtxt(loantime_txt,delimiter=',',dtype=float)[:,1:]
v_loantime=norm_time(v_loantime)
#v=np.hstack((v1,v2,v_info,v_loantime))
#arg=(v1[:,1]>0.1)
#v=np.hstack((v1[:,1:2],v_loantime))
#v_sq=np.hstack((v,v**2))
#v=np.hstack((v1[:,1:2],v_loantime,v2[:,4:6],v2[:,9:10],v2[:,11:13],
#             v2[:,14:15],v2[:,17:20]))

'''
def get_cross_sqr(vec=v):
    n=vec.shape[1]
    sq=np.zeros((vec.shape[0],n**2),dtype=float)
    for i in range(n):
        for j in range(n):
            sq[:,i*n+j]=vec[:,i]*vec[:,j]
    return sq
v_cq=np.hstack((v,get_cross_sqr()))
'''


n_test_usr=13899

'''
tbill=load_bill(tbill_sorted)
tbank=np.load(tbank_sorted)
tbrowse=np.load(tbrowse_sorted)
tu_bk,tv_bk=vectorize(tbank)
tu_bl,tv_bl=vectorize(tbill)
tu_br,tv_br=vecz_brs(tbrowse)
tv_info=np.genfromtxt(tuinfo_txt,delimiter=',',dtype=int)[:,1:]
tv_loantime=np.genfromtxt(tloantime_txt,delimiter=',',dtype=int)[:,1:]
tv1=fill_vec(tu_bl,tv_bl,n_test_usr)
tv2=fill_vec(tu_br,tv_br,n_test_usr)
#tv=np.hstack((tv1,tv2,tv_info))
#tv=np.hstack((tv1[:,1:2],tv_loantime,tv2[:,4:6],tv2[:,9:10],tv2[:,11:13],
#              tv2[:,14:15],tv2[:,17:20]))
tv=np.hstack((tv1[:,1:2],tv_loantime))
'''  


  
###########################################
###########################################    
################## Models #################
###########################################

#problems to be fixed
def xgb_clf(train_v,train_od,test_v,n_round=50):
    import xgboost as xgb
    dtrain=xgb.DMatrix(train_v,label=train_od)
    param_default = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    random_seed = 1225
    params_huang={
    'booster':'gbtree',
        'objective': 'binary:logistic',
        'early_stopping_rounds':100,
        'scale_pos_weight': 1500.0/13458.0,			#这个值是因为类别十分不平衡
    		'scale_pos_weight': 1,
            'eval_metric': 'auc',							
        'gamma':0.1,#0.2 is ok								#起始值也可以选其它比较小的值，在0.1到0.2之间就可以。这个参数后继也是要调整的。 
        'max_depth':8,	
        'lambda':550,
            'subsample':0.7,
            'colsample_bytree':0.3,
            'min_child_weight':2.5, 
            'eta': 0.005,
        'seed':random_seed,
        'nthread':7
        }
    bst = xgb.train(params_huang, dtrain, num_boost_round=n_round)
    dtest=xgb.DMatrix(test_v)
    return bst.predict(dtest)

#####################################

def random(test_v):
    import random
    a=[random.uniform(0,1) for i in range(len(test_v))]
    return np.array(a)
    
def baseline(train_v,train_od,test_v):
#    a=np.zeros(n,dtype=float)
    a=(test_v>0)
    return (a+0.1)*0.8
    

'''
def get_naive():
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()    
    
def get_knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import RadiusNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=200)   
'''
    
def predict(train_v,train_od,test_v,clf,pred_proba=True):
    clf.fit(train_v,train_od)
    if pred_proba:
        return clf.predict_proba(test_v)[:,1]
    else:
        return clf.predict(test_v)
        
def tree(train_v,train_od,test_v,d=9,m_nodes=None):
    from sklearn import tree
    t=tree.DecisionTreeClassifier(max_depth=d,max_leaf_nodes=m_nodes)
    return predict(train_v,train_od,test_v,clf=t)
    
def tree2(train_v,train_od,test_v):
    p=np.zeros(len(test_v))
    p+=0.1
    test_arg=(test_v[:,0]>0)
    train_arg=(train_v[:,0]>0)
    p0=tree(train_v[train_arg],train_od[train_arg],test_v[test_arg])
    p0+=0.2
    p0/=1.2
    p[test_arg]=p0
    return p
    
def forest(train_v,train_od,test_v,d=9):
    from sklearn.ensemble import RandomForestClassifier
    rf= RandomForestClassifier(max_depth=d)
    return predict(train_v,train_od,test_v,clf=rf)
    
def logi(train_v,train_od,test_v):
    from sklearn.linear_model import LogisticRegression
    l=LogisticRegression()   
    return predict(train_v,train_od,test_v,clf=l)
    
def svm(train_v,train_od,test_v):
    from sklearn import svm
    s=svm.SVC(probability=True,max_iter=50)
    def norm(v):
        v-=v.mean()
        v/=v.var()
    norm(train_v)
    norm(test_v)        
    return predict(train_v,train_od,test_v,clf=s,pred_proba=False)
    
##################### Boosting ######################
'''
def voting(train_v=v_bl,train_od=od,test_v=v_bl):
    from sklearn.ensemble import RandomForestClassifier 
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import VotingClassifier
    forest=RandomForestClassifier(max_depth=10)
    logi=LogisticRegression()
    knn=KNeighborsClassifier(n_neighbors=200)
    clf = VotingClassifier([('f',forest),('l',logi),('k',knn)],voting='soft')
#    clf = VotingClassifier([('f',forest),('l',logi)],voting='soft')
    clf.fit(train_v,train_od)
    return clf.predict_proba(test_v)[:,1]    
    
def bag(train_v=v_bl,train_od=od,test_v=v_bl):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(max_depth=10)
    knn= KNeighborsClassifier(n_neighbors=100)
    tree=tree.DecisionTreeClassifier(max_depth=10)
    clf=BaggingClassifier(forest,max_samples=0.5)
    clf.fit(train_v,train_od)
    return clf.predict_proba(test_v)[:,1]      

def ada(train_v=v_bl,train_od=od,test_v=v_bl):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import tree
    tree=tree.DecisionTreeClassifier(max_depth=10)
    clf=AdaBoostClassifier(base_estimator=tree)
    clf.fit(train_v,train_od)
    return clf.predict_proba(test_v)[:,1]  
'''    

#######################################

def single_split(train_v,train_od,test_v):
    t=train_v[:,0]
    mn=t.min()
    mx=t.max()
    step=(mx-mn)/10**3.
    b=np.arange(mn,mx+step,step)
    f0=distribution(t[train_od==0],bins=b)
    f1=distribution(t[train_od==1],bins=b)
#    return f0-f1
#    return np.abs(f0-f1).max()
    idx=np.abs(f0-f1).argmax()
#    return mn,idx,step
    s=mn+idx*step
#    return s
#    return (test_v[train_od==1,0]<s).mean()
#    return test_v[:,0]<(s+0.00001)
    return ((test_v[:,0]<(s+0.00001))+0.1)*0.8 # resolves some issue caused by numerical error

###########################################    
###########################################    
################## Evaluation #############
###########################################
    
def get_empirical_distr_old(p,n=1000):
    bins=np.arange(0.,1.+1./n,1./n)
    hist=np.histogram(p,bins,normed=True)[0]
    for i in range(n-1):
        hist[i+1]+=hist[i]
    return hist
    
def distribution(x,bins):
    hist=np.histogram(x,bins=bins)[0].astype(float)
    hist/=float(len(x))     #normalize to one
    n=len(bins)
    for i in range(n-2):
        hist[i+1]+=hist[i]
    return hist
    
def distribution2d(x,y,bins):
    hist=np.histogram2d(x,y,bins=bins)[0].astype(float)
    hist/=len(x)
#    return hist
    n0=len(bins[0])
    n1=len(bins[1])
    for i0 in range(n0-2):
        for i1 in range(n1-2):
            hist[i0,i1+1]+=hist[i0,i1]
    for i0 in range(n0-2):
        hist[i0+1,:]+=hist[i0,:]
    return hist
    
    
def get_empirical_distr(p,n=100):
    bins=np.arange(0.,1.+1./n,1./n)
    return distribution(p,bins)
    
def compute_KS(pred,true=od):
    x=get_empirical_distr(pred[true==0])
    y=get_empirical_distr(pred[true==1])   
#    return x,y
    return np.abs(x-y).max()
    
def cv_rs(vec,od,clf=tree,n=20):
    # random splitting
    from sklearn.model_selection import train_test_split
    import random
    ks=np.zeros((n,),dtype=float)
#    scale=10**6
    for i in range(n):
        v_train,v_test,od_train,od_test=train_test_split(vec,od,
             test_size=0.2,random_state=random.randrange(1000))
        ks[i]=compute_KS(clf(v_train,od_train,v_test),od_test)
#    return np.mean(ks),np.var(ks)
    return ks.min()

def cv_kf(vec,od,clf=tree,n=5):
    # shuffled kfold
    from sklearn.model_selection import KFold
    import random
    ks=[]
    idx=range(len(vec))
    for i in range(n):
        kf=KFold(n_splits=5,shuffle=True,random_state=random.randrange(1000))
        for train, test in kf.split(idx):
            ks.append(compute_KS(
                clf(vec[train],od[train],vec[test]),od[test]))
#                clf(vec[train],od[train],vec[test],d=1,m_nodes=None),od[test]))
#    return np.mean(ks),np.var(ks)        
    return np.min(ks)
    
def cv_split(vec,od,clf=tree,beta0=.0,beta1=0.8):
    n=len(vec)
    m1=int(n*beta1)
    m0=int(n*beta0)    
    return compute_KS(clf(v[m0:m1],od[m0:m1],v[m1:]),od[m1:])

def write_predictions(pred,file_name='201702'):
    usr_test=np.genfromtxt(root+'test/usersID_test.txt',dtype=int)
    f=open(sub_path+file_name+'.csv','w')
    f.writelines('userid,probability\n')
    for i in range(len(usr_test)):
        f.writelines(str(usr_test[i])+','+str(pred[i])+'\n')
    f.close()



