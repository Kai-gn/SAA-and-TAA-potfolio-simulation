# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:44:21 2021

"""


import pandas as pd 
import numpy as np 

def drawdown (portfolio_saa):
    portfolio_saa2 = portfolio_saa.copy()
    #on crée le graphe de returns
    for i in range(0, np.size(portfolio_saa, 0)):
        if i ==0:
            portfolio_saa2.iloc[i,0] = 1
        else:
            portfolio_saa2.iloc[i,0] = portfolio_saa2.iloc[i-1,0]*(1+portfolio_saa2.iloc[i,0])
    #max drawdown
    max_dd = 1
    portfolio_saa2["drawdown"]=0
    portfolio_saa2["max_dd"]=0
    print('test 3')

    for i in range(0, np.size(portfolio_saa2, 0)):
        if i == 0:
            max_dd = 1
            portfolio_saa2.iloc[i,2]= max_dd
            portfolio_saa2.iloc[i,1] = (portfolio_saa2.iloc[i,0]/max_dd)-1
        
        else:
            if portfolio_saa2.iloc[i-1,0] > max_dd:
                max_dd = portfolio_saa2.iloc[i-1,0]
                portfolio_saa2.iloc[i,2]= max_dd
            portfolio_saa2.iloc[i,2]= max_dd
            portfolio_saa2.iloc[i,1] = (portfolio_saa2.iloc[i,0]/max_dd)-1

    drawdown = portfolio_saa2.copy()
    drawdown.drop(columns=[0], inplace = True)
    portfolio_saa2.drop(columns=['drawdown', 'max_dd'], inplace = True)
    drawdown_min =drawdown.min(axis = 0)
    return drawdown_min
    
    
    
def MCR_calc(alloc,Returns):
    ptf=np.multiply(Returns,alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    Sigma=np.cov(np.transpose(Returns))
    MCR=np.matmul(Sigma,np.transpose(alloc))/vol_ptf;
    return MCR

def ERC(alloc,Returns):
    ptf=np.multiply(Returns,alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    indiv_ERC=alloc*MCR_calc(alloc,Returns);
    criterion=np.power(indiv_ERC-vol_ptf/len(alloc),2)
    criterion=np.sum(criterion)*1000000
    return criterion

def hit_ratio(x): 
    n = 0 
    for i in x: 
        if i>=0: 
            n+=1 
    return np.around(n/np.size(x,0)*100,3)




from dateutil.relativedelta import relativedelta
def momentum(df): 
    mlt_ret = df.pct_change().resample('M').agg(lambda x:(x+1).prod()-1)
    past_11 = (mlt_ret +1).rolling(11).apply(np.prod)-1

    return past_11



def momentum_adjust(sample_returns, init_momentum_weight, is_cov):
    
    #médiane des carry
    init_momentum_weight_median = init_momentum_weight.median(axis=1)
    init_momentum_weight_median = pd.DataFrame(init_momentum_weight_median)
    
    init_momentum_weight_temp = init_momentum_weight.copy()
    #on crée carry_value qui contient les poids
    for i in range(0,np.size(init_momentum_weight,0)):
        for j in range(0, np.size(init_momentum_weight,1)):
            init_momentum_weight_temp.iloc[i,j] = (init_momentum_weight.iloc[i,j]-init_momentum_weight_median.iloc[i,0])

    init_momentum_weight_sig = init_momentum_weight_temp.copy()
    
            #poids = +1 si positif ou -1 si negatif
    for i in range(0,np.size(init_momentum_weight,0)):
        for j in range(0, np.size(init_momentum_weight,1)):
            if init_momentum_weight_temp.iloc[i,j] >0:
                init_momentum_weight_sig.iloc[i,j] = 1
            if init_momentum_weight_temp.iloc[i,j] <0:
                init_momentum_weight_sig.iloc[i,j] = -1
                #pas sûr des 0
            if init_momentum_weight_temp.iloc[i,j]==0:
                init_momentum_weight_sig.iloc[i,j] = 0
    
    init_momentum_weight_sig["sum_long"]= 0
    init_momentum_weight_sig["sum_short"]= 0
    init_momentum_weight_sig["sum_total"]= 7
    for i in range(0, np.size(init_momentum_weight, 0)):
        for j in range(0, np.size(init_momentum_weight_sig, 1)):
            if init_momentum_weight_sig.iloc[i,j] == 1 :
                init_momentum_weight_sig.iloc[i,7]+=1
            if init_momentum_weight_sig.iloc[i,j] == -1:
                init_momentum_weight_sig.iloc[i,8]-=1
                     
    weight_mom = init_momentum_weight_temp.copy()
    
    for i in range(0, np.size(weight_mom, 0)):
        for j in range(0, np.size(weight_mom, 1)): 
            if init_momentum_weight_sig.iloc[i,j]==1:
                weight_mom.iloc[i,j]=init_momentum_weight_sig.iloc[i,j]-(init_momentum_weight_sig.iloc[i,7]/init_momentum_weight_sig.iloc[i,9])
            if init_momentum_weight_sig.iloc[i,j]==-1:
                weight_mom.iloc[i,j]=init_momentum_weight_sig.iloc[i,j]-(init_momentum_weight_sig.iloc[i,8]/init_momentum_weight_sig.iloc[i,9])    
           
    #----- ajustement -------
    weight_mom_final = weight_mom.copy()
    
 
    for i in range(0, np.size(weight_mom, 0)):
        for j in range(0, np.size(weight_mom, 1)): 
            if i <=121:
                vol_mom= np.sqrt(np.dot(weight_mom.iloc[i,:],np.dot(is_cov,weight_mom.iloc[i,:].T)))
                adjust = 0.02/vol_mom
                weight_mom_final.iloc[i,j]=weight_mom.iloc[i,j]*adjust       
            if i>121:
                vol_mom = np.sqrt(np.dot(weight_mom.iloc[i,:],np.dot(np.cov(np.transpose(sample_returns.iloc[i-50:i,:])),weight_mom.iloc[i,:].T)))    
                adjust = 0.02/vol_mom
                weight_mom_final.iloc[i,j]=weight_mom.iloc[i,j]*adjust  
    # test6 = np.sqrt(np.dot(weight_mom_final.iloc[150,:],np.dot(np.cov(np.transpose(sample_returns.iloc[150-50:150,:])),weight_mom_final.iloc[150,:].T)))                            

    #returns
    
    #marche pas jsp pk
    #is_ret_mom = in_sample_returns*is_weight_mom_final.loc['2000-11-30':'2010-12-30']
    #is_ret_mom_test = in_sample_returns.mul(is_weight_mom_final)
    
    ret_mom=sample_returns.copy()
    
    for i in range(0, np.size(sample_returns, 0)):
        for j in range(0, np.size(sample_returns, 1)): 
            ret_mom.iloc[i,j]=weight_mom_final.iloc[i+1,j]*sample_returns.iloc[i,j]
    ret_mom = ret_mom.sum(axis = 1)
    ret_mom = pd.DataFrame(ret_mom)
    
    return weight_mom_final, ret_mom




def value_weights_full(carry,full_sample_returns):
#médiane des carry
    carry_mean = carry.mean (axis = 1)
    carry_mean = pd.DataFrame(carry_mean)
    carry_median = carry.median(axis=1)
    carry_median = pd.DataFrame(carry_median)
    carry_value = carry.copy()
    
    
    #on crée carry_value qui contient les poids
    for i in range(0,np.size(carry_value,0)):
        for j in range(0, np.size(carry,1)):
            carry_value.iloc[i,j] = (carry.iloc[i,j]-carry_median.iloc[i,0])

    carry_value_sig = carry_value.copy()
    
            #poids = +1 si positif ou -1 si negatif
    for i in range(0,np.size(carry,0)):
        for j in range(0, np.size(carry,1)):
            if carry_value.iloc[i,j] >0:
                carry_value_sig.iloc[i,j] = 1
            if carry_value.iloc[i,j] <0:
                carry_value_sig.iloc[i,j] = -1
                #pas sûr des 0
            if carry_value.iloc[i,j]==0:
                carry_value_sig.iloc[i,j] = 0
    
    carry_value_sig["sum_long"]= 0
    carry_value_sig["sum_short"]= 0
    carry_value_sig["sum_total"]= 7
    for i in range(0, np.size(carry, 0)):
        for j in range(0, np.size(carry_value_sig, 1)):
            if carry_value_sig.iloc[i,j] == 1 :
                carry_value_sig.iloc[i,7]+=1
            if carry_value_sig.iloc[i,j] == -1:
                carry_value_sig.iloc[i,8]-=1
            
    weight_value = carry_value.copy()
    for i in range(0, np.size(weight_value, 0)):
        for j in range(0, np.size(weight_value, 1)): 
            if carry_value_sig.iloc[i,j]==1:
                weight_value.iloc[i,j]=carry_value_sig.iloc[i,j]-(carry_value_sig.iloc[i,7]/carry_value_sig.iloc[i,9])
            if carry_value_sig.iloc[i,j]==-1:
                weight_value.iloc[i,j]=carry_value_sig.iloc[i,j]-(carry_value_sig.iloc[i,8]/carry_value_sig.iloc[i,9])
      
        
    is_cov = np.cov(np.transpose(full_sample_returns))
    weight_value_final = weight_value.copy()
    

    for i in range(0, np.size(weight_value, 0)):
        for j in range(0, np.size(weight_value, 1)): 
            if i<=121:
                vol= np.sqrt(np.dot(weight_value.iloc[i,:],np.dot(is_cov,weight_value.iloc[i,:].T)))
                adjust = 0.02/vol
                weight_value_final.iloc[i,j]=weight_value.iloc[i,j]*adjust
            
            if i>121:
                #vol= np.sqrt(np.dot(weight_value.iloc[i,:],np.dot(is_cov,weight_value.iloc[i,:].T)))
                vol= np.sqrt(np.dot(weight_value.iloc[i,:],np.dot(np.cov(np.transpose(full_sample_returns.iloc[i-50:i,:])),weight_value.iloc[i,:].T)))
                adjust = 0.02/vol
                weight_value_final.iloc[i,j]=weight_value.iloc[i,j]*adjust      
            
            
    return weight_value_final

def get_returns_from_ptf(weights, in_sample_returns, spt = False): 
        
    ptf = np.multiply(in_sample_returns.iloc[:,:], weights)
    # print(ptf)
    ptf_returns = np.sum(ptf ,1)
    # print(ptf_returns)
    mean_returns = np.mean(ptf_returns)*12
    # print(mean_returns)
    vol_returns = np.std(ptf_returns)*np.power(12,.5)
    sp = mean_returns/vol_returns
    # print(vol_returns)
    
    if spt == True:
        return { 'Mean Returns' : mean_returns, 'Vol Returns' : vol_returns, 'Ptf Returns' : ptf_returns, 'Sharpe Ratio':sp} #modif
    else:
        return { 'Mean Returns' : mean_returns, 'Vol Returns' : vol_returns, 'Ptf Returns' : ptf_returns}

def TE_ex_ante(x, target, Sigma):
 
    target_opt=target.values
    x_temp=target_opt*0;
    x=pd.DataFrame(x);
    x_temp[0:6,:]=x.values;
    diff_alloc=np.transpose(x_temp-target_opt);
    temp1=diff_alloc.dot(Sigma)
    temp2=np.transpose(diff_alloc);
    temp3=np.dot(temp1,temp2);
    output=np.power(temp3.T,.5)*np.power(252,.5)
    # print('x temp = ', x_temp, '\n')
    # print('diff alloc : ', diff_alloc)
    # print(output.item())
    return output.item()

def performance_plot(weights, returns, starting_value):
    stats_ptf = get_returns_from_ptf(weights, returns)
    ptf_returns = stats_ptf['Ptf Returns']
    total_return = 0
    weekly_returns = []
    weekly_returns.append(ptf_returns.iloc[0]) #needed for the return value but not used for the cumul
#because it's the sarting day
    re_oss_week_cumul = ptf_returns
    re_oss_week_cumul.iloc[0] = starting_value
    
    for i in range (1, np.size(returns,0)):
        this_week = ptf_returns.iloc[i]
        re_oss_week_cumul.iloc[i] = re_oss_week_cumul[i-1]*(1+this_week)
        weekly_returns.append(this_week)
    
    weekly_returns = np.array(weekly_returns)
    total_return = (re_oss_week_cumul.iloc[np.size(returns,0) -1]/re_oss_week_cumul.iloc[0]) - 1
    
    return re_oss_week_cumul, total_return, weekly_returns

