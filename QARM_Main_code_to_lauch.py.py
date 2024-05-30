
import pandas as pd 
import numpy as np 
import seaborn as sns
from numpy import matlib as mb
import matplotlib
import math
import copy
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.optimize import minimize
import pathlib as pl
from pathlib import *
import function
import momentum_adjust




df = pd.read_excel("Data_QAM2.xlsx")
df["Dates"] = pd.to_datetime(df["Dates"], format = "%Y/%m/%d")
df.set_index("Dates",inplace = True)

df_num_lines = np.size(df,0)
df_num_rows = np.size(df,1)

df_returns = np.divide(df.iloc[1:(df_num_lines),:],df.iloc[0:(df_num_lines-1),:])-1



in_sample = df.loc[:"2010-12-31"]
out_sample = df.loc["2011-01-31":]

in_sample_num_lines = np.size(in_sample,0)
in_sample_num_rows = np.size(in_sample,1)

df_returns = np.divide(df.iloc[1:(df_num_lines),:],df.iloc[0:(df_num_lines-1),:])-1

in_sample_returns = df_returns.loc[:"2010-12-31"]
out_sample_returns = df_returns.loc["2011-01-31":]

# ax = sns.heatmap(in_sample_returns.corr(), annot=True)
# plt.show()


#################################################
##################BENCHMARK#############
#define the Benchmark 
bench_in_sample = in_sample_returns[["World Equities","World Bonds"]]


bench = [0.5,0.5]
portfolio_bench = np.dot(bench_in_sample,bench)
#portfolio_bench_cum = portfolio_bench.cumsum()
portfolio_bench_cum= (1+portfolio_bench).cumprod()
plt.plot(portfolio_bench_cum)



portfolio_bench = pd.DataFrame(portfolio_bench)



#############################################
##SAA

df_asset = in_sample_returns

labels=list(df_asset)

x0 = np.array([0]*np.size(df_asset,1))+0.1

cons=({'type':'eq', 'fun': lambda x:sum(x)-1})
Bounds= [(0 , 1) for i in range(0,np.size(df_asset,1))]
    
res = minimize(function.ERC, x0, method='SLSQP', args= df_asset,bounds=Bounds,constraints=cons,options={'disp': True})
weights = res.x
weights = pd.DataFrame(weights,labels)
weights = np.round(weights,4)


portfolio_saa = np.dot(df_asset,weights)
#portfolio_saa_cum = portfolio_saa.cumsum()
portfolio_saa_cum = (1 + portfolio_saa).cumprod()

plt.plot(bench_in_sample.index, portfolio_saa_cum, "-b" , label="SAA")
plt.plot(bench_in_sample.index, portfolio_bench_cum, "-g", label = "Benchmark")
plt.legend()

portfolio_saa = pd.DataFrame(portfolio_saa)




##########################################
## OUT of sample 

bench_out_sample = out_sample_returns[["World Equities","World Bonds"]]
bench = [0.5,0.5]
portfolio_out_bench = np.dot(bench_out_sample,bench)
portfolio_out_bench_cum = (1 + portfolio_out_bench).cumprod()
#plt.plot(out_sample_returns.index,portfolio_out_bench_cum, "-g", label = "Benchmark")
#plt.legend()


df_out_asset = out_sample_returns
portfolio_out_saa = np.dot(df_out_asset,weights)
portfolio_out_saa_cum = (1 + portfolio_out_saa).cumprod()

# plt.plot(out_sample_returns.index,portfolio_out_saa_cum, "-b" , label="SAA")
# plt.plot(out_sample_returns.index,portfolio_out_bench_cum, "-g", label = "Benchmark")
# plt.legend()











###############################################
##################TAA########################
##################################################
filename="Data_QAM2.xlsx"
xls = pd.ExcelFile(filename)
carry = pd.read_excel(filename, 'Carry')
carry.rename(columns={'Unnamed: 0':'date'}, inplace=True)
carry['date'] = pd.to_datetime(carry['date'], format = "%Y/%m/%d")
carry.set_index('date', inplace=True)

for i in labels: 
    carry[i] = (carry[i]-carry[i].mean())/carry[i].std()

carry_in_sample = carry.loc[:"2010-12-31"]
carry_out_sample = carry.loc["2011-01-31":]

weights_value_in_sample = function.value_weights_full(carry_in_sample, in_sample_returns)

TAA_return_in_sample= []
for i in range(len(weights_value_in_sample)-1): 
    TAA_return_in_sample.append(np.dot(weights_value_in_sample.iloc[i,:],in_sample_returns.iloc[i-1,:]))
TAA_return_in_sample = pd.DataFrame(TAA_return_in_sample)

portfolio_in_TAA_cum = np.cumproduct(1 + TAA_return_in_sample)


plt.plot(in_sample_returns.index,portfolio_in_TAA_cum.values , "-g", label = "TAA")
plt.plot(in_sample_returns.index,portfolio_saa_cum, "-b" , label="SAA")
plt.plot(bench_in_sample.index, portfolio_bench_cum, "-v", label = "Benchmark")
plt.legend()





###############################################
#out sample 
#################################################


weights_value_out_sample= function.value_weights_full(carry_out_sample, df_returns)

TAA_return_out_sample= []
for i in range(len(weights_value_out_sample)): 
    TAA_return_out_sample.append(np.dot(weights_value_out_sample.iloc[i,:],out_sample_returns.iloc[i,:]))



TAA_return_out_sample = pd.DataFrame(TAA_return_out_sample)


portfolio_out_TAA_cum = np.cumproduct(1 + TAA_return_out_sample)


plt.plot(out_sample_returns.index,portfolio_out_TAA_cum.values , "-g", label = "TAA")
plt.plot(out_sample_returns.index,portfolio_out_saa_cum, "-b" , label="SAA")
plt.plot(out_sample_returns.index, portfolio_out_bench_cum, "-v", label = "Benchmark")
plt.legend()




### momentum 
#########################################
from momentum_adjust import momentum_adjust
momentum = function.momentum(df)
is_cov = np.cov(np.transpose(in_sample_returns))

full_weight_mom_final, returns_momentum = momentum_adjust(df_returns, momentum, is_cov)

is_weight_mom_final = full_weight_mom_final.loc["2001-08-31":"2010-12-31"]
oos_weight_mom_final = full_weight_mom_final.loc["2011-01-31":]

TAA_momentum_return_in_sample= []
for i in range(len(is_weight_mom_final)-1): 
    TAA_momentum_return_in_sample.append(np.dot(is_weight_mom_final.iloc[i,:],in_sample_returns.iloc[i,:]))
    
    
TAA_momentum_return_in_sample = pd.DataFrame(TAA_momentum_return_in_sample)
portfolio_in_TAA_momentum_cum  = np.cumproduct(1+TAA_momentum_return_in_sample)

index_graph_out = out_sample_returns.iloc[1:].index.tolist()
index_graph_in = in_sample_returns.iloc[1:].index.tolist()

plt.plot(index_graph_in[9:], portfolio_in_TAA_momentum_cum.values, "-g", label = "TAA momentum")
plt.plot(index_graph_in, portfolio_saa_cum[1:], "-b" , label="SAA")
plt.legend()

########################################################
#####################################################


################################################################
###############################################################
##out_sample 

# from momentum_adjust import momentum_adjust_full


TAA_momentum_return_out_sample= []
for i in range(len(oos_weight_mom_final)-1): 
    TAA_momentum_return_out_sample.append(np.dot(oos_weight_mom_final.iloc[i,:],out_sample_returns.iloc[i,:]))
    
    
TAA_momentum_return_out_sample = pd.DataFrame(TAA_momentum_return_out_sample)
portfolio_out_TAA_momentum_cum  = np.cumproduct(1+TAA_momentum_return_out_sample)


plt.plot(portfolio_out_TAA_momentum_cum, "-g", label = "TAA momentum")
plt.plot(portfolio_out_saa_cum, "-b" , label="SAA")
plt.legend()







####################################################################################
# Question 2.2
###################################################################################


filename="Data_QAM2.xlsx"
xls = pd.ExcelFile(filename)
vix = pd.read_excel(xls, 'VIX')
vix["Dates"] = pd.to_datetime(vix["Dates"], format = "%Y/%m/%d")
vix.set_index("Dates",inplace = True)



vix = (vix -vix.mean())/vix.std()


vix_in_sample = vix.loc[:"2010-12-31"]
vix_out_sample = vix.loc["2011-01-31":]

vix_in_sample_value = vix_in_sample.iloc[:,:]
vix_dummy_value= []
for i in range(len(vix_in_sample_value)):
    if vix_in_sample_value.iloc[i,0] > 2 :
        vix_dummy_value.append(2)
    elif vix_in_sample_value.iloc[i,0]>0:
        vix_dummy_value.append(1)
    elif vix_in_sample_value.iloc[i,0]<-1:
        vix_dummy_value.append(-2)
    else: 
        vix_dummy_value.append(-1)

vix_out_sample_value = vix_out_sample.iloc[:,:]
vix_dummy_value_out= []
for i in range(len(vix_out_sample_value)):
    if vix_out_sample_value.iloc[i,0] > 2 :
        vix_dummy_value_out.append(2)
    elif vix_out_sample_value.iloc[i,0]>0:
        vix_dummy_value_out.append(1)
    elif vix_out_sample_value.iloc[i,0]<-1:
        vix_dummy_value_out.append(-2)
    else: 
        vix_dummy_value_out.append(-1)
        

vix_dummy_value = pd.DataFrame(vix_dummy_value)
vix_dummy_value_out = pd.DataFrame(vix_dummy_value_out)

Y = TAA_return_in_sample
X =vix_dummy_value.iloc[:len(vix_dummy_value)-1,:]

plt.scatter(X, Y)


############################################################



Y = TAA_momentum_return_in_sample.iloc[1:,:]
X = vix_dummy_value.iloc[11:len(vix_dummy_value)-1,:]

plt.scatter(X, Y)    


high_vol = [0.35,0.75]
low_vol =  [0.75,0.35]
        
factor_portfolio = pd.DataFrame (index = weights_value_in_sample.index , columns = labels)
        
        

###################################################################
##Target portfolio 

weights_value_in_sample = weights_value_in_sample.loc["2001-08-31":"2010-12-31"]
target_portfolio = pd.DataFrame (index = weights_value_in_sample.index , columns = labels)

for i in range (len(weights_value_in_sample)): 
    for n in range(len(weights)): 
        if vix_dummy_value.iloc[i,0]==2 or vix_dummy_value.iloc[i,0]==-2:
            target_portfolio.iloc[i,n] = 0.375*weights_value_in_sample.iloc[i,n] + 0.375*is_weight_mom_final.iloc[i,n] +0.35*weights.iloc[n,0]
        if vix_dummy_value.iloc[i,0]==1 or vix_dummy_value.iloc[i,0]==-1:
            target_portfolio.iloc[i,n] = 0.175*weights_value_in_sample.iloc[i,n] + 0.175*is_weight_mom_final.iloc[i,n] +0.75*weights.iloc[n,0]

target_portfolio_return_in= []
for i in range(len(target_portfolio)-1): 
    target_portfolio_return_in.append(np.dot(target_portfolio.iloc[i,:],in_sample_returns.iloc[i,:]))
    
#weights_value_out_sample = weights_value_in_sample.loc["2001-08-31":"2010-12-31"]


target_portfolio_out = pd.DataFrame (index = weights_value_out_sample.index , columns = labels)

for i in range (len(weights_value_out_sample)): 
    for n in range(len(weights)): 
        if vix_dummy_value_out.iloc[i,0]==2 or vix_dummy_value_out.iloc[i,0]==-2:
            target_portfolio_out.iloc[i,n] = 0.375*weights_value_out_sample.iloc[i,n] + 0.375*oos_weight_mom_final.iloc[i,n] +0.35*weights.iloc[n,0]
        if vix_dummy_value_out.iloc[i,0]==1 or vix_dummy_value_out.iloc[i,0]==-1:
            target_portfolio_out.iloc[i,n] = 0.175*weights_value_out_sample.iloc[i,n] + 0.175*oos_weight_mom_final.iloc[i,n] +0.75*weights.iloc[n,0]

target_portfolio_return_out= []
for i in range(len(target_portfolio_out)-1): 
    target_portfolio_return_out.append(np.dot(target_portfolio_out.iloc[i,:],out_sample_returns.iloc[i,:]))


target_portfolio_return_in = pd.DataFrame(target_portfolio_return_in)
target_portfolio_return_out = pd.DataFrame(target_portfolio_return_out)

target_portfolio_return_in_cum  = np.cumproduct(1+target_portfolio_return_in)
target_portfolio_return_out_cum  = np.cumproduct(1+target_portfolio_return_out)
index_graph_out = out_sample_returns.iloc[1:].index.tolist()

portfolio_out_bench_cum = portfolio_out_bench_cum[1:]

plt.plot(index_graph_out, target_portfolio_return_out_cum.values, "-b" , label="Target")
plt.plot(index_graph_out, portfolio_out_bench_cum, "-r" , label="Benchmark")
plt.legend()

###INFOS
mean = target_portfolio_return_out.mean(axis = 0)
std = np.array(target_portfolio_return_out).std()*np.power(12, 0.5)
aaa = function.drawdown(target_portfolio_return_out)

count_pos = pd.DataFrame(target_portfolio_return_out)
positive_vals = count_pos[count_pos.iloc[:,0] > 0]
aa = len(positive_vals)/len(count_pos)


portfolio_bench_out = np.dot(bench_in_sample,bench)
#portfolio_bench_cum = portfolio_bench.cumsum()
portfolio_bench_cum_out= (1+portfolio_bench_out).cumprod()
plt.plot(portfolio_bench_cum_out)

plt.plot(target_portfolio_return_out_cum, "-b" , label="Target")
plt.plot(portfolio_bench_cum_out, "-g", label = "Benchmark")
plt.legend()

##########################
# ATTRIBUTION&ALLOCATION #
##########################
equities = ["World Equities"]
bonds = ["World Bonds","US Investment Grade","US High Yield"]
commodities = ["Gold","Energy","Copper"]

target_weightss = copy.copy(target_portfolio_out)

from momentum_adjust import TE_ex_ante


Asset_allocation = []
Security_selection = []
track_error = []
adj_weight_list = []
for i in range(0, len(target_weightss)):
    target_weights = pd.DataFrame(target_weightss.iloc[i,:])
    
    
    
    temp = target_weights.transpose()
    temp = temp.drop('US Investment Grade', 1).transpose()
    temp_us = pd.DataFrame(target_weights.loc[('US Investment Grade')]).transpose()
    # cons=({'type':'eq'})
    
    def ConstraintFullInv(x):
        return sum(x)-1
    cons=({'type':'eq', 'fun':ConstraintFullInv})
    f_target_weights = pd.concat([temp, temp_us], axis = 0)
    
    
    Sigma=np.cov(np.transpose(in_sample_returns))
    
    #initalize with evenly distributed among the 6 assets
    poids_test = 1/6
    x0  = []
    for i in range (0,6):
        x0.append(poids_test)
    
    res_TO_control = minimize(TE_ex_ante, x0, args = (f_target_weights, Sigma), method='SLSQP',options={'disp': False})
    track_error.append(res_TO_control.fun)
    ajd_weights = res_TO_control.x
    ajd_weights = np.append(ajd_weights, 0)
    df_adj_weights = pd.DataFrame(index = f_target_weights.index, data = ajd_weights)
    adj_weight_list.append(ajd_weights)
    ############
    equities = ["World Equities"]
    bonds = ["World Bonds","US Investment Grade","US High Yield"]
    commodities = ["Gold","Energy","Copper"]
    
    portfolio_weight = copy.copy(df_adj_weights)
    target_weight = copy.copy(target_weights)
    realized_returns = copy.copy(out_sample_returns)
    
    un = portfolio_weight
    un.columns = ['Ptf Weight']
    
    und = target_weight
    und.columns = ['Target Weight']
    
    undt = realized_returns
    undt = pd.DataFrame(realized_returns.iloc[0,:])*12
    undt.columns = ['Realized returns']
    
    final_3 = pd.concat([un, und], axis = 1)
    final_3 = pd.concat([final_3, undt], axis = 1)
    
    # final_3['Sector'] = ['Equities', 'Bonds', 'Bonds', 'Commodities', 'Commodities', 'Commodities', 'Bonds']
    final_3['Portofolio'] = final_3['Ptf Weight']*final_3['Realized returns']
    final_3['Benchmark'] = final_3['Target Weight']*final_3['Realized returns']
    
    final_bonds = pd.DataFrame(final_3.loc[bonds].sum().transpose(), columns = ['Bonds'])
    final_como = pd.DataFrame(final_3.loc[commodities].sum().transpose(), columns = ['Commodities'])
    final_eq = pd.DataFrame(final_3.loc[equities].sum().transpose(), columns = ['equities'])
    
    final_4 = pd.concat([final_bonds, final_como], axis = 1)
    final_4 = pd.concat([final_4, final_eq], axis = 1).transpose()
    
    R = (final_4['Ptf Weight']*final_4['Portofolio']).sum()
    B = (final_4['Target Weight']*final_4['Benchmark']).sum()
    R_S = (final_4['Target Weight']*final_4['Portofolio']).sum()
    B_S = (final_4['Ptf Weight']*final_4['Benchmark']).sum()
    
    Asset_allocation.append(R_S - B)
    Security_selection.append(B_S - B)

df_allocation = pd.DataFrame(data = [Asset_allocation, Security_selection], index = [['Asset Allocation', 'Security Selection']]).mean(axis= 1)


mean_track = np.array(track_error).mean()
plt.plot(track_error[1:])
plt.hlines(mean_track, xmin = 0, xmax = 120, color = 'red')

real_weights = pd.DataFrame(adj_weight_list).drop(columns = [6])
out_sample_no_US = out_sample_returns.drop(columns = ['US Investment Grade'])

graph_real_ptf_perf, a, b = function.performance_plot(real_weights, out_sample_no_US, 1)


plt.plot(index_graph_out, target_portfolio_return_out_cum.values, "-b" , label="Target ptf")
plt.plot(graph_real_ptf_perf, '-r', label = 'Real Ptf')
plt.legend()


















