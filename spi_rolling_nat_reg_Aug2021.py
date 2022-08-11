import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats 
from scipy.stats import norm
import seaborn as sns
import math

from climate_indices import compute, indices

#### AUTHOR: LEGD
#### UPDATED: 14 JAN 2022


#DOCUMENTATION
# Units: MM3
# This code plots the Natural SDI of the gauges and correlated the natural SDI with the
# regulated streamflow. It stores the csv files and saves the 6m and 12m SDI in the folder
#=========================================================================================

#=========================================================================================
# Read csv files
#=========================================================================================

folder = '/Users/lauraelisa/Desktop/RioGrande/SPI/data'
results = '/Users/lauraelisa/Desktop/RioGrande/SPI/results'
file_name_1 = 'Pecos_monthly_natural_streamflow'
file_name_2 = 'Pecos_monthly_regulated_streamflow'
# 
# Alamo_monthly_natural_streamflow
# Salado_monthly_natural_streamflow
# Escondido_monthly_natural_streamflow
# Pinto_monthly_natural_streamflow
# SanFelipe_monthly_natural_streamflow
# LV_SD_SR_monthly_natural_streamflow


#For Natural
df = pd.read_csv('{}/{}.csv'.format(folder,file_name_1),skiprows=6, index_col=0,encoding='ISO-8859-1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = pd.DataFrame(df.values.ravel(), columns=['one_month']) 
df.set_index(pd.date_range(start='1/1/1900', end='12/31/2010', freq='MS'), inplace=True) 


#For regulatetd 
df_reg = pd.read_csv('{}/{}.csv'.format(folder,file_name_2),skiprows=6, index_col=0,encoding='ISO-8859-1')
df_reg = df_reg.loc[:, ~df_reg.columns.str.contains('^Unnamed')]
df_reg = pd.DataFrame(df_reg.values.ravel(), columns=['aggregated_reg_streamflow']) 
df_reg.set_index(pd.date_range(start='1/1/1900', end='12/31/2010', freq='MS'), inplace=True)

# plt.plot(df)
# plt.show()


#=========================================================================================
# Group monthly data into preferred time window e.g. 12 months
#=========================================================================================

# Sum of monthly into yearly values
# df = df.groupby([lambda x: x.year]).sum()


#rolling window 12 months
# df = df.Streamflow.rolling(12).sum() #change 12 to any amount of months that needs to be sumed


df['three_month'] = df.one_month.rolling(3).sum()
df['six_month'] = df.one_month.rolling(6).sum()
df['twelve_month'] = df.one_month.rolling(12).sum()
df['ten_years'] = df.one_month.rolling(120).sum()
# df['fifteen_years'] = df.one_month.rolling(180).sum()
# df['twenty_years'] = df.one_month.rolling(240).sum()


df_nat_3m = df['three_month'] = df.one_month.rolling(3).sum().rename('3m_nat_streamflow')
df_nat_6m = df['six_month'] = df.one_month.rolling(6).sum().rename('6m_nat_streamflow')
df_nat_12m = df['twelve_month'] = df.one_month.rolling(12).sum().rename('12m_nat_streamflow')
df_nat_10y = df['ten_years'] = df.one_month.rolling(120).sum().rename('10y_nat_streamflow')

df_reg_3m = df_reg['three_month'] = df_reg.aggregated_reg_streamflow.rolling(3).sum()
df_reg_6m = df_reg['six_month'] = df_reg.aggregated_reg_streamflow.rolling(6).sum()
df_reg_12m = df_reg['twelve_month'] = df_reg.aggregated_reg_streamflow.rolling(12).sum()
df_reg_10y = df_reg['ten_years'] = df_reg.aggregated_reg_streamflow.rolling(120).sum()
# df['fifteen_years'] = df.one_month.rolling(180).sum()
# df['twenty_years'] = df.one_month.rolling(240).sum()


#========================================================================================
# Normalize natural monthly  streamflow and plot
# Perform  Kolmorov statistics to check if it is normal.
#=========================================================================================

df_normalized = np.log(df).dropna()
df_normalized = df_normalized[df_normalized.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

columns = list(df_normalized)

for months in columns:

    y = df_normalized[months]
    N = len(y)
    m = np.mean(y)
    s = np.std(y)
    g = stats.skew(y)
    
    print (months)
    print('Mean = %f' % m)
    print('Std. Dev. = %f' % s)
    print('Skew Coef. = %f' % g) # no skew function in numpy
    print(stats.kstest(df_normalized[months], 'norm', args=(df_normalized[months].mean(), df_normalized[months].std()))) 
    print('--------------------')

    # sns.distplot(df_normalized[months], bins=12, kde=False, fit=stats.norm)

# #=========================================================================================
# # Plot normal distributions
# #=========================================================================================
   
# plt.xlabel('Log-Normal Streamflow Distribution')
# plt.ylabel('Probability')
# plt.legend(['1-Month','3-Month','6-Month','12-Month'], loc='upper left', fontsize='small')

# one = mpatches.Patch(color='lightsteelblue', label='1-Month')
# three = mpatches.Patch(color='bisque', label='3-Month')
# six = mpatches.Patch(color='lightgreen', label='6-Month')
# twelve = mpatches.Patch(color='lightcoral', label='12-Month')
# ten = mpatches.Patch(color='purple', label='10-Years')

# plt.legend(handles=[one,three,six,twelve,ten], title='Timestep', loc= 'upper left', fontsize = 8)


# # plt.xlabel('Log-Normal Streamflow Distribution')
# # plt.ylabel('Probability')
# # plt.legend(['1-year','10-year','20-year'], loc='upper left', fontsize='small')

# # one_y = mpatches.Patch(color='lightsteelblue', label='1-Year')
# # ten_y = mpatches.Patch(color='bisque', label='10-Years')
# # twenty_y = mpatches.Patch(color='lightgreen', label='20-Years')
# # plt.legend(handles=[one_y,ten_y,twenty_y], title='Timestep', loc= 'upper left', fontsize = 8)


# # SAVE AS PDF
# # plt.savefig('lognormal_distribution.pdf')
# plt.show()


#=========================================================================================
# Cummulative probability and random SPI 
#=========================================================================================

# Sort the streamflow values
df_sorted = df_normalized.apply(lambda x: x.sort_values().values) #independent from index

# calculate the proportional values of samples
p = 1. * np.arange(len(df_normalized)) / (len(df_normalized) - 1)
lenght = len(df_normalized)

#generate random normal distribution for SPI
spi_data = np.random.randn(lenght) #down is the list of SPI values was set as 1308 for all monthly valyes
spi_sorted_data = np.sort(spi_data) ##sort the data from 

# # Plot cumulative probability 
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.plot(df_sorted, p) # plot each month separately use -->     ax1.plot(df_sorted[months], p)
# ax1.set_xlabel('Streamflow (normalized)')
# ax1.set_ylabel('Cummulative Probability')


# ax2 = fig.add_subplot(122)
# ax2.plot(spi_sorted_data, p)
# ax2.set_xlabel('Spi index')
# ax2.set_ylabel('')


# for months in columns:

# #     # plot the sorted data:
# #     fig = plt.figure()
# #     ax1 = fig.add_subplot(121)
# #     ax1.plot(df_sorted[months], p)
# #     ax1.set_xlabel('Streamflow (normalized)')
# #     ax1.set_ylabel('Cummulative Probability')
    
    
#     # ax2 = fig.add_subplot(122)
#     # ax2.plot(spi_sorted_data, p)
#     # ax2.set_xlabel('Spi index')
#     # ax2.set_ylabel('')
    
#     # plt.show() #activate when plot each month separately
    

#=========================================================================================
# # Create EQUIPROBABILITIES df with normalized sorted streamflow and sorted SPI
# #========================================================================================
    
spi_sorted_data = np.around(spi_sorted_data, decimals = 2) ##round the SPI index to only one integer, maintain after plotting
spi = pd.DataFrame(spi_sorted_data)
spi.columns = ['spi']
# spi_sorted_data = pd.Series(spi_sorted_data)


#Create specific df for every time step (ts) ( 1- month, 3-month etc) and sort (needs to be individual sorting bc each column is independent) 
# #create equipotentials: Join time series with spi values and then sort the df by index
#EQUIPOTENTIAL 1 MONTH
ts1 =  pd.DataFrame(df_normalized.one_month.sort_values())
eq_1 = ts1.join(spi.set_index(ts1.index)).sort_index()
eq_1['index']=eq_1.index
eq_1['year']=eq_1.index.to_series().dt.year
eq_1['spi_values'] = np.around(eq_1['spi'], decimals =1)

ts3 = pd.DataFrame(df_normalized.three_month.sort_values())
eq_3 = ts3.join(spi.set_index(ts3.index)).sort_index()
eq_3['index']=eq_3.index
eq_3['year']=eq_3.index.to_series().dt.year
eq_3['spi_values'] = np.around(eq_3['spi'], decimals =1)

ts6 = pd.DataFrame(df_normalized.six_month.sort_values())
eq_6 = ts6.join(spi.set_index(ts6.index)).sort_index()
eq_6['index']=eq_6.index
eq_6['year']=eq_6.index.to_series().dt.year
eq_6['spi_values'] = np.around(eq_6['spi'], decimals =1)

ts12 = pd.DataFrame(df_normalized.twelve_month.sort_values())
eq_12 = ts12.join(spi.set_index(ts12.index)).sort_index()
eq_12['index']=eq_12.index
eq_12['year']=eq_12.index.to_series().dt.year
eq_12['spi_values'] = np.around(eq_12['spi'], decimals =1)

ts120 = pd.DataFrame(df_normalized.ten_years.sort_values())
eq_120 = ts120.join(spi.set_index(ts120.index)).sort_index()
eq_120['index']=eq_120.index
eq_120['year']=eq_120.index.to_series().dt.year
eq_120['spi_values'] = np.around(eq_120['spi'], decimals =1)

# ts180 = pd.DataFrame(df_normalized.fifteen_years.sort_values())
# eq_180 = ts180.join(spi.set_index(ts180.index)).sort_index()
# eq_180['index']=eq_180.index
# eq_180['year']=eq_180.index.to_series().dt.year
# eq_180['spi_values'] = np.around(eq_180['spi'], decimals =1)

# ts240 = pd.DataFrame(df_normalized.twenty_years.sort_values())
# eq_240 = ts240.join(spi.set_index(ts240.index)).sort_index()
# eq_240['index']=eq_240.index
# eq_240['year']=eq_240.index.to_series().dt.year
# eq_240['spi_values'] = np.around(eq_240['spi'], decimals =1)


#=========================================================================================
# # Create DF with nath and reg
# #========================================================================================

df_nat_reg_6m = eq_6
df_nat_reg_6m = df_nat_reg_6m.join(df_nat_6m).sort_index().join(df_reg_6m).sort_index()
df_nat_reg_6m = df_nat_reg_6m.drop(['six_month', 'spi','index','year'], axis=1)


df_nat_reg_10y = eq_120
df_nat_reg_10y = df_nat_reg_10y.join(df_nat_10y).sort_index().join(df_reg_10y).sort_index()
df_nat_reg_10y = df_nat_reg_10y.drop(['ten_years', 'spi','index','year'], axis=1)


# # #=========================================================================================
# # # Plot SPI 
# # #=========================================================================================
     

#Create a list of dataframes to be called
# equiprobabilities_months = [eq_1, 
#                             eq_3,
#                             eq_6, 
#                             eq_12, 
#                             eq_120, 
#                             # eq_180, 
#                             # eq_240
#                             ]
equiprobabilities_months = [ 
                            eq_6,  
                            eq_120, 
                            ]

names_plot = [ 
              '6-months', 
              '10-Years', 
              # '15-Years', 
              # '20-Years'
              ]

# names_plot = ['1-month',
#               '3-months', 
#               '6-months',
#               '12-months', 
#               '10-Years', 
#               # '15-Years', 
#               # '20-Years'
#               ]

for equiprobability, name in zip(equiprobabilities_months, names_plot):
        


    fig, ax = plt.subplots()
    
    # SET COLORS 
    norm = plt.Normalize(equiprobability.spi_values.min(), equiprobability.spi_values.max())
    cmap = mpl.colors.ListedColormap(['darkred','orangered','gold','bisque','lavender','greenyellow', 'darkturquoise','royalblue'])
    
    
    # # BARPLOT: it automatically takes the mean of multiple observations
    sns.barplot(y="index", x="spi", data=equiprobability, ax=ax, orient='h', palette=cmap(norm(equiprobability.spi.values)))
    
    #code to divide the y labels into different frequencies
    freq = int(60) # set the frequency for labelling the yaxis
    ax.set_yticklabels(equiprobability.iloc[::freq].year, fontsize=9)
    ytix = ax.get_yticks()
    ax.set_yticks(ytix[::freq])
    ax.set_title("Streamflow Drought Index for Pecos")
    plt.xlim(-3, 3)
    plt.xlabel('SDI for {}'.format(name))
    plt.ylabel('Years')
    plt.tight_layout()

      
    #LEGEND
    
    ext_dry = mpatches.Patch(color='darkred', label='Extremely dry')
    sev_dry = mpatches.Patch(color='orangered', label='Severely dry')
    dry = mpatches.Patch(color='gold', label='Dry')
    mod_dry = mpatches.Patch(color='bisque', label='Moderately dry')
    mod_wet = mpatches.Patch(color='lavender', label='Moderately wet')
    wet = mpatches.Patch(color='yellowgreen', label='Wet')
    sev_wet = mpatches.Patch(color='darkturquoise', label='Severely wet')
    ext_wet = mpatches.Patch(color='royalblue', label='Extremely wet')
    
    plt.legend(handles=[ext_dry,sev_dry,dry,mod_dry,mod_wet,wet,sev_wet,ext_wet], loc= 'upper right', fontsize = 6)
    
    
    # SAVE AS PDF
    
#     plt.savefig('images/Laredo_updated{}_SPI.pdf'.format(name,))
    plt.show()
    

# df_nat_reg_6m.to_csv('results/Laredo_spi_nat_reg_6m.csv') #Save dictionaries to CSV files
# df_nat_reg_10y.to_csv('results/Laredo_spi_nat_reg_120m.csv') #Save dictionaries to CSV files


