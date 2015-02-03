
# In[91]:

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsaplots
from pandas.tools.plotting import autocorrelation_plot
import scipy.stats as ss
import statsmodels.tsa.api as api
tsa = sm.tsa
get_ipython().magic(u'matplotlib inline')


# In[124]:

# loading the data
path = "c:\\Users\\Ana\\Documents\\Dropbox\\Job search\\Data incubator\\Argentina\\02 Clean"

goog = pd.read_csv(path+"\\goog_dolar.csv")
rates = pd.read_csv(path+"\\rates.csv")


# In[125]:

# cleaning and formatting the data

# set datetime index
goog = goog.set_index(pd.to_datetime(goog['start']))
rates=rates.set_index(pd.to_datetime(rates['date']))

# drop unecessary column in google data
goog=goog['goog_dolar']

# separate formal and black market (blue) exchange rates
formal = rates[rates['tipo']=='Casas de cambio']
blue = rates[rates['tipo']=='Informal']

# resample weekly
blue_week = blue['venta'].resample('W', how='mean')
formal_week = formal['venta'].resample('W', how='mean')

# make google and rate data the same lenght
goog = goog[goog.index >= blue_week.index.min()]
blue_week = blue_week[blue_week.index>=goog.index.min()]
blue_week = blue_week[blue_week.index<=goog.index.max()]
formal_week = formal_week[formal_week.index>=goog.index.min()]
formal_week = formal_week[formal_week.index<=goog.index.max()]

# log series to smooth
log_blue = np.log10(blue_week)
log_formal = np.log10(formal_week)
log_goog = np.log10(goog)

# plot series
plt.figure(figsize=(15,5))
plt.title("Blue and Official rates, Google search index")
log_goog.plot(secondary_y=True, label="Google search", color='red', legend=True)
log_blue.plot(label= "Blue dollar", legend=True) 
log_formal.plot(label = "Official dollar", legend=True)    
plt.savefig('graph1.png')


# correlations 
xcorr_blue = ss.pearsonr(log_blue, log_goog)
print "Google and Blue rate cross-correlation (rho): %1.2f" % xcorr_blue[0]
print "Google and Blue rate cross-correlation (pval): %1.2f" % xcorr_blue[1]

xcorr_formal = ss.pearsonr(log_formal, log_goog)
print "Google and Official rate cross-correlation (rho): %1.2f" % xcorr_formal[0]
print "Google and Official rate cross-correlation (pval): %1.2f" % xcorr_formal[1]


# Out[125]:

#     Google and Blue rate cross-correlation (rho): 0.85
#     Google and Blue rate cross-correlation (pval): 0.00
#     Google and Official rate cross-correlation (rho): 0.79
#     Google and Official rate cross-correlation (pval): 0.00
#     

# image file:

# In[94]:

# differentiating the series to get rid of trend
diff_blue = log_blue.diff()
diff_blue = diff_blue.dropna()

diff_goog = log_goog.diff()
diff_goog = diff_goog.dropna()

diff_formal = log_formal.diff()
diff_formal = diff_formal.dropna()

# plotting differetiated series
plt.figure(figsize=(15,5))
plt.title("Differentiated series, Blue and Official rates, Search index")
diff_goog.plot(secondary_y=True, label= "Google search", legend=True, color='red')
diff_blue.plot(label= "Blue dollar", legend=True)
diff_formal.plot(label= "Official dollar", legend=True)
plt.savefig('graph2.png')

# correlations 
xcorr_blue = ss.pearsonr(diff_blue, diff_goog)
print "Differentiated Google and Blue rate cross-correlation (rho): %1.2f" % xcorr_blue[0]
print "Differentiated Google and Blue rate cross-correlation (pval): %1.2f" % xcorr_blue[1]

xcorr_formal = ss.pearsonr(diff_formal, diff_goog)
print "Differentiated Google and Official rate cross-correlation (rho): %1.2f" % xcorr_formal[0]
print "Differentiated Google and Official rate cross-correlation (pval): %1.2f" % xcorr_formal[1]


# Out[94]:

#     Differentiated Google and Blue rate cross-correlation (rho): 0.17
#     Differentiated Google and Blue rate cross-correlation (pval): 0.02
#     Differentiated Google and Official rate cross-correlation (rho): -0.06
#     Differentiated Google and Official rate cross-correlation (pval): 0.35
#     

# image file:

# In[95]:

# Auto and partial autocorrelation plots for blue rate 
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(221)
fig = sm.graphics.tsa.plot_acf(log_blue, lags=40, ax=ax1)
plt.title('Autocorrelation Blue rate')
ax2 = fig.add_subplot(222)
fig = sm.graphics.tsa.plot_pacf(log_blue, lags=40, ax=ax2)
plt.title('Partial Autocorrelation Blue rate')
plt.savefig('graph3.png')


# Out[95]:

# image file:

# In[96]:

# Auto and partial autocorrelation plots for google rate 
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(221)
fig = sm.graphics.tsa.plot_acf(log_goog, lags=40, ax=ax1)
plt.title('Autocorrelation Google Searches')
ax2 = fig.add_subplot(222)
fig = sm.graphics.tsa.plot_pacf(log_goog, lags=40, ax=ax2)
plt.title('Partial Autocorrelation Google Searches')
plt.savefig('graph4.png')


# Out[96]:

# image file:

# In[97]:

# fit an AR(2) model on the differentiated dolar blue rate
ar_mod = sm.tsa.AR(diff_blue, freq='W')

# checking that the optimal order for model is indeed 2 using the BIC
order_opt = ar_mod.select_order(20, 'bic')
print ('optimal order for AR model = %d' % order_opt)


# Out[97]:

#     optimal order for AR model = 2
#     

# In[98]:

# Fiting the model
ar_res = ar_mod.fit(3)
print ('parameters for the model: ')
print ar_res.params


# Out[98]:

#     parameters for the model: 
#     const    0.001775
#     L1.y     0.427645
#     L2.y    -0.269917
#     L3.y     0.128358
#     dtype: float64
#     

# In[99]:

# plot observed and fitted values 
ar_fit = ar_res.fittedvalues
fig1 = plt.figure(figsize=(12,8))
ax1 = fig1.add_subplot(211)
ar_fit.plot(label='fitted values', ax=ax1, color='green')
diff_blue.plot(label='observed', ax=ax1, color='blue')
plt.title('Observed and fitted values, Dolar Blue')
plt.legend()
plt.show()



# Out[99]:

# image file:

# In[100]:

# root mean  squared error of in-sample forecast
def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

mse1 = mse(ar_res.fittedvalues, diff_blue)


"%1.12f" % np.sqrt(mse1)


# Out[100]:

#     '0.009616562307'

# In[101]:

# merging datasets together into one dataframe
goog = pd.DataFrame(log_goog)
goog = goog.reset_index()
goog.columns = [['date', 'goog']]

blue = pd.DataFrame(log_blue)
blue = blue.reset_index()
blue.columns = ['date', 'blue']

data = pd.merge(blue, goog)
data.index = pd.DatetimeIndex(data['date'])
data = data[['blue', 'goog']]
data = data.diff().dropna()
data.head()


# Out[101]:

#                     blue      goog
#     2011-01-16  0.000358 -0.074634
#     2011-01-23  0.001575  0.000000
#     2011-01-30  0.000335  0.026329
#     2011-02-06 -0.002415  0.024824
#     2011-02-13  0.001094 -0.051153

# In[122]:

# make a VAR model
model = api.VAR(data)

# check on order of variables
model.select_order(8)


# Out[122]:

#                      VAR Order Selection                 
#     =====================================================
#                aic          bic          fpe         hqic
#     -----------------------------------------------------
#     0       -14.51       -14.48    4.972e-07       -14.50
#     1       -14.71       -14.61    4.080e-07       -14.67
#     2       -14.79       -14.63    3.780e-07       -14.72
#     3      -14.88*      -14.65*   3.447e-07*      -14.79*
#     4       -14.88       -14.59    3.454e-07       -14.76
#     5       -14.86       -14.50    3.537e-07       -14.71
#     6       -14.85       -14.43    3.543e-07       -14.68
#     7       -14.82       -14.33    3.662e-07       -14.62
#     8       -14.82       -14.26    3.682e-07       -14.59
#     =====================================================
#     * Minimum
#     
#     

#     {'aic': 3, 'bic': 3, 'fpe': 3, 'hqic': 3}

# In[104]:

# fit model 
results = model.fit(3) 
results.summary()


# Out[104]:

#       Summary of Regression Results   
#     ==================================
#     Model:                         VAR
#     Method:                        OLS
#     Date:           Sun, 01, Feb, 2015
#     Time:                     20:16:08
#     --------------------------------------------------------------------
#     No. of Equations:         2.00000    BIC:                   -14.6946
#     Nobs:                     208.000    HQIC:                  -14.8284
#     Log likelihood:           975.326    FPE:                3.31638e-07
#     AIC:                     -14.9193    Det(Omega_mle):     3.10395e-07
#     --------------------------------------------------------------------
#     Results for equation blue
#     ==========================================================================
#                  coefficient       std. error           t-stat            prob
#     --------------------------------------------------------------------------
#     const           0.001872         0.000669            2.800           0.006
#     L1.blue         0.360886         0.068928            5.236           0.000
#     L1.goog         0.055969         0.010053            5.567           0.000
#     L2.blue        -0.367170         0.071707           -5.120           0.000
#     L2.goog         0.012833         0.010526            1.219           0.224
#     L3.blue         0.156227         0.065651            2.380           0.018
#     L3.goog         0.032017         0.010433            3.069           0.002
#     ==========================================================================
#     
#     Results for equation goog
#     ==========================================================================
#                  coefficient       std. error           t-stat            prob
#     --------------------------------------------------------------------------
#     const           0.002385         0.004697            0.508           0.612
#     L1.blue         1.415123         0.484260            2.922           0.004
#     L1.goog        -0.203676         0.070631           -2.884           0.004
#     L2.blue        -0.366462         0.503788           -0.727           0.468
#     L2.goog        -0.224549         0.073949           -3.037           0.003
#     L3.blue        -0.602678         0.461237           -1.307           0.193
#     L3.goog        -0.150240         0.073298           -2.050           0.042
#     ==========================================================================
#     
#     Correlation matrix of residuals
#                 blue      goog
#     blue    1.000000  0.231636
#     goog    0.231636  1.000000
#     
#     
#     

# In[105]:

# testing granger causality 
results.test_causality('blue', ['goog'], kind='f')


# Out[105]:

#     Granger causality f-test
#     ==============================================================
#        Test statistic   Critical Value          p-value         df
#     --------------------------------------------------------------
#             11.656369         2.627103            0.000  (3, 402L)
#     ==============================================================
#     H_0: ['goog'] do not Granger-cause blue
#     Conclusion: reject H_0 at 5.00% significance level
#     

#     {'conclusion': 'reject',
#      'crit_value': 2.6271027427068758,
#      'df': (3, 402L),
#      'pvalue': 2.4388685011552741e-07,
#      'signif': 0.05,
#      'statistic': 11.65636896072697}

# In[106]:

# root mean squared error of in-sample forecast
mse2 = mse(results.fittedvalues['blue'], diff_blue)

"%1.12f" % np.sqrt(mse2)


# Out[106]:

#     '0.008875448414'

# In[108]:

# plot observed and AR and VAR fitted values 
ar_fit = ar_res.fittedvalues
var_fit = results.fittedvalues['blue']
fig1 = plt.figure(figsize=(12,8))
ax1 = fig1.add_subplot(211)
ar_fit.plot(label='Blue-only model', ax=ax1, color='green')
var_fit.plot(label='Search-augmented model', color='red')
diff_blue.plot(label='Observed', ax=ax1, color='blue')
plt.title('Blue-only and Search-augmented in-sample fitted values for Dolar Blue')
plt.legend(loc='upper left')
plt.savefig('graph5.png')
plt.show()


# Out[108]:

# image file:

# In[109]:

# Out of sample predicitons

# 1 steps ahead AR predictor function
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample

out_sample_AR = []
for i in range(0, 112):
    end = 100+i
    res = sm.tsa.ARMA(data['blue'].iloc[i:end], (2, 0)).fit(trend="nc")
    
    # get what you need for predicting one-step ahead
    params = res.params
    residuals = res.resid
    p = res.k_ar
    q = res.k_ma
    k_exog = res.k_exog
    k_trend = res.k_trend
    steps = 1
    
    pred = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=data['blue'].iloc[i:end], exog=None, start=len(data['blue'].iloc[i:end]))
    out_sample_AR.append(pred)


# In[110]:

# 1-step ahead prediction for VAR model 

out_sample_VAR = []
for i in range(0, 112):
    end = 100+i
    model = api.VAR(data.iloc[i:end])
    results = model.fit(3) 
    
    lag_order = results.k_ar
    pred = results.forecast(data.iloc[i:end].values[-lag_order:], 1)
    out_sample_VAR.append(pred)


# In[111]:

out_VAR = pd.DataFrame([row[0][0] for row in out_sample_VAR])
out_VAR.columns = ["out_VAR"]

out_AR = pd.DataFrame(out_sample_AR)
out_AR.colums = ["out_AR"]


# In[115]:

fig1 = plt.figure(figsize=(15,5))
plt.plot(data['blue'].iloc[100:], color='blue', label="Blue rate")
plt.plot(out_AR, color='green',  label="Blue-only model")
plt.plot(out_VAR, color='red', label="Search-augmented model")
plt.title('1-step ahead predictions for the Dolar Blue')
plt.legend(loc='upper left')
plt.savefig('graph6.png')


# Out[115]:

# image file:

# In[82]:

concat = pd.concat([ out_VAR, out_AR], axis=1)
concat.columns = ['out_VAR', 'out_AR']
concat.head()


# Out[82]:

#         out_VAR    out_AR
#     0  0.002594 -0.001093
#     1  0.005110  0.001139
#     2  0.003020  0.001702
#     3  0.009235  0.003418
#     4  0.009214  0.005210

# In[83]:

blue = data['blue'].iloc[100:].values
blue = pd.DataFrame(blue)
blue.columns = ['blue']
blue.head()

concat = pd.concat([concat, blue], axis=1)


# In[84]:

concat['se_AR'] = (concat['out_AR']-concat['blue'])**2
concat['se_VAR'] = (concat['out_VAR']-concat['blue'])**2


# In[85]:

print np.sqrt(concat['se_AR'].mean())
print np.sqrt(concat['se_VAR'].mean())


# Out[85]:

#     0.0122140971427
#     0.0115773247722
#     

# In[126]:

get_ipython().system(u'ipython nbconvert --to python blue.ipynb')


# Out[126]:

#     [NbConvertApp] Using existing profile dir: u'C:\\Users\\Ana\\.ipython\\profile_default'
#     [NbConvertApp] Converting notebook blue.ipynb to python
#     [NbConvertApp] Support files will be in blue_files\
#     [NbConvertApp] Loaded template python.tpl
#     [NbConvertApp] Writing 14697 bytes to blue.py
#     

# In[ ]:



