import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
#import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import roc_auc_score, roc_curve, auc
import catboost
from catboost import Pool
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings, gc


# Data Preprocessing


test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 
                                                  'item_id': 'int32'})
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv', 
                              dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 
                                                 'item_category_id': 'int32'})
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], 
                    dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 
                          'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})




train = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)



train.date     = train.date.apply(pd.to_datetime)
train['day']   = train.date.apply(lambda x: x.day)
train['year'] =train.date.apply(lambda x: x.year)
order=['day','year','date_block_num', 'shop_id',  'item_category_id','item_id', 'item_price',
       'item_cnt_day']
train=train[order]
train['item_pricmean_inshop']=train.groupby(['item_id','shop_id'])['item_price'].transform('mean')

# Two of the columns have negative data: 
# item_price: only one row, it's probably a wrong data, replace it with mean value of that product in the same shop.
# item_cnt_day: there are 7000+ '-1', probably means returned products.


#deal with those negative data
for i in range(train.shape[0]):
    if train.iloc[i,5]<0:
        train.iloc[i,5]=train.iloc[i,8]


# Since we only need to predict the total sales in the next month, we change the dataframe from daily record into monthly record. In this way, we can also reduce the dimensions.


trainM=train.groupby(['date_block_num', 'shop_id', 'item_category_id','item_id','item_price'],as_index=False)
trainM=trainM.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']})
trainM.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price','item_cnt', 'mean_item_cnt', 'transactions']

trainM.isnull().sum()


# There is no any missing value in the dataframe. 
#However, we still need to fill in some data in order to see the complete pattern of each shop and item.
#(fill in empty data will influence the mean value, so we calculate those mean value before the next step）
shop_ids = trainM['shop_id'].unique()
item_ids = trainM['item_id'].unique()
empty_df = []
for i in range(34):
    for shop in shop_ids:
        for itemc,item in zip(items['item_category_id'],items['item_id']):
            empty_df.append([i, shop,itemc,item])

empty_df = pd.DataFrame(empty_df, columns=['date_block_num','shop_id','item_category_id','item_id'])
trainM = pd.merge(empty_df, trainM, on=['date_block_num','shop_id','item_category_id','item_id'], how='left')
#merge test set
test['date_block_num']=34
items=items.drop('item_name',axis=1)
test=pd.merge(test,items,on=['item_id'])

#trainM['cnt_lag1_inshop']=trainM.groupby(['shop_id','item_id'])['item_cnt'].shift(1)
#trainM['cnt_lag2_inshop']=trainM.groupby(['shop_id','item_id'])['item_cnt'].shift(2)
trainM['cnt_lag1']=trainM.groupby('item_id')['item_cnt'].shift(1)
trainM['cnt_lag2']=trainM.groupby('item_id')['item_cnt'].shift(2)
trainM['cnt_lag12']=trainM.groupby('item_id')['item_cnt'].shift(12)

#trainM['cnt_cum_inshop']=trainM.groupby(['shop_id','item_id'])['item_cnt'].cumsum()
#trainM['cnt_cum_lag1_inshop']=trainM.groupby(['shop_id','item_id'])['cnt_cum_inshop'].shift(1)
#trainM['cnt_cum_lag2_inshop']=trainM.groupby(['shop_id','item_id'])['cnt_cum_inshop'].shift(2)
#trainM=trainM.drop(['cnt_cum_inshop'],axis=1)

trainM['cnt_cum']=trainM.groupby('item_id')['item_cnt'].cumsum()
trainM['cnt_cum_lag1']=trainM.groupby(['item_id'])['cnt_cum'].shift(1)
trainM=trainM.drop(['cnt_cum'],axis=1)

trainM['item_mean_allshop']=trainM[trainM['item_cnt']!=0].groupby(['date_block_num','item_id'])['mean_item_price'].transform('mean')
trainM['item_mean_diff']=trainM['mean_item_price']-trainM['item_mean_allshop']

trainM['item_cnt_mean_3y']=trainM.groupby('item_id')['item_cnt'].transform('mean')
trainM['item_pri_mean_3y']=trainM[trainM['item_cnt']!=0].groupby(['item_id'])['mean_item_price'].transform('mean')

trainM.fillna(0, inplace=True)


# EDA

# Our final target is to predict the sale of particular item in particular shop in the next month, so EDA will focus more on the sales trend and pattern of individual shops and products.

shop_itemkind=train.groupby(['shop_id','item_id'],as_index=False)['item_price'].agg('mean')
shop_itemkind=pd.DataFrame(shop_itemkind)
shop_itemkind=shop_itemkind.groupby(['shop_id'],as_index=False)['item_id'].agg('count')
shop_itemkind=pd.DataFrame(shop_itemkind)

shop_cnt_sum=trainM.groupby(['shop_id'],as_index=False)['item_cnt'].agg('sum')
shop_cnt_sum=pd.DataFrame(shop_cnt_sum)


plt.style.use('ggplot')
f, (ax1, ax2) = plt.subplots(2, 1,figsize=(60, 40),dpi=200)
ax1.bar(x = range(shop_itemkind.shape[0]),  
        height =shop_itemkind.item_id,  
        tick_label = shop_itemkind.shop_id,  
        color = 'blueviolet',
        width=0.8
        )
ax1.set_ylabel('kinds of product')
ax1.set_xlabel('shop_id')
ax1.tick_params(labelsize=23)
ax1.set_title('The number of items each shop sells',fontsize=35)


ax2.bar(x = range(shop_cnt_sum.shape[0]),  
        height =shop_cnt_sum.item_cnt,  
        tick_label = shop_cnt_sum.shop_id,  
        color = 'hotpink',
        width=0.8
        )
ax2.set_ylabel('total item_cnt')
ax2.set_xlabel('shop_id')
ax2.tick_params(labelsize=23)
ax2.set_title('Total sales of each shop',fontsize=35)
plt.show()


# It shows that shop31 and shop25 are very popular (the total number of sold items is very high) and the kinds of items they sell are the most various among all the shop. We can also see that the trends of the 2 barchart above are very similar, which imply that the more kinds of item a shop sell, the more popular the shop will be.

# Now, we want to know whether the category of an item is closely relative to its price or sales.



item_cum=train.groupby(['item_id'],as_index=False)['item_cnt_day'].agg('sum')
item_cum=pd.DataFrame(item_cum)

itemc_cum=train.groupby(['item_category_id'],as_index=False)['item_cnt_day'].agg('sum')
itemc_cum=pd.DataFrame(itemc_cum)

itemc_mean=train.groupby(['item_category_id'],as_index=False)['item_price'].agg('mean')
itemc_mean=pd.DataFrame(itemc_mean)

itemc_kind=items.groupby('item_category_id')['item_id'].agg('count')
itemc_kind=pd.DataFrame(itemc_kind)

plt.style.use('ggplot')

f, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(80, 60),dpi=200)
ax1.bar(x = range(itemc_cum.shape[0]),  
        height =itemc_cum.item_cnt_day,  
        tick_label = itemc_cum.item_category_id,  
        color = 'seagreen',
        width=0.8
        )
ax1.set_ylabel('total cnt')
ax1.set_xlabel('item_category_id')
ax1.tick_params(labelsize=23)
ax1.set_title('Total sale of each item category',fontsize=30)

ax2.bar(x = range(itemc_mean.shape[0]),  
        height =itemc_mean.item_price,  
        tick_label = itemc_mean.item_category_id,  
        color = 'skyblue',
        width=0.8
        )
ax2.set_ylabel('mean price')
ax2.set_xlabel('item_category_id')
ax2.tick_params(labelsize=23)
ax2.set_title('Mean price of each item category',fontsize=30)

ax3.bar(x = range(itemc_kind.shape[0]),  
        height =itemc_kind.item_id,  
        tick_label = itemc_cum.item_category_id,  
        color = 'firebrick',
        width=0.8
        )
ax3.set_ylabel('item')
ax3.set_xlabel('item_category_id')
ax3.tick_params(labelsize=23)
ax3.set_title('The number of items in each item category',fontsize=30)

plt.show()

itemc_kind.describe()


# category40 is the most popular item_category(sell over 600000 in the given period,rank No.1), and its mean price is not so expensive as well as includes over 5000 different items.



itemc_eachmean=train.groupby(['item_category_id','item_id'],as_index=False)['item_price'].agg('mean')
itemc_eachmean=pd.DataFrame(itemc_eachmean)



f, axes = plt.subplots(14, 6, figsize=(200, 250))
row=0
col=[0,1,2,3,4,5]*14
for i in range(0,84):
    if (i!=0)&(i%6==0):
        row+=1
    df=itemc_eachmean[itemc_eachmean['item_category_id']==i]
    xx=sns.barplot(x="item_id", y="item_price", data=df, ax=axes[row,col[i]])
    xx.set_title(i,fontsize=60)
    xx.tick_params(labelsize=40)

itemc_eachcum=train.groupby(['item_category_id','item_id'],as_index=False)['item_cnt_day'].agg('sum')
itemc_eachcum=pd.DataFrame(itemc_eachcum)

f, axes = plt.subplots(14, 6, figsize=(100, 250))
row=0
col=[0,1,2,3,4,5]*14
for i in range(0,84):
    if (i!=0)&(i%6==0):
        row+=1
    df=itemc_eachcum[itemc_eachcum['item_category_id']==i]
    xx=sns.barplot(x="item_id", y="item_cnt_day", data=df, ax=axes[row,col[i]])
    xx.set_title(i,fontsize=50)
    xx.tick_params(labelsize=40)


# Although some items are in the same item category, their average price and sales are not the same.

mon_cnt=train.groupby(['date_block_num'],as_index=False)['item_cnt_day'].agg('sum')
shop_mon_cnt=train.groupby(['date_block_num','shop_id'],as_index=False)['item_cnt_day'].agg('sum')
itemc_mon_cnt=train.groupby(['date_block_num','item_category_id'],as_index=False)['item_cnt_day'].agg('sum')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                           height=500, width=1000))

fig=go.Figure()
fig.add_trace(go.Scatter(x=mon_cnt['date_block_num'], 
                         y=mon_cnt['item_cnt_day'], mode='lines',
                         line=dict(color='darkred', width=3), 
                         hovertemplate = ''))
fig.update_layout(template=temp, title="Total sales", 
                  hovermode="x unified", width=1000,height=400,
                  xaxis_title='Month', yaxis_title='Sale')
fig.show()


# The total sale of all the items reached a peak at the 11st month and the 23rd month.

f, axes = plt.subplots(10, 6, figsize=(200, 100),sharex=True)
row=0
col=[0,1,2,3,4,5]*10
for i in range(0,60):
    if (i!=0)&(i%6==0):
        row+=1
    df=shop_mon_cnt[shop_mon_cnt['shop_id']==i]
    xx=sns.lineplot(x="date_block_num", y="item_cnt_day", data=df,linewidth=5,color='blue', ax=axes[row,col[i]])
    xx.set_title(i,fontsize=60)
    xx.tick_params(labelsize=60)


# Most shops have been trading during the given period,but some shops only trades for a very short time,e.g. shop0,shop1,shop14...
# Besides, most of the shops have 2 peaks during the given period. Those peaks often appeared around 11st(Dec.2013) or 23rd (Dec.2014) month,which is similar to the trend of total sale. 




f, axes = plt.subplots(14, 6, figsize=(200, 150),sharex=True)
row=0
col=[0,1,2,3,4,5]*14
for i in range(0,84):
    if (i!=0)&(i%6==0):
        row+=1
    df=itemc_mon_cnt[itemc_mon_cnt['item_category_id']==i]
    xx=sns.lineplot(x="date_block_num", y="item_cnt_day", data=df,linewidth=5,color='purple', ax=axes[row,col[i]])
    xx.set_title(i,fontsize=60)
    xx.tick_params(labelsize=60)

itemsale=train.groupby(['item_id'],as_index=False)['item_cnt_day'].agg('sum')
itemsale=pd.DataFrame(itemsale)
itemsale=itemsale.sort_values(by=['item_cnt_day'],ascending=False)
sns.boxplot(x=itemsale['item_cnt_day'])


def RANGE(x):
    if x <= 10:
        return 0
    elif 10 < x <=100:
        return 1
    elif 100 < x <= 500:
        return 2
    elif 500 < x <= 1000:
        return 3
    elif 1000< x:
        return 4


itemsale['cnt_range']=itemsale['item_cnt_day'].map(RANGE)
itemsale['cnt_range']=itemsale['cnt_range'].astype('str')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                           height=500, width=1000))


target=itemsale.cnt_range.value_counts(normalize=True)
#target.rename(index={0:'<10',1:'<100',2:'<500',3:'<1000',4:'>1000'},inplace=True)
label=['[10,100)','<10','[100,500)','[500,1000)','>1000']
pal, color=['cornsilk','seashell','honeydew','aliceblue','mistyrose'], ['gold','sandybrown','darkseagreen','powderblue','tomato']
fig=go.Figure()
fig.add_trace(go.Pie(labels=label, values=target*100, hole=.45, 
                     showlegend=True,sort=False, 
                     marker=dict(colors=color,line=dict(color=pal,width=2.5)),
                     hovertemplate = "%{label} Accounts: %{value:.2f}%<extra></extra>"))
fig.update_layout(template=temp, title='Sales Distribution', 
                  legend=dict(traceorder='reversed',y=1.05,x=0),
                  uniformtext_minsize=15, uniformtext_mode='hide',width=700)
fig.show()


# Most of the sales are between 0 and 100, accounts for 71% of all the items. Only 2.78% item sell over 1000 in the given period.



itempri=train.groupby(['item_id'],as_index=False)['item_price'].agg('mean')
itempri=pd.DataFrame(itempri)
itempri=itempri.sort_values(by=['item_price'],ascending=False)




sns.boxplot(x=itempri['item_price'])


sns.boxplot(x=itempri.iloc[1:-1,1])

def PRI(x):
    if x < 100:
        return 0
    elif 100 <= x <500:
        return 1
    elif 500 <= x < 1000:
        return 2
    elif 1000 <= x < 5000:
        return 3
    elif 5000<= x:
        return 4

itempri['pri_range']=itempri['item_price'].map(PRI)
itempri['pri_range']=itempri['pri_range'].astype('str')


target=itempri.pri_range.value_counts(normalize=True)
#target.rename(index={0:'<100',1:'<500',2:'<1000',3:'<5000',4:'>5000'},inplace=True)
label=['[100,500)','[1000,5000)','[500,1000)','<100','>5000']
pal, color=['cornsilk','seashell','honeydew','aliceblue','mistyrose'], ['gold','sandybrown','darkseagreen','powderblue','tomato']
fig=go.Figure()
fig.add_trace(go.Pie(labels=label, values=target*100, hole=.45, 
                     showlegend=True,sort=False, 
                     marker=dict(colors=color,line=dict(color=pal,width=2.5)),
                     hovertemplate = "%{label} Accounts: %{value:.2f}%<extra></extra>"))
fig.update_layout(template=temp, title='Mean Price Distribution', 
                  legend=dict(traceorder='reversed',y=1.05,x=0),
                  uniformtext_minsize=15, uniformtext_mode='hide',width=700)
fig.show()


# Most items' mean prices are from 100 to 500, only 1.71% of these items' mean prices are greater than 5000.
