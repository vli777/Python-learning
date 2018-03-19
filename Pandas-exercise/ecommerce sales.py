import numpy as np
import pandas as pd
import os

os.chdir('') # wd path
transactions    = pd.read_csv('sales_train_v2.csv')
items           = pd.read_csv('items.csv')
item_categories = pd.read_csv('item_categories.csv')
shops           = pd.read_csv('shops.csv')

cols = ['date','shop_id', 'item_id', 'item_price', 'item_cnt_day']
a = transactions[cols]
a[['day','month','year']]=a['date'].str.split('.',expand=True).astype(int)

#select target
b = a.loc[(a['month'] == 9) & (a['year']==2014)]
# max_revenue
b['sales'] = b['item_price']*b['item_cnt_day']
b['total'] = b.groupby('shop_id')['sales'].transform('sum')
b=b.sort_values(by='total', ascending = False)
max_revenue = b['total'].iloc[0]

# summer target
c = a.loc[(a['year']==2014)]
c = c.loc[(c['month'].between(6,8))]
c['sales'] = c['item_price']*c['item_cnt_day']

c = c.join(items, on='item_id', how='inner', lsuffix='c_', rsuffix='item_')
c['best'] = c.groupby('item_category_id')['sales'].transform('sum')
c=c.sort_values(by='best', ascending=False)
category_id_with_max_revenue = c['item_category_id'].iloc[1]

#items with constant prices
d= a[['item_id','item_price']]
e=d.groupby(['item_id', 'item_price']).size().reset_index(name='ct')
delta = e.groupby('item_id').size().reset_index(name='ct')

num_items_constant_price = len(delta[delta.ct==1])

# var of dec sales
shop_id=25
yr = 2014
mo = 12

c = a.loc[(a['year']==yr) & (a['month']==mo) & (a['shop_id']==shop_id)]
d = c.sort_values(by='day').groupby(['day','item_id'], as_index=False)['item_cnt_day'].sum()
d= d.groupby(['day'], as_index=False)['item_cnt_day'].sum()

total_num_items_sold = d['item_cnt_day'].as_matrix()
days = d['day'].as_matrix()

total_num_items_sold_var= np.var(total_num_items_sold, ddof=1)
