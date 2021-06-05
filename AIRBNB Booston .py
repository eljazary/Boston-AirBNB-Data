#!/usr/bin/env python
# coding: utf-8

# ## CRISP-DM Process 
# 
# 1. Business Understanding
#     
# 2. Data Understanding
# 
# 3. Prepare Data
# 
# 4. Data Modeling
# 
# 5. Evaluate the Results
# 
# 6. Deploy

# # I- Business understanding

# Who can host on Airbnb?
# Behind every stay is a host, a real person who can give you the details you need to check in and feel at home. They can interact with guests in different ways, depending on the type of place or experience they booked
# lmost anyone can be a host. It's free to sign up and list both stays and experiences. Whether they’re hosting a place to stay or a local activity, all hosts are expected to meet our quality standards every time
# (<a href=" https://www.airbnb.com/help/article/18/who-can-host-on-airbnb#:~:text=Behind%20every%20stay%20is%20a,place%20or%20experience%20they%20booked.&text=It's%20free%20to%20sign%20up%20and%20list%20both%20stays%20and%20experiences.">link</a>)
# 
# ### so we try to figure out the below issues <br>
# a. the vairance of  price across specific period which data set include  after removing some outlier <br>
# b. also trying to make scheme for correlation between parameters <br>
# c. making a trail to analysis the text and comments of customer to know little bit about what customer need to know <br> 
# d. predict the price of unit based on two model and evalute our models  

# 

# ## I.I question neeed to ansewrs  <br>
# what the most major price  reange required ? <br>
# what is the factors affect the price ? <br>
# what are the comments  of the guest they would to say ?<br> 
# what the factors affects the price ? <br>

# In[522]:


#importing Labiraires 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud 


# In[523]:


#reading file 
calendar = pd.read_csv ('calendar.csv')
calendar.info()


# In[524]:


calendar['date'] = pd.to_datetime(calendar['date']).dt.normalize()


# In[525]:


calendar['listing_id'] = calendar['listing_id'].astype(str)


# In[526]:


calendar['price'] = calendar['price'].replace({'\$':''}, regex = True)


# In[527]:


calendar.price.notnull().value_counts()


# In[528]:


calendar.available.value_counts()


# In[529]:


calendar.dropna(subset = ['price'] , inplace= True )


# In[530]:


calendar['price'] = pd.to_numeric(calendar['price'],errors='coerce')


# In[531]:


calendar.head(20)


# In[532]:


calendar.date.max () ,  calendar.date.min()  


# In[533]:


calendar.info()


# In[ ]:





# In[534]:


calendar['quarter'] = pd.PeriodIndex(calendar.date, freq='Q')


# In[535]:


calendar['month'] = pd.PeriodIndex(calendar.date, freq='m')


# In[536]:


plt.rcParams["figure.figsize"] = (20,8)
sns.boxplot( data = calendar  , x='month',  y = 'price' )


# In[537]:


plt.rcParams["figure.figsize"] = (20,8)
sns.distplot(calendar['price'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title ('disrtibution of price over time')


# ### Rigth skewed 

# ### Removing Outlier 

# In[538]:


calendar.describe().price


# ### price of major unit lies between 20 to 500 and others is outliers 

# In[539]:


calendar_new = calendar[(calendar['price'] >20) & (calendar['price'] <500)]
calendar_new.head()


# In[540]:


plt.rcParams["figure.figsize"] = (20,8)
sns.distplot(calendar_new['price'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title ('disrtibution of price over time')


# #### After Removing Of Outlier 
# above histogram of price show tendancy to right skewed means 
# the higher price mean less hosting times and hosting increase by the less of price 

# In[541]:


plt.rcParams["figure.figsize"] = (20,8)
sns.boxplot( data = calendar_new  , x='month',  y = 'price' )


# ### Listings 
# =============================================================

# In[542]:


listings = pd.read_csv ('listings.csv')
listings.info()


# In[543]:


listings.columns


# In[544]:


listings.head()


# 

# In[545]:


listings['price']  = listings['price'].str.replace(',', '')
listings['price']  = listings['price'].str.replace('$', '')
listings['price'] = pd.to_numeric(listings['price'])


# In[546]:


listings['last_scraped'] = pd.to_datetime(listings.last_scraped)


# In[547]:


listings['cleaning_fee']  = listings['cleaning_fee'].str.replace(',', '')
listings['cleaning_fee']  = listings['cleaning_fee'].str.replace('$', '')
listings['cleaning_fee'] = pd.to_numeric(listings['cleaning_fee'])


# In[548]:


drop_column = ['id','host_id', 'listing_url' , 'scrape_id','jurisdiction_names','neighbourhood_group_cleansed','has_availability','license','neighbourhood_cleansed','has_availability', 'square_feet',
             'thumbnail_url' , 'medium_url' , 'picture_url' 
             , 'xl_picture_url' , 'host_url', 'host_thumbnail_url', 'host_picture_url']


# In[549]:


listings.drop(drop_column , axis = 1  , inplace = True )


# In[550]:


listings.info()


# In[551]:


listings.dropna(how = 'all')


# In[552]:


listings.shape


# In[553]:


listings.property_type.dropna()


# In[554]:


# get categorical data 
listings_categorical =listings.select_dtypes(include=['object'])
listings_categorical.shape


# In[555]:


# get numerical data 
listings_numerical = listings.select_dtypes(include=['int64','float'])
listings_numerical.head()


# In[556]:


listings_numerical = listings_numerical.dropna(how = 'all')
listings_numerical.reset_index(inplace=True)
listings_numerical.head()


# ## studing the correlation between numerical parameters 

# In[557]:


n_corr = listings.select_dtypes(include=['int64', 'float64']).corr()
mask = np.zeros_like(n_corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(24,12))
plt.title('Heatmap of corr of features')
sns.heatmap(n_corr, mask = mask, vmax=.3, square=True, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


# In[558]:


import plotly.express as px
fig = px.scatter_matrix(listings , dimensions=["price", "bathrooms", "bathrooms", "cleaning_fee", "beds"] , color="room_type")
fig.show()


# In[559]:


listings['last_scraped'] = listings['last_scraped'].dt.normalize()


# In[560]:


listings['last_scraped']


# In[561]:


import plotly.express as px
fig = px.scatter_mapbox(listings, lat="latitude", lon="longitude", color="room_type", size="price",
                  color_continuous_scale=px.colors.cyclical.IceFire, animation_frame="last_scraped" , size_max=15, zoom=10,
                  mapbox_style="carto-positron")
fig.show()


# In[562]:


listings.property_type.dropna(how = 'all')


# In[563]:


listings.dropna(subset=['summary'], inplace=True)
listings.shape


# In[564]:


listings.reset_index(inplace=True)
listings.head()


# ## 2.1 analysis of summary column by tools on NLP by measuring 
# 1- measuring subjectivity & polarity <br>
# 2- measuring the most repeated words 

# In[565]:


#listings.reset_index(inplace=True)
listings.head()


# In[566]:


summary = ''
for i in range(listings.shape[0]):
    summary = summary + listings['summary'][i]


# In[567]:


sns.set(color_codes=True)
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(summary) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()


# In[568]:


from nltk.tokenize import word_tokenize
import pandas as pd

import nltk
nltk.download('punkt')


# In[569]:


listings['tokens'] = listings['summary'].apply(word_tokenize)
listings['tokens'].head()


# In[570]:


all_tokens = []
for i in range(listings.shape[0]):
    all_tokens = all_tokens + listings['tokens'][i]


# In[571]:


from nltk.probability import FreqDist
fdist = FreqDist(all_tokens)
fdist


# In[572]:


tokens1 = [word for word in all_tokens if word.isalnum()]


# In[573]:


nltk.download('stopwords')
from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))


# In[574]:


tokens2 = [x for x in tokens1 if x.lower() not in english_stopwords]


# In[575]:


tokens2_string = ''
for value in tokens2:
    tokens2_string = tokens2_string + value + ' '


# In[576]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', colormap='plasma', 
                min_font_size = 10).generate(tokens2_string) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()


# In[577]:


fdist1 = FreqDist(tokens2)
fdist1
counts = pd.Series(fdist1)
counts = counts[:20]
counts.sort_values(ascending=False, inplace=True)
counts


# In[578]:


#Generates bar graph
ax = counts.sort_values(ascending=True).plot(kind='barh', figsize=(10, 10), fontsize=12)

#X axis text and display style of categories
ax.set_xlabel("Frequency", fontsize=12)

#Y axis text
ax.set_ylabel("Word", fontsize=12)

#Title
ax.set_title("Top 20 words with frequency in summary", fontsize=20)

#Annotations
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.1, i.get_y()+.31, str(round((i.get_width()), 2)), fontsize=10, color='dimgrey')


# In[579]:


from textblob import TextBlob


# In[580]:


TextBlob("Today is a great day").sentiment


# In[581]:


TextBlob("Today is not a great day").sentiment


# In[582]:


def generate_polarity(summary):
    '''Extract polarity score (-1 to +1) for each comment'''
    return TextBlob(summary).sentiment[0]


# In[583]:


listings['polarity'] = listings['summary'].apply(generate_polarity)
listings['polarity'].head()


# In[584]:


def generate_subjectivity(summary):
    '''Extract subjectivity score (0 to +1) for each comment'''
    return TextBlob(summary).sentiment[1]


# In[585]:


listings['subjectivity'] = listings['summary'].apply(generate_subjectivity)
listings['subjectivity'].head()


# In[586]:


num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(listings['polarity'], num_bins, facecolor='green', alpha=0.6)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of polarity / sentiment score')
plt.show();


# In[587]:


positive_sentiment = listings.nlargest(2000, 'polarity')
positive_sentiment = positive_sentiment[['summary', 'polarity','state','city', 'subjectivity']]
positive_sentiment.style.set_properties(subset=['summary'], **{'width': '300px'})


# In[588]:


positive_sentiment.state.value_counts()


# In[589]:


num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(listings['subjectivity'], num_bins, facecolor='green', alpha=0.6)
plt.xlabel('subjectivity')
plt.ylabel('Count')
plt.title('Histogram of polarity / sentiment score')
plt.show();


# In[590]:


positive_sentiment['summary']


# In[591]:


negative_sentiment = listings.nsmallest(2000, 'polarity')
negative_sentiment = negative_sentiment[['summary', 'city','state','name','notes','polarity']]
negative_sentiment.style.set_properties(subset=['summary'], **{'width': '300px'})


# In[592]:


negative_sentiment.city.value_counts()


# In[593]:


negative_sentiment[['summary' ,'notes']]


# In[594]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[595]:


def generate_named_entities(comment):
    '''Return the text snippet and its corresponding enrity label in a list'''
    return [(ent.text.strip(), ent.label_) for ent in nlp(comment).ents]


# In[596]:


listings['named_entities'] = listings['summary'].apply(generate_named_entities)
listings.head()


# In[597]:


from spacy import displacy


# In[598]:


for i in range(10,40):
    if listings['named_entities'][i]:
        displacy.render(nlp(listings['summary'][i]), style='ent', jupyter=True)


# # 3-prediction of price according to different parameter 

# In[599]:


listings_final = listings[(listings['price'] > 20) & (listings['price'] < 500)]
listings_final.info()


# In[600]:


# select numeric cols
num_cols = ['price', 'latitude','longitude', 'accommodates', 'bedrooms', 'bathrooms', 'beds', 
             'cleaning_fee', 'guests_included', 'availability_30', 'availability_60', 'availability_90', 
             'availability_365', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
             'review_scores_location', 'review_scores_value', 'calculated_host_listings_count']

numeric = listings_final.select_dtypes(include=['int64', 'float64'])[num_cols]
print(numeric.info())

# transform categorical columns into numeric and prepare new data frame
cat_cols = ['host_response_time', 'host_is_superhost', 'room_type', 'bed_type',
             'cancellation_policy', 'property_type', 'host_identity_verified', 'instant_bookable',
            'host_has_profile_pic', 'require_guest_profile_picture', 'require_guest_phone_verification']

numeric[cat_cols] = listings_final[cat_cols]

num_copy = numeric.copy()

num_copy = num_copy.replace({ "host_is_superhost": {"t": 1, "f": 2}, "instant_bookable": {"t": 1, "f": 2}, 
                                "host_identity_verified": {"t": 1, "f": 2}, "require_guest_profile_picture": {"t": 1, "f": 2},
                                "room_type": {"Entire home/apt": 1, "Private room": 2, "Shared room": 3}, "host_has_profile_pic": {"t": 1, "f": 2},
                               "bed_type": {"Real Bed": 1, "Futon": 2, "Airbed": 3, "Pull-out Sofa": 4, "Couch": 5},
                               "require_guest_phone_verification": {"t": 1, "f": 2},
                               "cancellation_policy": {"moderate": 1, "flexible": 2, "strict": 3, "super_strict_30": 4}})

dummies = pd.get_dummies(num_copy)
print(dummies.info())


# ## 3.1 linear Regression  

# In[601]:


for col in dummies:
    dummies[col].fillna((dummies[col].mean()), inplace=True)


# In[602]:


y = dummies['price'].astype(float)


# In[603]:


X = dummies.drop('price',  axis =1 )


# In[604]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[605]:


lm_model =  LinearRegression()
lm_model.fit(X_train , y_train )
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)


# In[606]:


#r2 value
r2_scores_test = r2_score(y_test, y_test_preds)
r2_scores_train = r2_score(y_train, y_train_preds)


# In[607]:


print (r2_scores_test , r2_scores_train )


# In[608]:


fig = plt.figure(figsize =(10, 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = plt.axes(aspect = 'equal')
plt.subplot(121)
plt.title('regplot for distribution', fontsize=14)
sns.regplot(y_test, y_test_preds, color='blue')
plt.xlabel('Real Price', fontsize=14)
plt.ylabel('Predicted Price', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(122)
sns.distplot(y_test_preds, hist=False,
             kde_kws={'color': 'b', 'lw': 2, 'label': 'Predicted price'})
sns.distplot(y_test, hist=False,
             kde_kws={'color': 'r', 'lw': 2, 'label': 'Real price'})
plt.title('Distribution comparison', fontsize=14)
plt.ylabel('Probablity', fontsize=12)
plt.xlabel('Price (USD)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Predicted price', 'Real price'], prop={"size":12})
plt.show()


# In[609]:


feature_importance = pd.DataFrame(
    {'features': X.columns, 'coefficients': lm_model.coef_}
).sort_values(by='coefficients')
feature_importance['features'] = feature_importance['features']

import plotly.express as px
fig = px.bar(x='features', y='coefficients',
             data_frame=feature_importance, height=60)
fig.show();


# In[610]:


from sklearn.ensemble import RandomForestRegressor
# Create instance of Random Forest Regressor and evaluate model
model_rf = RandomForestRegressor(n_estimators=76, random_state=42) 
model_rf.fit(X_train , y_train )
y_test_preds = model_rf.predict(X_test)
y_train_preds = model_rf.predict(X_train)
r2_scores_test = r2_score(y_test, y_test_preds)
r2_scores_train = r2_score(y_train, y_train_preds)
print (r2_scores_test , r2_scores_train )


# In[611]:


fig = plt.figure(figsize =(10, 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = plt.axes(aspect = 'equal')
plt.subplot(121)
plt.title('regplot for distribution', fontsize=14)
sns.regplot(y_test, y_test_preds, color='blue')
plt.xlabel('Real Price', fontsize=14)
plt.ylabel('Predicted Price', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(122)
sns.distplot(y_test_preds, hist=False,
             kde_kws={'color': 'b', 'lw': 2, 'label': 'Predicted price'})
sns.distplot(y_test, hist=False,
             kde_kws={'color': 'r', 'lw': 2, 'label': 'Real price'})
plt.title('Distribution comparison', fontsize=14)
plt.ylabel('Probablity', fontsize=12)
plt.xlabel('Price (USD)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Predicted price', 'Real price'], prop={"size":12})
plt.show()


# ## Final Outcomes 

# used two model to predcit the price based on some numerical and Categorical Varibales ; these two model are <br>
# 1- linear Regression <br>
# 2- Random Forest Regressor <br>
# and plots show thr result from Random Forest Regressor more accurate and able to get high score for both test and train set 

# In[ ]:


<module 'pandas' from '/usr/local/lib/python3.6/site-packages/pandas/__init__.py'>
globals()['np']
<module 'numpy' from '/usr/local/lib/python3.6/site-packages/numpy/__init__.py'>


# In[ ]:




