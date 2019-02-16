#!/usr/bin/env python
# coding: utf-8

# In[43]:


import graphlab as gl


# # Load music Data

# In[44]:


song_data = gl.SFrame('song_data.gl/')


# # Explore data

# In[45]:


song_data.head()


# In[46]:


gl.canvas.set_target('ipynb')


# In[47]:


song_data['song'].show()


# In[48]:


len(song_data)


# ## Count number of users

# In[49]:


users = song_data['user_id'].unique()


# In[50]:


len(users)


# # Create a song recommender

# In[51]:


train_data, test_data = song_data.random_split(.8, seed = 0)


# ## Simple popularity-based recommender

# In[52]:


popularity_model = gl.popularity_recommender.create(train_data, 
                                                   user_id='user_id',
                                                   item_id = 'song')


# ## Use the popularity_model to make some predictions

# In[53]:


popularity_model.recommend(users=[users[0]])


# In[54]:


popularity_model.recommend(users=[users[1]])


# ## Build a recommender with personalization

# In[62]:


personalized_model = gl.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')


# ## Applying the personalized model to make a song recommendation

# In[56]:


personalized_model.recommend(users=[users[0]])


# In[65]:


personalized_model.recommend(users = [users[1]])


# In[67]:


personalized_model.get_similar_items(['With Or Without You - U2'])


# In[68]:


personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])


# # Quantitative Comparison between the models

# In[70]:


if graphlab.version[:3] >= "1.6":
    model_performance = gl.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
    gl.show_comparison(model_performance,[popularity_model, personalized_model])
else:
    get_ipython().run_line_magic('matplotlib', 'inline')
    model_performance = gl.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample=.05)


# In[74]:


model_performance = gl.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
gl.show_comparison(model_performance,[popularity_model, personalized_model])


# # Assignment 

# In[88]:


unique_users_kanye = song_data[song_data['artist'] ==  "Kanye West"]['user_id'].unique()


# In[89]:


len(unique_users_kanye)


# In[90]:


unique_users_foo = song_data[song_data['artist'] == "Foo Fighters"]['user_id'].unique()


# In[91]:


len(unique_users_foo)


# In[94]:


unique_users_TS = song_data[song_data['artist'] == 'Taylor Swift']['user_id'].unique()


# In[95]:


len(unique_users_TS)


# In[103]:


unique_users_LG = song_data[song_data['artist'] == 'Lady GaGa']['user_id'].unique()


# In[104]:


len(unique_users_LG)


# In[105]:


count = song_data.groupby(key_columns='artist', operations={'count': gl.aggregate.SUM('listen_count')})


# In[106]:


count.sort('count')


# In[107]:


count.sort('count', ascending= False)


# In[108]:


train,test = song_data.random_split(.8,seed=0)


# In[111]:


item_similarity_model = gl.item_similarity_recommender.create(train,
                                                       user_id="user_id",
                                                       item_id='song')


# In[114]:


subset_test_users = test['user_id'].unique()[0:10000]


# In[115]:


subset_test_users.head()


# In[116]:


subset_test_model = personalized_model.recommend(subset_test_users,k=1)


# In[117]:


song_count_total = subset_test_model.groupby(key_columns='song', operations={'count': gl.aggregate.COUNT()})


# In[118]:


song_count_total


# In[124]:


song_count_total.sort('count',ascending = True)


# In[123]:


song_count_total.sort('count',ascending = False)


# In[ ]:




