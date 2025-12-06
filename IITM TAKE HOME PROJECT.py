#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


import matplotlib.pyplot as plt 


# In[6]:


df = pd.read_csv("Copy of E-commerce_Customer_Retention_Dataset - E-commerce_Customer_Retention_Dataset.csv")

#Uploads our data set into the complier


# In[8]:


df.head() #To view the rows from the top of the data set #Shows the first five rows by default


# In[12]:


df.tail() #Shows last 5 rows by default from the data set


# In[15]:


df.shape  # Tells the number of rows and columns


# In[16]:


df.columns #shows the column's title


# In[21]:


df.info() #Tells us All the information about the data columns


# In[22]:


df.isnull().sum() #Checks the missing values in the data set


# In[23]:


df.describe() #Summarizes the entire statistics of the data set


# In[27]:


df['Gender'].value_counts() #counts the values stored in gender column 


# In[32]:


df['MembershipType'].value_counts() #counts the values stored in memebership column


# In[33]:


df['Region'].value_counts() #counts the values stored in region column


# In[34]:


df['IsRetained'].value_counts() #counts the values stored in retained column


# In[35]:


plt.hist(df['Age']) #represents the age value of the retained column


# In[37]:


plt.xlabel("Age") #represents the age value of x axis in the graph


# In[38]:


plt.ylabel("Count") #represents the count on the y axis in the graph


# In[39]:


plt.title("Age Distribution") #The title of graph "Age distribution"


# In[43]:


plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show() #The final presentation of the graph.


# In[45]:


plt.hist(df['AvgPurchaseValue']) #represents the average purchase value
plt.xlabel("Avg Purchase Value") #represents the avg pur. value on x axis
plt.ylabel("Count") #represents the count section on the y axis
plt.title("Average Purchase Value Distribution") #presents the title of graph "avg pur value distribution"
plt.show() #The final presentation of the graph


# In[22]:


df['IsRetained'].value_counts().plot(kind='bar')
plt.xlabel("Is Retained?")
plt.ylabel("Number of Customers")
plt.title("Retention Count")
plt.show()


# In[7]:


corr = df.corr()
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.title("Correlation Heatmap")
plt.show()


# In[8]:


df.groupby("MembershipType")["AvgPurchaseValue"].mean()


# In[9]:


df.groupby("Region")["PurchasesLastMonth"].mean()


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# In[12]:


df_knn = df.copy()


# In[13]:


le = LabelEncoder()

df_knn['Gender'] = le.fit_transform(df_knn['Gender'])
df_knn['Region'] = le.fit_transform(df_knn['Region'])


# In[14]:


X = df_knn.drop("IsRetained", axis=1)
y = df_knn["IsRetained"]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[16]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[17]:


knn.fit(X_train, y_train)


# In[18]:


y_pred = knn.predict(X_test)


# In[19]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[20]:


print(classification_report(y_test, y_pred))


# In[21]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[22]:


df_cluster = df.copy()


# In[25]:


le = LabelEncoder()

df_cluster['Gender'] = le.fit_transform(df_cluster['Gender'])
df_cluster['Region'] = le.fit_transform(df_cluster['Region'])


# In[26]:


features = df_cluster.drop(["CustomerID", "IsRetained"], axis=1)


# In[27]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[28]:


inertia_values = []

for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    inertia_values.append(km.inertia_)

plt.plot(range(1, 10), inertia_values)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method to Choose k")
plt.show()


# In[29]:


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)


# In[30]:


df_cluster['Cluster'] = clusters
df_cluster.head()


# In[31]:


plt.scatter(df_cluster['Age'], df_cluster['AvgPurchaseValue'], 
            c=df_cluster['Cluster'])
plt.xlabel("Age")
plt.ylabel("Average Purchase Value")
plt.title("Customer Clusters")
plt.show()


# In[32]:


df_cluster.groupby("Cluster").mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




