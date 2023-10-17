#!/usr/bin/env python
# coding: utf-8

# <h1 style = "font-family: Comic Sans MS;background-color:#CD5C5C	"> Exploratory Data Analysis - Terrorism </h1>
# 
# # Objectives:
# 
# 1. **Determine geographical regions or countries that have experienced the highest frequency and severity of terrorist   incidents.**
# 
# 2. **Investigate trends in terrorist activities over time, including yearly, monthly, and daily patterns.** 
# 
# 3. **Understand how the frequency and nature of attacks have evolved over the years.**
# 
# 4. **Create profiles of the most active and dangerous terrorist groups, including their tactics, targets, and geographical presence.**
# 
# 5. **Analyze the types of targets that terrorists choose, such as civilians, military, government, or infrastructure.**
# 
# 6. **Examine the methods employed in terrorist attacks, such as bombings, shootings, hijackings, and kidnappings.**
# 
# 7. **Determine which methods are most frequently used and how they vary by region and time.**
# 
# 
# 
# ### Submitted by : - Sanket Madavi
# 
# ### Role: Data Science Intern
# 

# # Library Importing

# In[1]:


# This Python 3 environment is equipped with numerous useful analytics libraries that are readily available. 
# For instance, we have access to a variety of helpful packages for data analysis and manipulation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# # Data Loading & Understanding

# In[2]:


#"Let's bring in our dataset and start by examining the fundamental aspects."

Terror = pd.read_csv('D:\Disk F\My Stuff\Internships\Coders Cave\Phase 1\Terrorism.csv', encoding='ISO-8859-1')
Terror


# In[3]:


Terror.columns


# In[4]:


Terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                       'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                       'weaptype1_txt':'Weapon_type','weapsubtype1_txt':'Weapon_subtype', 'motive':'Motive'},inplace=True)


# In[5]:


# I am selectively extracting and retaining essential data from the entire dataset for future processing.

Terrorism = Terror[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed',
               'Wounded','Target','Summary','Group','Target_type','Weapon_type','Weapon_subtype','Motive']]


# In[6]:


Terrorism.head(10)


# In[7]:


Terrorism.dropna(inplace = True)


# In[8]:


Terrorism.drop_duplicates(inplace = True)


# In[9]:


Terrorism.shape


# In[10]:


Terrorism.info()


# In[11]:


Terrorism.describe()


# In[12]:


Terrorism.isna().sum()


# In[13]:


Terrorism.duplicated().sum()


# # Check the maximum and minimum values in each feature across the dataset
# 

# In[14]:


# Max Parematers

print("Year experienced the highest no. of attacks:",Terrorism.Year.value_counts().idxmax())

print("Month in which highest no. of attacks occured:",Terrorism.Month.value_counts().idxmax())

print("The Day with the highest no. of attacks:",Terrorism.Day.value_counts().idxmax())

print("Nation experienced the highest no. of attacks:",Terrorism.Country.value_counts().idxmax())

print("State experienced the highest no. of attacks:",Terrorism.state.value_counts().idxmax())

print("Region experiencing the highest frequency of attacks:",Terrorism.Region.value_counts().idxmax())

print("City experienced the highest no. of attacks:",Terrorism.city.value_counts().index[1]) #as first entry is 'unknown'

print("Most common Attack Types:",Terrorism.AttackType.value_counts().idxmax())

print("Major Target Types:",Terrorism.Target.value_counts().idxmax())

print("Organisation conducted the highest no. of attacks:",Terrorism.Group.value_counts().index[1])

print("Majority of Attack aimed at:",Terrorism.Target_type.value_counts().idxmax())

print("Most weapon type used for attacks:",Terrorism.Weapon_type.value_counts().index[1])

print("Commonly employed weapon subcategory for attacks:",Terrorism.Weapon_subtype.value_counts().index[1])

print("Major motive behind conducting the attack is:", Terrorism.Motive.value_counts().index[0])


# In[15]:


# Min Parematers

print("Year with the less attacks:",Terrorism.Year.value_counts().idxmin())

print("Month with the less attacks:",Terrorism.Month.value_counts().idxmin())

print("Month with the less attacks:",Terrorism.Day.value_counts().idxmin())

print("Country with the less attacks: ",Terrorism.Country.value_counts().idxmin())

print("State with the less attacks:",Terrorism.state.value_counts().idxmin())

print("Region with the less attacks:",Terrorism.Region.value_counts().idxmin())

print("City experienced the least no. of attacks:",Terrorism.city.value_counts().idxmin()) #as first entry is 'unknown'

print("Fewer Attack Types:",Terrorism.AttackType.value_counts().idxmin())

print("Less Target Types:",Terrorism.Target.value_counts().idxmin())

print("Organisation conducted the least no. of attacks:",Terrorism.Group.value_counts().idxmin())

print("Less Attack carried out on:",Terrorism.Target_type.value_counts().index[-2])

print("Least weapon type used for attacks:",Terrorism.Weapon_type.value_counts().index[-1])

print("Rarely employed weapon subcategory for attacks:",Terrorism.Weapon_subtype.value_counts().index[-1])

print("Uncommon reasons for carrying out an attack is:", Terrorism.Motive.value_counts().idxmin())


# # Data Visualisation
# 

# Count the occurrences of terrorist activity in each year.

# In[16]:


plt.figure(figsize = (20,12))

Year = Terrorism['Year'].unique()
Attacks = Terrorism['Year'].value_counts().sort_index()

sns.lineplot(x = Year, y = Attacks)
plt.xticks(rotation = 50, fontsize = 10)
plt.xlabel('Attack Year', fontweight = 'bold', fontsize = 14)
plt.ylabel('Number of Attacks', fontweight = 'bold', fontsize = 14)
plt.title('Count of Attacks in each Years', fontweight = 'bold', fontsize = 16)
plt.show()


# #### The number of terrorist attacks from 1970 to 1997 remained consistently low, but starting from 1998 to 2017, there was a sharp and rapid increase in the frequency of attacks. Notably, in the years 2008, 2009, 2010, and 2011, there was a continuous surge in the number of terrorist incidents, indicating a significant period of heightened terrorist activity during that time frame.

# Describe the frequency of terrorist incidents across various attack methods.

# In[17]:


plt.figure(figsize = (16,16))

Year = Terrorism['AttackType'].unique()
Attacks = Terrorism['AttackType'].value_counts().sort_index()

palette_color = sns.color_palette('bright') 
  
plt.pie(Attacks, labels=Year, colors=palette_color, autopct='%.1f%%', startangle=45) 
   
plt.title('Types of terrorist attacks.', fontweight = 'bold')

plt.legend(Year, loc='upper right', bbox_to_anchor = (0, 0.7), fontsize=15) # location legend
    
plt.show() 


# #### In the global distribution of terrorist attack types, "Bombing/Explosion" stands out as the most prevalent method, accounting for 53.9% of incidents, followed by "Armed Assault" at 25.4%. Conversely, "Hijacking" and "Unarmed Assault" represent the least common methods, making up only 0.2% and 0.45% of attacks, respectively.

# Visualize terrorist activity in each region by year using an area plot.

# In[18]:


pd.crosstab(Terrorism.Year, Terrorism.Region).plot(kind='area',figsize=(16,7))
plt.title('Terrorist Activities by Region in each Year', fontweight = 'bold', fontsize = 14)
plt.xlabel('Attack Year', fontweight = 'bold', fontsize = 14)
plt.ylabel('Number of Attacks', fontweight = 'bold', fontsize = 14)
plt.show()


# #### The peak period for terrorist attacks occurred between 2005 and 2012. Middle East & North Africa, as well as South Asia, experienced the highest frequency of attacks, while East Asia, Central Asia, and Central America & the Caribbean had the lowest incidence of terrorist attacks between 1970 and 2017.

# Analyze the varieties of weapons employed in attacks across different regions.

# In[19]:


pd.crosstab(Terrorism.Region, Terrorism.Weapon_type).plot(kind='area',figsize=(16,7))
plt.title('Weapons type used for attacks in different Regions', fontweight = 'bold', fontsize = 14)
plt.xlabel('Attack Year', fontweight = 'bold', fontsize = 14)
plt.ylabel('Number of Attacks', fontweight = 'bold', fontsize = 14)
plt.show()


# #### In Eastern Europe, North America, South Asia, and Sub-Saharan Africa, the primary weapon used in terrorism attacks is explosives, followed by firearms. Incendiary weapons are rarely employed in these regions. Overall, explosives are consistently the weapon of choice across these top four regions, indicating a common trend in terrorist tactics.

# # Analysis
# Specific year's terrorist attacks, focusing on their occurrences and locations.

# In[20]:


Terrorism['casualities'] = Terrorism['Killed'] + Terrorism['Wounded']

Terrorism1 = Terrorism.sort_values(by='casualities',ascending=False)[0:18]


# In[21]:


heat=Terrorism1.pivot_table(index='Country',columns='Year',values='casualities')
heat.fillna(0,inplace=True)


# In[22]:


heat.head()


# Heatmap to visually represent the distribution of casualties across various countries.

# In[23]:


py.init_notebook_mode(connected=True)


colorscale = [[0, '#000BFF'], [1, '#8856a7'],  [.1, '#810f7c'],  [1, '#8856a7']]
heatmap = go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale = colorscale)
data = [heatmap]
layout = go.Layout(title='Top 10 countries that suffered the most casualties in terrorist attacks from 1998 to 2017', 
                   xaxis = dict(ticks='', nticks=10), yaxis = dict(ticks='')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)


# In[24]:


# Let's examine terrorist incidents worldwide within a specific year.

import folium
from folium.plugins import MarkerCluster 
filterYear = Terrorism['Year'] == 2008
filterData = Terrorism[filterYear] # filter data

reqFilterData = filterData.loc[:,'city':'longitude'] #We are getting the required fields
reqFilterData = reqFilterData.dropna() 
reqFilterDataList = reqFilterData.values.tolist()


# In[25]:


map = folium.Map(location = [0, 30], tiles='CartoDB positron', zoom_start=2)

# clustered marker
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for point in range(0, len(reqFilterDataList)):
    folium.Marker(location=[reqFilterDataList[point][1],reqFilterDataList[point][2]],
                  popup = reqFilterDataList[point][0]).add_to(markerCluster)
map


# #### In 2008, the majority of terrorist attacks, specifically 47%, occurred in Asia Continent. Middle East Region experienced 41% of these attacks. North America and South America joint suffered 3.8% attacks, while Europe saw 7%. The African continent, despite their current status as hotspots for conflicts and terrorism, had only recorded 0.3% terrorist attack during that year. Meanwhile, the Oceania continent experienced only 0.1.% of the total attacks, the lowest of all.

# To determine which terrorist organizations have conducted operations in each country, we'll perform a value count analysis. This analysis will reveal the most active terrorist groups in terms of carrying out attacks. We started the indexing from 1 to exclude the category labeled as 'Unknown.'

# In[26]:


Terrorism.Group.value_counts()[1:10]
Test = Terrorism[Terrorism.Group.isin(['Shining Path (SL)','Taliban','Islamic State of Iraq and the Levant (ISIL)'])]
Test.Country.unique()


# In[27]:


Terrorism_df_group = Terrorism.dropna(subset=['latitude','longitude'])
Terrorism_df_group = Terrorism_df_group.drop_duplicates(subset=['Country','Group'])
Terrorist_groups = Terrorism.Group.value_counts()[1:8].index.tolist()

Terrorism_df_group = Terrorism_df_group.loc[Terrorism_df_group.Group.isin(Terrorist_groups)]
print(Terrorism_df_group.Group.unique())


# In[28]:


map = folium.Map(location=[20, 0], tiles="CartoDB positron", zoom_start=2)
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for i in range(0,len(Terrorism_df_group)):
    folium.Marker([Terrorism_df_group.iloc[i]['latitude'],Terrorism_df_group.iloc[i]['longitude']], 
                  popup = 'Country:{}<br>Group:{}'.format(Terrorism_df_group.iloc[i]['Country'], 
                                                         Terrorism_df_group.iloc[i]['Group'])).add_to(map)
map


# In[29]:


#Number of individuals who lost their lives in a terrorist attack.
killed = Terrorism.loc[:,'Killed']
print('Total number of people killed in terror attack:', int(sum(killed.dropna())))


# In[30]:


# The total count of individuals injured in a terrorist attack.
Wounded = Terrorism.loc[:,'Wounded']
print('Total number of people wounded in terror attack:', int(sum(Wounded.dropna())))


# In[31]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size


# # "Fatalities in Terrorist Attacks by Various Parameters"

# In[32]:


Terrorism_2 = Terrorism[['Country', 'Killed']].groupby('Country').sum().sort_values('Killed', ascending=False)
Terrorism_2


# In[33]:


Terrorism_2[0:15].plot(kind='bar', figsize=(12,6), color='gray', subplots=True)
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Total People Killed', fontweight = 'bold')
plt.title('Top 15 countries with highest no. of lives lost.', fontweight ='bold')
plt.show();


# #### Iraq ranks first with 32,498 lives lost in terrorist attacks, while India is fifth, with a total of 7,803 fatalities, highlighting the significant impact of terrorism in these countries.

# In[34]:


Terrorism_3 = Terrorism[['AttackType', 'Killed']].groupby('AttackType').sum().sort_values('Killed', ascending=False)
Terrorism_3


# In[35]:


Terrorism_3[0:].plot(kind='bar', figsize=(12,6), color='plum', width=0.5)
plt.xlabel('Attack Type', fontweight = 'bold')
plt.ylabel('Total People Killed', fontweight = 'bold')
plt.show();


# #### Bombing / Explosion stands as the deadliest terrorist attack type, claiming a total of 59,390 lives, followed by Armed Assault, which resulted in 31,226 casualties. In contrast, Hijacking caused the fewest deaths, with a count of just 145.

# In[36]:


Terrorism_4 = Terrorism[['Target_type', 'Killed']].groupby('Target_type').sum().sort_values('Killed', ascending=False)
Terrorism_4


# In[37]:


Terrorism_4[0:5].plot(kind='bar', figsize=(12,6), color='red', width=0.5)
plt.xlabel('Target Type',  fontweight = 'bold')
plt.ylabel('Total People Killed',  fontweight = 'bold')
plt.title('Top 5 Target Type with highest no. of lives lost.',  fontweight = 'bold')
plt.show();


# #### The data reveals that Private Citizens were the primary target in terrorist attacks, resulting in the highest number of casualties, with a total of 39,475 lives lost. In the second position, the Police was the next most frequently targeted group, accounting for 13,480 fatalities.

# In[38]:


Terrorism_5 = Terrorism[['Month', 'Killed']].groupby('Month').sum().sort_values('Killed', ascending=True)
Terrorism_5


# In[39]:


Terrorism_5[1:].plot(kind='bar', figsize=(12,6), color='maroon', width=0.5)
plt.xlabel('Months', fontweight = 'bold')
plt.ylabel('Total People Killed', fontweight = 'bold')
plt.title('People Killed in Monthly Terrorist attacks.', fontweight = 'bold')
plt.show();


# #### In every 12-month period, the total number of lives lost in terrorist attacks consistently ranges between 7,000 - 11,000. Notably, the summer months of July and August experience the highest casualties, with counts reaching 10,319 and 9577, respectively.

# In[40]:


Terrorism_6 = Terrorism[['city', 'Killed']].groupby('city').sum().sort_values('Killed', ascending=False)
Terrorism_6


# In[41]:


Terrorism_6[0:15].plot(kind='bar', figsize=(12,6), color='purple', width=0.5)
plt.xlabel('City', fontweight = 'bold')
plt.ylabel('Total People Killed', fontweight = 'bold')
plt.title('Top 15 cities with highest no. of lives lost.', fontweight = 'bold')
plt.show();


# #### Baghdad recorded the highest number of lives lost in terrorist attacks, with 12,601 people, significantly surpassing all other cities. Meanwhile, the remaining cities reported casualties below 10,000, highlighting the stark disparity in the impact of terrorism.

# In[42]:


Terrorism_7 = Terrorism[['Region', 'Killed']].groupby('Region').sum().sort_values('Killed', ascending=False)
Terrorism_7


# In[43]:


Terrorism_7[0:].plot(kind='bar', figsize=(12,6), color='gold', width=0.5)
plt.xlabel('Region', fontweight = 'bold')
plt.ylabel('Total People Killed', fontweight = 'bold')
plt.title('Comparative Analysis of Lives Lost in Terrorism Across Global Regions.', fontweight = 'bold')
plt.show();


# #### The Middle East & North Africa region suffered the highest number of lives lost in terrorist attacks, totaling 43,801 casualties, followed by South Asia & Sub-Saharan Africa with 33,466 and 13,682 lives lost, respectively. In contrast, the Australia & Oceania region had the lowest casualty count, with just 15 lives lost.

# In[44]:


Terrorism_8 = Terrorism[['Group', 'Killed']].groupby('Group').sum().sort_values('Killed', ascending=False)
Terrorism_8


# In[45]:


Terrorism_8[1:16].plot(kind='bar', figsize=(12,6), color='lawngreen', width=0.5)
plt.xlabel('Organisations / Group', fontweight = 'bold')
plt.ylabel('Total People Killed', fontweight = 'bold')
plt.title('Top 15 Terrorist organisations responsible for highest no. casualties.', fontweight = 'bold')
plt.show();


# #### Among the terrorist groups responsible for attacks resulting in high casualties, ISIS (Islamic State of Iraq and the Levant) had the most significant impact, with 5,737 lives lost. The Taliban and TPP (Tehril-i-Taliban Pakistan) followed closely, with 5,626 and 4,134 lives lost, respectively.

# In[46]:


Report = Terrorism.groupby(['Region', 'Country', 'Group'])['Killed'].sum().reset_index()
Report.head()


# In[47]:


fig = px.sunburst(Report, path=['Region', 'Country', 'Group'], values = 'Killed',
                  title = 'Multilevel Pie-Chart of Unemployment Rate in different Areas of Indian States.', height = 800)

fig.show()


# In[ ]:




