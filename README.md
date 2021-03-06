# Climbing gym usage analysis

My local climbing gym publishes real-time visitor information. They introduced this data in 2020 because of capacity restrictions for obvious reasons. The data help visitors to get a rough estimate if they have to wait when entering the gym. I used the real-time visitor data quite a lot before visiting the gym. But I was curious not only about the current usage but also about the 'best' time to go climbing. So I started collecting and analyzing the visitor data. I will not disclose the actual gym or any API urls. But if your gym offers such data it should be easy to adapt.

Disclaimer: no predictive aspects are implemented yet. I look at historical data only.

Disclaimer1: I'm not a data scientist, I turn computers off and on again.

**TLDR; Scroll down for the fancy heatmap**

So first of all we need data. I just hacked a little Python script using `requests` to download the website, `pyquery` to extract the data from the html dom and `json` to save the data to a file. This script is executed by a systemd timer/cronjob every 15 minutes.

```python
#!/usr/bin/env python
# coding: utf-8

import requests
import json
import time
from pyquery import PyQuery
from pathlib import Path

# get request to visitors endpoint
response = requests.get('https://example.com/currentVisitors.php')
usage = response.content

# TODO: implement errror handling

# load response content to pyquery for easy access of dom objects
pq = PyQuery(usage)

# query for visitor data
# here we only query for the class b because the dom is very simple
visitors_raw,free_raw,waiting_raw = pq('b')

# extract visitor data
visitors_raw = visitors_raw.text.replace(' ', '')
if visitors_raw == '-':
    visitors = 0
else:
    visitors = int(visitors_raw)

# extract free data (free slots left)
free_raw = free_raw.text.replace(' ', '')
if free_raw == '-':
    free = 0
else:
    free = int(free_raw)

# extract waiting data
waiting_raw = waiting_raw.text.replace(' ', '')
if waiting_raw == '-':
    waiting = 0
else:
    waiting = int(waiting_raw)

# create file and folder structure
filename = int(time.time())
foldername = time.strftime("%Y%m%d", time.gmtime())
Path("./data/" + foldername).mkdir(parents=True, exist_ok=True)

# create and fill json object to store the data
statistics = {
    'timestamp' : filename,
    'visitors' : visitors,
    'waiting' : waiting,
    'free': free
}

# write json object to file
f = open('./data/' + foldername + "/" + str(filename) + ".json", "w")
f.write(json.dumps(statistics))
f.close()
```

Now we import some packages to work with:


```python
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import matplotlib.dates as mdates
import time
import seaborn as sns
import pandas as pd
```

Next we need a function to load the data again from the stored json files. I create a function to load the data of one complete day:


```python
def load_guests(day):
    guests = []
    with os.scandir('./data/' + day + '/') as entries:
        for entry in entries:
            with open('./data/' + day + '/' + entry.name) as json_file:
                data = json.load(json_file)
                guests.append(data)
    return guests
```


```python
load_guests('20210816')[:5] # it works!
```




    [{'timestamp': 1629120601, 'visitors': 23, 'waiting': 0, 'free': 37},
     {'timestamp': 1629119702, 'visitors': 21, 'waiting': 0, 'free': 39},
     {'timestamp': 1629117001, 'visitors': 21, 'waiting': 0, 'free': 39},
     {'timestamp': 1629101703, 'visitors': 6, 'waiting': 0, 'free': 54},
     {'timestamp': 1629111602, 'visitors': 17, 'waiting': 0, 'free': 43}]



Next we need a function to prepare the data for analyzing and plotting. As you may noticed the array of json objects is not sorted by time. And we also want to add additional data to simplify the plotting (e.g. adding a human readable timestamp).


```python
# formats timestamps to MM:HH for easy plotting later
def extract_time(dates):
    time = []
    for date in dates:
        hour = str(date.hour)
        minute = str(date.minute - (date.minute % 5)) # round to full n*5 minutes. Some scrapes are off by one minute
        if minute == "0":
            minute = "00"
        time.append( hour + ":" + minute )
    return time

def prepare_data(guests):
    
    # this may be obsolet if implemented in pandas, but the plot command needs data sorted by time
    guests_map = {}
    for guest in guests:
        guests_map[guest['timestamp']] = guest
    
    guests_map_sorted = sorted(guests_map.items()) # sort timestamps by key (key = timestamp)
    
    timestamps = [] # timestamps used for plot of single plot
    visitors = []   # visitors in glimbing gym
    waiting = []    # waiting visitors
    free = []       # free slots (max capacity - visitors)
    dates = []      # day in the format YYYYMMDD used for index in heatmap
    days = []       # day of the week used for index in heatmap
    
    # loop over all guests and fill arrays with values
    for timestamp,guest in guests_map_sorted:
        timestamps.append(guests_map[timestamp]['timestamp'])
        visitors.append(guests_map[timestamp]['visitors'])
        waiting.append(guests_map[timestamp]['waiting'])
        free.append(guests_map[timestamp]['free'])
        dates.append(dt.datetime.fromtimestamp(timestamp).strftime("%Y%m%d"))
        days.append(dt.datetime.fromtimestamp(timestamp).strftime("%a"))
        
    dateconv = np.vectorize(dt.datetime.fromtimestamp)
    timestamps = dateconv(timestamps)
    time_of_day = extract_time(timestamps)
    
    return visitors, waiting, free, timestamps, time_of_day, dates, days
```

So lets prepare the data of one single day:


```python
guests = load_guests('20210925')
visitors, waiting, free, timestamps, time_of_day, dates, days = prepare_data(guests)
```

And plot the data for a single day:


```python
fig = plt.figure(figsize=(15,4))
ax1 = plt.subplot2grid((1,1), (0,0))
plt.plot_date(timestamps, visitors,'b-', label='visitors')
plt.plot_date(timestamps, free,'g-', label='free')
plt.plot_date(timestamps, waiting,'r-', label='waiting')
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(45)
ax1.grid(True)

plt.xlabel('Time of day')
plt.ylabel('visitors')
plt.title('Climbing gym usage Monday - 2021-08-16')
plt.legend()
plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
plt.show()
```


    
![png](output_13_0.png)
    


This plot is useful to check the visitors during one single day. However we are interested in usage over the course of weeks and month. So lets load the visitor data of all available days.


```python
all_guests = []
for folder in os.listdir("./data/"):
    guests = load_guests(folder)
    all_guests = all_guests + guests
```


```python
all_guests[:5]
```




    [{'timestamp': 1633618801, 'visitors': 48, 'waiting': 0, 'free': 9952},
     {'timestamp': 1633632301, 'visitors': 79, 'waiting': 0, 'free': 9921},
     {'timestamp': 1633594502, 'visitors': 0, 'waiting': 0, 'free': 10000},
     {'timestamp': 1633612501, 'visitors': 19, 'waiting': 0, 'free': 9981},
     {'timestamp': 1633625101, 'visitors': 72, 'waiting': 0, 'free': 9928}]



We are using the same function `prepare_data` again to prepare the data for plotting. This time we don't care about the ordering and some other fields, because `pandas` handles this quite well. So the function could skip the sorting. But its easier to just reuse the existing code.


```python
visitors, waiting, free, timestamps, time_of_day, dates, days = prepare_data(all_guests)
```

Next we load the data to `pandas`. We drop the columns `dates`, `free` and `waiting` because we don't need them for the first heatmap. Later I may also use this data as well:


```python
d = {'visitors': visitors, 'time_of_day': time_of_day, 'dates': dates, 'days': days }
df = pd.DataFrame(data=d)
```


```python
df[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>visitors</th>
      <th>time_of_day</th>
      <th>dates</th>
      <th>days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>14:15</td>
      <td>20210808</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>14:30</td>
      <td>20210808</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>14:45</td>
      <td>20210808</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>15:00</td>
      <td>20210808</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39</td>
      <td>15:15</td>
      <td>20210808</td>
      <td>Sun</td>
    </tr>
  </tbody>
</table>
</div>



Now we merge the two columns `dates` and `days` to a new column `dates_day`. This helps to read the y-axis in the heatmap better and compare diffrent weekdays.


```python
df["dates_days"] = df.dates + " " + df.days
```

Next we have to pivot the data frame for easier plotting in a heatmap. We want to have data for each day as a row (`dates_days` column) and use the other columns for the time of day (`time_of_day` column which merges HH:MM into a string).


```python
df_pivot = df.pivot(index='dates_days', columns='time_of_day', values='visitors')
```


```python
df_pivot[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>time_of_day</th>
      <th>10:00</th>
      <th>10:15</th>
      <th>10:30</th>
      <th>10:45</th>
      <th>11:00</th>
      <th>11:15</th>
      <th>11:30</th>
      <th>11:45</th>
      <th>12:00</th>
      <th>12:15</th>
      <th>...</th>
      <th>19:45</th>
      <th>20:00</th>
      <th>20:15</th>
      <th>20:30</th>
      <th>20:45</th>
      <th>21:00</th>
      <th>21:15</th>
      <th>21:30</th>
      <th>21:45</th>
      <th>22:00</th>
    </tr>
    <tr>
      <th>dates_days</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20210808 Sun</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>27.0</td>
      <td>28.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20210809 Mon</th>
      <td>5.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>28.0</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.0</td>
      <td>24.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>20210810 Tue</th>
      <td>0.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>18.0</td>
      <td>24.0</td>
      <td>28.0</td>
      <td>34.0</td>
      <td>34.0</td>
      <td>33.0</td>
      <td>29.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>44.0</td>
      <td>38.0</td>
      <td>31.0</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>20210811 Wed</th>
      <td>22.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20210812 Thu</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>29.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows ?? 49 columns</p>
</div>



Due to missing error handling and testing some data is missing (`NaN`). We just fill the missing data with the maximum capacity for now. Most of the missing data comes from errors in the scraper code. The code failed when the visitor capacity was at maximum and some expected `int` values were `-` strings.


```python
df_pivot.loc[:] =  np.nan_to_num(df_pivot, nan=60)
```

Now we can plot the heatmap:


```python
fig = plt.figure(figsize=(30,30))
sns.set(font_scale=2)
heatmap = sns.heatmap(df_pivot, cmap=sns.color_palette("RdYlGn_r"), cbar_kws={'label': 'visitors'}, vmin=0, vmax=60)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=69) 
plt.xlabel('Time of day')
plt.ylabel('Day')
plt.title('Climbing gym visitors per day')
plt.show()

```


    
![png](output_30_0.png)
    


It would be also interesting to know the average usage over the course of a week. To calculate this average usage we pivot the data similar to the first pivot. But this time we use `days` column as the index. We also tell pandas to aggregate all similar rows (meaning similar days) using the `np.mean` function.


```python
df_pivot = df.pivot_table(index='days', columns='time_of_day', values='visitors', aggfunc = { 'visitors': np.mean })
```


```python
df_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>time_of_day</th>
      <th>10:00</th>
      <th>10:15</th>
      <th>10:30</th>
      <th>10:45</th>
      <th>11:00</th>
      <th>11:15</th>
      <th>11:30</th>
      <th>11:45</th>
      <th>12:00</th>
      <th>12:15</th>
      <th>...</th>
      <th>19:45</th>
      <th>20:00</th>
      <th>20:15</th>
      <th>20:30</th>
      <th>20:45</th>
      <th>21:00</th>
      <th>21:15</th>
      <th>21:30</th>
      <th>21:45</th>
      <th>22:00</th>
    </tr>
    <tr>
      <th>days</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fri</th>
      <td>1.222222</td>
      <td>5.777778</td>
      <td>13.555556</td>
      <td>22.555556</td>
      <td>23.666667</td>
      <td>25.333333</td>
      <td>28.000000</td>
      <td>28.666667</td>
      <td>29.333333</td>
      <td>22.444444</td>
      <td>...</td>
      <td>45.222222</td>
      <td>44.444444</td>
      <td>45.444444</td>
      <td>44.333333</td>
      <td>42.333333</td>
      <td>38.777778</td>
      <td>33.111111</td>
      <td>31.222222</td>
      <td>30.000</td>
      <td>18.111111</td>
    </tr>
    <tr>
      <th>Mon</th>
      <td>0.777778</td>
      <td>5.555556</td>
      <td>9.666667</td>
      <td>11.777778</td>
      <td>12.333333</td>
      <td>13.500000</td>
      <td>17.666667</td>
      <td>16.888889</td>
      <td>16.222222</td>
      <td>22.888889</td>
      <td>...</td>
      <td>45.625000</td>
      <td>43.875000</td>
      <td>41.000000</td>
      <td>38.875000</td>
      <td>35.375000</td>
      <td>33.750000</td>
      <td>31.375000</td>
      <td>28.111111</td>
      <td>26.000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>1.625000</td>
      <td>8.500000</td>
      <td>14.875000</td>
      <td>18.125000</td>
      <td>21.000000</td>
      <td>24.750000</td>
      <td>26.750000</td>
      <td>27.500000</td>
      <td>27.625000</td>
      <td>27.875000</td>
      <td>...</td>
      <td>14.625000</td>
      <td>15.000000</td>
      <td>13.875000</td>
      <td>14.875000</td>
      <td>14.750000</td>
      <td>13.500000</td>
      <td>11.250000</td>
      <td>9.250000</td>
      <td>7.250</td>
      <td>5.250000</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>1.625000</td>
      <td>5.500000</td>
      <td>12.250000</td>
      <td>19.500000</td>
      <td>22.375000</td>
      <td>28.625000</td>
      <td>34.250000</td>
      <td>36.750000</td>
      <td>38.125000</td>
      <td>39.250000</td>
      <td>...</td>
      <td>23.111111</td>
      <td>23.000000</td>
      <td>22.777778</td>
      <td>21.000000</td>
      <td>17.888889</td>
      <td>17.000000</td>
      <td>15.333333</td>
      <td>13.888889</td>
      <td>11.000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>Thu</th>
      <td>0.125000</td>
      <td>3.750000</td>
      <td>9.875000</td>
      <td>11.000000</td>
      <td>12.875000</td>
      <td>14.875000</td>
      <td>15.625000</td>
      <td>15.000000</td>
      <td>15.222222</td>
      <td>11.888889</td>
      <td>...</td>
      <td>45.428571</td>
      <td>46.250000</td>
      <td>42.142857</td>
      <td>38.571429</td>
      <td>35.375000</td>
      <td>31.875000</td>
      <td>29.000000</td>
      <td>26.625000</td>
      <td>25.125</td>
      <td>14.375000</td>
    </tr>
    <tr>
      <th>Tue</th>
      <td>8.000000</td>
      <td>14.111111</td>
      <td>18.444444</td>
      <td>22.111111</td>
      <td>25.000000</td>
      <td>27.222222</td>
      <td>30.000000</td>
      <td>31.555556</td>
      <td>32.111111</td>
      <td>31.777778</td>
      <td>...</td>
      <td>48.857143</td>
      <td>47.571429</td>
      <td>48.000000</td>
      <td>44.625000</td>
      <td>41.000000</td>
      <td>37.375000</td>
      <td>34.375000</td>
      <td>30.375000</td>
      <td>16.750</td>
      <td>13.375000</td>
    </tr>
    <tr>
      <th>Wed</th>
      <td>4.666667</td>
      <td>2.777778</td>
      <td>6.222222</td>
      <td>8.333333</td>
      <td>9.888889</td>
      <td>12.333333</td>
      <td>14.555556</td>
      <td>14.888889</td>
      <td>14.111111</td>
      <td>14.777778</td>
      <td>...</td>
      <td>55.428571</td>
      <td>54.000000</td>
      <td>53.000000</td>
      <td>51.750000</td>
      <td>47.875000</td>
      <td>43.875000</td>
      <td>35.750000</td>
      <td>33.625000</td>
      <td>29.250</td>
      <td>23.750000</td>
    </tr>
  </tbody>
</table>
<p>7 rows ?? 49 columns</p>
</div>



One last step is to adjust the order of the rows to represent a week:


```python
sorter = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df_pivot = df_pivot.loc[sorter]
```


```python
fig = plt.figure(figsize=(25,15))
sns.set(font_scale=2)
heatmap = sns.heatmap(df_pivot, cmap=sns.color_palette("RdYlGn_r"), cbar_kws={'label': 'visitors'}, vmin=0, vmax=60)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=69) 
plt.xlabel('Time of day')
plt.ylabel('Day')
plt.title('Average climbing gym visitors per day')
plt.show()
```


    
![png](output_36_0.png)
    

