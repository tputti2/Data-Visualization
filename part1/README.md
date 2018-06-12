# CUMTD Visualization
## Group Totoro

# Data Preprocessing

Firstly, we prepare our dataset to comply with our visualization purposes. We define three functions, convertToSecond, getTravelTime, and getFrequency, and apply it to our dataset. It will take time to finishing this process since the data in the shapes table is quite big, so be patient when running this preprocessing step.

# CUMTD

To have a general idea of the Champaign-Urbana MTD routes, let us take a look at this All-Route plot.
We plotted all the routes based on the shapes data. In this visualization, the color for each route matches the name of the route. Some routes have the same name but are operated at different times. In this case, we combined them into single color in order to show the shape of the whole Champaign-Urbana MTD routes. The colors we used in the visualization are the exact same colors that CUMTD used in their routes color representation.

To visualize this we build function draw_plot, with routes and trips as the parameters. Because one route can have many trips, we filter the route that has maximum stop frequency to get the longest shape.  These functions can be reused to visualize other routes, as long as the routes and trips data that already preprocessed are provided.

# 1. Most Frequent Route in the Work Hours:

From the All-Route visualization above, we can see that almost all areas in the Urbana-Champaign are covered by CUMTD routes. This lead us to our next question, which route has the most frequent bus over the work hour (08:00 - 17:00). Since at this time, the transportation are needed the most. Therefore, in our this visualization, we want to visualize which route has the most frequent buses over the weekly schedule between work hours range (08:00 - 17:00)

In order to show only the most frequent routes, our first step is to get the 10 most frequent routes from the trips data. Then we plotted the 10 routes based on the shapes data. We also added legends to the graph, which are ordered by frequencies from highest to lowest, so that users can easily recognize which route is the most frequent one. To reduce confusion, we assigned the colors to the routes according to the route names, for example, we assigned yellow to Yellow Hopper. 

Because we defined a series of functions before plotting the graph, our visualization is reproducible. We can easily visualize another dataset with same structures using this approach. In addition, our visualization is easy to understand. Users can get clear information from the graph. However, the procedure of our approach is complex and data processing took us a lot of time. Moreover, the visualization is not informative enough. We could add an affiliated histogram to the graph showing the number of frequencies.

# 2. Bus Stops Observation

In this visualization, we observe the Bus Stops Density, which stops has most frequent buses coming for the time schedule range over a weekly schedule.

## Stop density visualization, Visualize Density of a stop based on CUMTD Weekly Schedule

We plotted the stops on a static google map to show the stop densities. When people are looking at the visualization, they can easily figure out which area has the highest stop density. In our graph, it is the center of campus. In our visualization, we counted the buses over 7 days, which makes the numbers of stop density to be very high. Therefore, we count the frequence as an average frequency over a day.

We choose viridis colo
r map to represent this visualization since it is a perceptually linear color and can provide frequency separation nicely. In addition, we visualize each scatter point with alpha=0.5 to provide transparancy. Therefore, some data point that clustered together in a spot will have strong color.

## Most Frequent Bus Stop According to Statistics

Visualization above displays all stops with their frequency. From the stops histogram, we found that the stops frequency data is skewed. Now we interested in displaying the most frequent route only by filtering the stops that has frequency above the (mean + 1 standard deviation)

The method we used for this visualization is basically the same as we used for the previous one, but we only care about the densities of the most frequent stops. In this graph, we could find many points outside the campus center, the high-density area we noticed from previous visualization. By visualizing only those stops we care about, this graph gives us a clearer picture of how densities of those stops distributed geographically. 

We initially designed the map to be dynamic. Users could zoom it in and out to see the points in different ways. We then changed it to the static one because the instruction suggests making the visualizations in this part to be static. However, visualizing it in a dynamic way could be better.

## Frequency of Bus coming to a stop over period of time (Bining)

Besides stop density, we want to give visualization about the bus frequency that stops over the top 5 most frequent stops over time (time binning)

We binned the arrival times into 30 minutes’ bins, so there are 48-time bins for a 24 hours’ length. We visualized the frequencies using stack bar chart to give comparison for each stops. We choose stack bar because for this visualization, it is better than using line plot since all stops mostly have a same values for each timeline. The strength of this visualization is its comparability. We can compare the frequencies both over time and across different stops. To improve this visualization, we could make it dynamic and interactive so that people can choose to see one or several stops instead of looking at the whole five stops at the same time.

From this plot, we can also see the trend in the CUMTD bus services, especially in these five most frequent stops. CUMTD services effectively start at 05:00 in the stops and start decreasing when the time reached 17:00. This plot also supporting our choose in the previous most frequent buses over a route plot to provide the active route over work hours.

# 3. Travel Time vs Stop Frequency and Travel Distance

In this plot, we try to give a trend plot that compares between Travel Time vs Stop Frequency and Travel Distance

The first graph compares the stop frequency with travel time, and the second one compares the travel distance with travel time. These two graphs are simple but useful. They show they trends and relationships between two factors very clearly. This visualization is good for looking at the routes as whole, but not good for looking at them separately. Although we assigned the routes with different colors, it is still hard to recognize the points and trend for a single route. To fix this problem, we can make it dynamic.

The first plot depicts the relation between Travel Time (in minutes) and Stop Frequency. 
From the plot, we can see that Travel Time has positive possibly linear relation with the Stop Frequency.

The second plot depicts the relation between Travel Time (in minutes) and Travel Distance (in miles). 
From the plot, we can see that Travel Time has positive possibly linear relation with the Travel Distance. However, different with the Stop Frequency, probably log transformation for the linear relation is better.

For this visualization Nikolaus works for the coding and Yiting works for the Write Up.

# References

Matplotlib: http://matplotlib.org

Salem: http://salem.readthedocs.io/en/latest/

mplleaflet: https://github.com/jwass/mplleaflet (We don't use this because it is buggy and slow)