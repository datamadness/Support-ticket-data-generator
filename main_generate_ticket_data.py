# -*- coding: utf-8 -*-
"""
Created on Monday June 18 17:39:08 2018

@author: DATAmadness
https://datamadness.github.io
https://github.com/datamadness/Support-ticket-data-generator
"""

#%%Import required packages for the data generation function
from generate_daily_data import daily_tickets
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats

#Packages for the data analysis / plots outside of the function
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% Function to generate custom support ticket data

def generate_ticket_data(
    
    #Optional input parameters
    annual_minimum = 5,                  # ~ annual daily minimum of tickets
    annual_maximum = 50,                 # ~ annual daily maximum of tickets
    busy_months = np.array([2,7]),       # typical busy montsh with high number of tickets (e.g. 2 = February)
    seasonal_factor = 1,                 # seasonal effect - 0 = none, 1 = large
    weekend_factor = 1,                  # Effect of weekends - 0 = no tickets on weekends, 1 = same as business day
    k = 4,                               # Degrees of freedom in Chi-squared distribution (logged time)
    nonc = 2,                            # non-centrality for chi-squared (logged time)
    accounts = 100,                      # Number of customer accounts
    logged_time_median_desired = 20,     # desired median of the ticket length
    logged_time_skewness_factor = 0.3,   # ticket length skewness: [0,1] - 0 for default chi^2 distribution; 1 for large positive skew
    logged_time_minimum = 5,             # minimum logged time per ticket in minutes
    fileName = 'myTicketData.csv' ):     # If string is provided, data will be saved into *csv  of given name, None if you do not want to save
    
    cols = ['ticketID','date','accountNumber','loggedTime']
    df = pd.DataFrame(columns = cols)
    
    #Generate complete ticket data for a year day-by-day
    for daynum in range(1,366):
    
        #Step 1: Calculate day distance from the closest busy month
        busy_level = np.concatenate((np.abs(365 - 15 + (busy_months * 30.4) - daynum), np.abs(daynum + 15 - (busy_months * 30.4))))
        #Step 2: Calculate maximum possible distance from busy month for normalization
        max_distance = np.max(np.ediff1d(np.sort(np.concatenate((busy_months,busy_months + 12))))*15.2)
        #Step 3: Calculate normalized busy level for current day
        busy_level = 1 - np.min(busy_level) / max_distance
        
        #Generate distribution function for number of tickets on given day for random generation
        
        #Calculate mean 
        mu = (annual_maximum - annual_minimum)/2 + seasonal_factor * (busy_level - 0.5) * ((annual_maximum - annual_minimum)/2) 
        #Calculate standard deviation
        sigma = (annual_maximum - annual_minimum) * 0.1
        #Calculate distribution fundtion
        dist = stats.truncnorm((annual_minimum - mu) / sigma, (annual_maximum - mu) / sigma, loc=mu, scale=sigma)
        #Generate number of tickets for given day
        ticket_num = int(dist.rvs())
        
        #Add generated data to the dataframe
        
        #date and ticket IDs
        if daynum == 1:
            date = dt.datetime(2018,1,1)
            ticketIDs=np.arange(1,1 + ticket_num)
        else:
            date = date + dt.timedelta(days = 1)
        #Check if date is a weekdate
        if date.weekday() > 4:
            ticket_num = int(ticket_num * weekend_factor)
            
        #get logged time for each ticket by calling the daily_tickets function
        if ticket_num > 0:
            ticket_times = daily_tickets(ticket_num, logged_time_median_desired, logged_time_minimum, logged_time_skewness_factor,k,nonc)
            ticketIDs=np.arange(ticketIDs[-1] + 1, ticketIDs[-1] + 1 + ticket_num)
            
            #Create temporary data frame with all data for single day
            tempDF = pd.DataFrame(columns = cols)
            tempDF[cols[0]] = ticketIDs
            tempDF[cols[1]] = date
            tempDF[cols[2]] = np.random.randint(1, accounts + 1, size = ticket_num)
            tempDF[cols[3]] = ticket_times
            
            #Append day data to the main data frame
            df = df.append(tempDF, ignore_index  = True)
       
    if fileName is not None:
        df.to_csv(fileName)
            
    return df

#%%Analyse the generated data

def dataDescription(ticketData):
    
    #Column data types
    print(ticketData.dtypes)
    
    #Basic statistical description of logged time
    ticketData['loggedTime'].describe()


#Plot logged time histogram + KDF
def loggedTimeHist(ticketData):
    plt.figure(figsize = (12, 7))
    sns.set_style('darkgrid')
    sns.set_context(font_scale=1.5, rc={"lines.linewidth": 2.5})
    dplot = sns.distplot(ticketData['loggedTime'], bins = 60, rug = True,
                 rug_kws={"alpha":0.11, "linewidth": 1, "height":0.1 }
                 )
    dplot.set(xlabel = 'Logged Time [minutes]', title = 'Distribution of logged time per ticket - Histogram + KDE')
    plt.savefig('logged_time_distribution_plot.png', bbox_inches='tight')
    plt.show(dplot)


#Visualize total number of opened tickets through the year  
def plotWeeklyTickets(ticketData):
    weekly_tickets = ticketData.groupby([pd.Grouper(key='date', freq='W')])['loggedTime'].count()
    
    plt.figure(figsize = (12, 7))
    wplot = sns.lineplot(data = weekly_tickets[0:len(weekly_tickets)-1])
    
    monthyearFmt = mdates.DateFormatter('%B')   
    months = mdates.MonthLocator(interval = 1)  # every month
    #weeks = mdates.WeekdayLocator(byweekday=1, interval=1, tz=None)
    wplot.xaxis.set_major_locator(months) 
    wplot.xaxis.set_major_formatter(monthyearFmt)
    plt.xticks(rotation=45)
    wplot.set(xlabel = 'Month', ylabel = 'Number of opened tickets', title = 'Total number of tickets opened each week of the year')
    plt.savefig('weekly_ticket_totals.png', bbox_inches='tight')    
    plt.show(wplot)

#%% Generate sample data and plot simple analytics

#Reset seed for reproducible results
np.random.seed(24)
myTicketData = generate_ticket_data()

dataDescription(myTicketData)
loggedTimeHist(myTicketData)
plotWeeklyTickets(myTicketData)

#%% Compare two daily ticket num plots
f, (ax1, ax2) = plt.subplots(2,figsize = (12, 7), sharex = True, sharey = True)
sns.lineplot(data = generate_ticket_data(weekend_factor=0.25).groupby([pd.Grouper(key='date', freq='D')])['loggedTime'].count(), ax = ax1)
sns.lineplot(data = generate_ticket_data(weekend_factor=1).groupby([pd.Grouper(key='date', freq='D')])['loggedTime'].count(), ax = ax2)
ax1.set_xlim([dt.date(2018, 1, 1), dt.date(2018, 2, 1)])
ax2.set_xlim([dt.date(2018, 1, 1), dt.date(2018, 2, 1)])
plt.xticks(rotation=45)
ax1.set(ylabel = '# tickets per day')
ax2.set(ylabel = '# tickets per day')
ax1.set(title = 'weekend_factor = 0.25')
ax2.set(title = 'weekend_factor = 1')
plt.savefig('weekend_factor_comparison.png', bbox_inches='tight')  
plt.show()