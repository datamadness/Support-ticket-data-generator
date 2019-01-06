# -*- coding: utf-8 -*-
"""
Created on Monday June 18 17:39:08 2018

@author: DATAmadness
https://datamadness.github.io
https://github.com/datamadness/Support-ticket-data-generator
"""

#Import required packages
import numpy as np


#Generate daily ticket length distribution using non-central chi^2 distribution
#Input is:
# size - number of tickets on given day
# median_desired - desired median of the ticket length
# skewness_factor [0,1] - 0 for default chi^2 distribution; 1 for large positive skew
# k - degrees of freedom
# nonc - non-centrality (stats typically uses lambda letter)  
# min_length - minimum ticket length in minutes

def daily_tickets(size, median_desired, min_length, skewness_factor = 0, k = 4, nonc = 2):

    
    #Generate default chi^2 distribution and apply additional skewness if desired
    ticket_length = np.power(np.random.noncentral_chisquare(k,nonc,size),(1 + skewness_factor))
    
    #normalize to median
    ticket_length = ticket_length/np.median(ticket_length)
    ticket_length = (ticket_length * (median_desired-min_length)) + min_length
    ticket_length = np.floor(ticket_length)
    
    return ticket_length
