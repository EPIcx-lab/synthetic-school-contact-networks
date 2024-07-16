"""
This script is published for the purpose of illustrating the class-mixing algorithm developed in 

Calmon, L., Colosi, E., Bassignana, G., Barrat, A. and Colizza, V.
‘Preserving friendships in school contacts: an algorithm to construct
synthetic temporal networks for epidemic modelling’


(c) the authors


Empirical data loaded in this script were obtained from 
http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/.
The file provided as input was postprocessed to 
- select the four full days recorded (removing the first day)
- create the variable "t_all_days" counting time (in seconds) that resets every day at midnight.
- create a variable "t_day" counting time (in seconds) over the full data collection, initialised at midnight on day 1.
- create the variable "day" that identifies the day (2,3,4,5) over which the contacts take place.
- create the variable "w" set to the duration of each contact (=20 seconds)


Please cite the following references when using this code:

[1] L. Calmon, E. Colosi, G. Bassignana, A. Barrat, and V. Colizza,
‘Preserving friendships in school contacts: an algorithm to construct
synthetic temporal networks for epidemic modelling’

[2] R. Mastrandrea, J. Fournet, A. Barrat,
Contact patterns in a high school: a comparison between data collected using wearable sensors, contact diaries and friendship surveys.
PLoS ONE 10(9): e0136497 (2015)


This script generates a fixed number of synthetic daily contacts using the class-mixing approach.

Inputs = contact data in the format ['t', 'n1', 'n2', 'c1', 'c2', 'day', 'w', 't_day', 't_all_days']
Outputs = synthetic contacts in the format ['t', 'n1', 'n2', 'c1', 'c2']


Parameters:
days_selected = list of base days to be used to generate copies
n_copies = numbers of copies per day to generate

"""

import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import time

import func_both_alg_and_opt #script with functions included

start = time.time()
random.seed(9715)

#PARAMETERS
w_step = 20
s_in_day = 60*60*24
s_in_15min = 60*15
str_header = ['t', 'n1', 'n2', 'c1', 'c2', 'day', 'w', 't_day', 't_all_days']

days_selected = [2,3,4,5] #BASE DAYS SELECTED TO CREATE THE COPIES.
n_copies = 10 #NUMBERS OF COPIES per day
n_start  = 1

#DIRECTORY
dir_root = ''
dir_data = dir_root + 'data_in/' #input directory
dir_out = dir_root + 'copies-random-%dd/'%len(days_selected) #

######################################

copies = np.arange(n_start,n_copies+1)
print('\ncopies: ', copies)

###############################################################################
## LOAD EMPIRICAL DATA
orig_data = pd.read_table(dir_data+'CP-tidy-2345.csv', \
                delimiter='\t', names = str_header)
orig_data = orig_data.astype({"t":int, "n1":int, "n2":int, "c1":str, "c2":str, \
    "day":int, "w":int, "t_day":int, "t_all_days":int})

listclasses = sorted(list(set(orig_data.c1).union(set(orig_data.c2))))


## BUILD DAILY NETWORKS
classes = {}
composition_classes = {}
classweights = {}
for c in listclasses:
    composition_classes[c] = set()

dailynets = dict()
for day in days_selected:
    dailynets[day] = nx.Graph()
    classweights[day] = {}
    for c in listclasses:
        classweights[day][c] = 0

df0 = orig_data.copy()


for day in days_selected:
    df = df0[df0['day']==day]
    for row in df.itertuples():
        t = row.t_all_days
        i = row.n1
        j = row.n2
        w = row.w
        ci = row.c1
        cj = row.c2

        classes[i] = ci
        classes[j] = cj
        composition_classes[ci].add(i)
        composition_classes[cj].add(j)
        if ci == cj:
            classweights[day][ci] += w_step

        w0 = 0
        tline = []
        if dailynets[day].has_edge(i,j):
            w0 = dailynets[day].get_edge_data(i,j)['weight']
            tline = dailynets[day].get_edge_data(i,j)['timeline']
        tline.append(t % s_in_day)
        dailynets[day].add_edge(i,j,weight = w0+w_step,timeline = tline)
        if w_step*len(tline) != w0+w_step:
            print(w0+w_step,w_step*len(tline))
        ## REMARK THE ABOVE IS NOT TRUE FOR AGGREGATED DATA 
        ## BECAUSE THE INTERACTION MAY BE LONGER THAN w_step

#print([(c,len(composition_classes[c]),classweights[days_selected[0]][c]) for c in listclasses], '\n')

# check that timelines are consistent with weights
for day in days_selected:
    for e in dailynets[day].edges():
        w = dailynets[day].get_edge_data(e[0],e[1])['weight']
        tline = dailynets[day].get_edge_data(e[0],e[1])['timeline']
        if w_step*len(tline) != w:
            print(w,w_step*len(tline)) 

###############################################################################
# builds subgraphs within each class, and of all links between different classes
graphclasses = func_both_alg_and_opt.build_classes_subgraphs( w_step, classes, dailynets )

###############################################################################

### COLLECT INPUTS
list_w = {} #list of weights and timelines, dictionary keyed per day, then pair of classes or class
n_edges = {} #number of edges, dictionary keyed per day, then pair of classes or class
for day in days_selected:
    list_w[day] = {}
    n_edges[day] = {}
    
    for cl in listclasses:
        list_w[day][cl] = []
        n_edges[day][cl]= 0
        for clprime in listclasses:
            if cl != clprime:
                list_w[day][(cl,clprime)] = []
                n_edges[day][(cl,clprime)] = 0

for day in days_selected:
    for e in dailynets[day].edges():
        ci = classes[e[0]]
        cj = classes[e[1]]
        w = dailynets[day].get_edge_data(e[0],e[1])['weight']
        tl = dailynets[day].get_edge_data(e[0],e[1])['timeline']
        if ci == cj:
            list_w[day][ci].append((w,tl))
            n_edges[day][ci] += 1
        else:
            list_w[day][(ci,cj)].append((w,tl))
            list_w[day][(cj,ci)].append((w,tl))
            n_edges[day][(ci,cj)] += 1
            n_edges[day][(cj,ci)] += 1



###############################################################################
## CREATE SYNTHETIC COPIES
for icopy in range(1,n_copies+1):
    for day in days_selected:
    ## Generate a copy for each day in days_selected

            print('DAY %d, '%day, 'COPY %02d'%icopy)
            
            glob_gw = nx.Graph()
            between_rnd = nx.Graph()
            synthclassweights = {}
            gw = {}
            
            #SYNTHETIC CONTACTS WITHIN CLASS:
            for cl in listclasses:         
                synthclassweights[cl] = 0
                lw = list_w[day][cl][:] #link weights and timelines of that class on that day.
                # we start from a random graph with same size as original class
                g = nx.gnm_random_graph(len(composition_classes[cl]), \
                                        len(graphclasses[day][cl].edges()) ) 
                    
    
                #rename nodes
                gw[cl] = nx.Graph() #the graph with the right node labelling 
                gw[cl].add_nodes_from(graphclasses[day][cl]) #nodes added with their ID
                mapnodes = {} #mapping
                for i in g.nodes():
                    mapnodes[i] = list(composition_classes[cl])[i]
                    
                # Associate a weight and timeline of a random contact link of the same class
                for e in g.edges():
                    ww = random.choice(lw)
                    tl = ww[1] #limeline
                    w = ww[0] #weight
                    gw[cl].add_edge(mapnodes[e[0]],mapnodes[e[1]],weight = w, timeline = tl)
                    glob_gw.add_edge(mapnodes[e[0]],mapnodes[e[1]],weight = w, timeline = tl)
                    synthclassweights[cl] += w #total weight of the class, dynamically updated.
           
            
            #SYNTHETIC CONTACTS BETWEEN CLASS:
            between_rnd = nx.Graph()
            for cl in listclasses:
                for clprime in listclasses:
                    if cl >= clprime: #only consider each pair once
                        continue
                    ne = n_edges[day][(cl,clprime)] #number of contact links to be reproduced
                    lw = list_w[day][(cl,clprime)][:] #link weights and timelines on that day between these classes
                    ne_rnd = 0 #number of contact links between the pair of classes in the synthetic network
                    while ne_rnd < ne:
                        i = random.choice(list(gw[cl].nodes()))
                        j = random.choice(list(gw[clprime].nodes()))
                        ww = random.choice(lw)
                        tl = ww[1]
                        w = ww[0]
                        if glob_gw.has_edge(i,j):
                            continue
                        glob_gw.add_edge(i, j, weight = w, timeline = tl)
                        between_rnd.add_edge(i, j, weight = w, timeline = tl)
                        ne_rnd += 1

            ###################################################################
            ## SAVE SYNTHETIC COPIES
            path = dir_out + 'DAY%d/'%day
            if not os.path.exists(path):          
                os.makedirs(path)

            events = {}
            for e in glob_gw.edges():
                tl = glob_gw.get_edge_data(e[0],e[1])['timeline']
                for t in tl:
                    if t not in events:
                        events[t] = []
                    events[t].append((e[0],e[1]))
            wd =  open(path + 'dynamic_copy_%02d.csv'%icopy,'w')
            for t in sorted(events.keys()):
                for e in events[t]:

                    if e[0] < e[1] :
                        n1 = e[0]
                        n2 = e[1]
                    else:
                        n1 = e[1]
                        n2 = e[0]
                    wd.write('%s\t%s\t%s\t%s\t%s\n' % (t, n1, n2, classes[n1], classes[n2]))
            wd.close()
          
###############################################################################

end = time.time()
print('\nElapsed time : ', (end - start)/60, 'min')

print('\nThe end.')