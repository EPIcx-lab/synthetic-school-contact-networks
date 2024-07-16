"""
This script is published for the purpose of illustrating the friendship-based algorithm developed in 

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

This script generates a fixed number of synthetic daily contacts using the friendship-based approach. 
Parameters (f, p, p_tr) are fixed here. Optimisation is carried out in the file:
3-optimisation.ipynb

Inputs = contact data in the format ['t', 'n1', 'n2', 'c1', 'c2', 'day', 'w', 't_day', 't_all_days']
Outputs = synthetic contacts in the format ['t', 'n1', 'n2', 'c1', 'c2']


Parameters:
    
f_values = values of f to iterate over
P_values = values of p to iterate over
t_values = values of p to iterate over
TOL = weight correction tolerance

days_selected = list of base days to be used to generate copies
n_copies = numbers of copies per day and triplet of parameters to generate

"""

import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import time

import func_both_alg_and_opt

def timestamp():
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Date and time =", dt_string)
    return

start = time.time()
random.seed(2279)

nrep_values = [2] #numbers of days a link must be repeated to be considered a friendship.

#optimised values
f_values = [0.8] #non optimised values: [0, 0.2, 0.4, 0.6, 0.8, 1]
P_values = [0.4] #non optimised values: [0, 0.2, 0.4, 0.6, 0.8, 1] 
t_values = [0.75] #non optimised values:[0, 0.25, 0.5, 0.75, 1]

fPt_values = []
for f in f_values:
    for P in P_values:
        for t in t_values:
            fPt_values.append([f,P,t])

#%% SETUP

### PARAMETERS
TOL = 0.1
w_step = 20 #resolution of the empirical contacts
s_in_day = 60*60*24
s_in_15min = 60*15
str_header = ['t', 'n1', 'n2', 'c1', 'c2', 'day', 'w', 't_day', 't_all_days']
days_selected = [2,3,4,5] #number of base days selected to construct the copies.
n_copies = 10 #number of copies per day selected, and per parameter triplet.
n_start  = 1

#Directories
dir_root = ''
dir_data = dir_root + 'data_in/'
dir_out_to_def = dir_root + 'copies-friendship-based-%dd/copies-f%03d-P%03d-t%03d/'
copies = np.arange(n_start,n_copies+1)
print('\ncopies: ', copies)

###############################################################################
## LOAD EMPIRICAL DATA
orig_data = pd.read_table(dir_data+'CP-tidy-2345.csv', \
                delimiter='\t', names = str_header)
orig_data = orig_data.astype({"t":int, "n1":int, "n2":int, "c1":str, "c2":str, \
    "day":int, "w":int, "t_day":int, "t_all_days":int})

TI = np.min(orig_data.t_day)
TF = np.max(orig_data.t_day)
#print('Time : %d - %d'%(TI, TF))

listclasses = sorted(list(set(orig_data.c1).union(set(orig_data.c2))))

# build daily networks
classes = {}
composition_classes = {}
classweights = {}
for c in listclasses:
    composition_classes[c] = set()

dailynets = dict()
for day in days_selected:
    dailynets[day] = nx.Graph()

df0 = orig_data.copy()

for day in days_selected:
    df = df0[df0['day']==day]
    for row in df.itertuples():
        t = row.t_all_days
        # t = row.t_day
        i = row.n1
        j = row.n2
        w = row.w
        ci = row.c1
        cj = row.c2
        # day = row.day

        classes[i] = ci
        classes[j] = cj
        composition_classes[ci].add(i)
        composition_classes[cj].add(j)

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


# check that timelines are consistent with weights
for day in days_selected:
    for e in dailynets[day].edges():
        w = dailynets[day].get_edge_data(e[0],e[1])['weight']
        tline = dailynets[day].get_edge_data(e[0],e[1])['timeline']
        if w_step*len(tline) != w:
            print(w,w_step*len(tline)) 

#######################################################################
## NODES PRESENT ON EACH DAY
df0 = orig_data.copy()
nodes_present_on_day = dict()

for day in days_selected:
    df = df0.copy()
    df = df[df['day']==day]
    nodes_present_on_day[day] = set(df.n1).union(set(df.n2))

nodes_present_every_day = nodes_present_on_day[days_selected[0]].copy()
for day in days_selected[1:]:
    nodes_present_every_day = nodes_present_every_day.intersection(nodes_present_on_day[day])

###############################################################################
# builds subgraphs within each class, and of all links between different classes
graphclasses = func_both_alg_and_opt.build_classes_subgraphs( w_step, classes, dailynets )

###############################################################################
# number of links within class, between classes
list_w = {}
for day in days_selected:
    list_w[day] = {}
    
    for cl in listclasses:
        list_w[day][cl] = []
        for clprime in listclasses:
            if cl != clprime:
                list_w[day][(cl,clprime)] = []

for day in days_selected:
    for e in dailynets[day].edges():
        ci = classes[e[0]]
        cj = classes[e[1]]
        w = dailynets[day].get_edge_data(e[0],e[1])['weight']
        tl = dailynets[day].get_edge_data(e[0],e[1])['timeline']
        if ci == cj:
            list_w[day][ci].append((w,tl))
        else:
            list_w[day][(ci,cj)].append((w,tl))
            list_w[day][(cj,ci)].append((w,tl))    


#%%COPYING MECHANISM

Niter = len(fPt_values) #NUMBER OF TRIPLETS OF PARAMETERS
kk = 0
for nrep in nrep_values: 
    ###########################################################################
    # IDENTIFY THE FRIENDSHIPS:
    n_repeated = 0
    n_repeat_w = 0
    n_repeat_b = 0
    edges_repeated = dict() #dictionary that will include all repeated links and their characteristics, keyed by class and pair of classes
    for cl in listclasses:
        edges_repeated[cl] = []
        for cl1 in listclasses:
            if cl > cl1:
                edges_repeated[(cl,cl1)] = []

    for e in dailynets[days_selected[0]].edges():
        check_contain = np.zeros(len(days_selected))
        check_contain[0] = 1

        i = 0
        for day in days_selected[1:] :
            i = i + 1 
            # print('day ', day)
            if dailynets[day].has_edge(e[0],e[1]) == True:
                check_contain[i] = 1

        if np.sum(check_contain) > (nrep-1):

            w = []
            t = []
            i = -1
            for day in days_selected :
                i = i + 1
                if check_contain[i] == 1:
                    w.append(dailynets[day].get_edge_data(e[0],e[1])['weight'])
                    t.append(dailynets[day].get_edge_data(e[0],e[1])['timeline'])
            if len(w)>0:
                edge = (e[0], e[1], w, t)
                n_repeated +=1
                if classes[e[0]] == classes[e[1]]:
                    n_repeat_w +=1
                    edges_repeated[classes[e[0]]].append(edge)
                else:
                    n_repeat_b +=1
                    if classes[e[0]] > classes[e[1]]:
                        cl = classes[e[0]]
                        cl1 = classes[e[1]]
                    else:
                        cl1 = classes[e[0]]
                        cl = classes[e[1]]
                    edges_repeated[(cl,cl1)].append(edge)
    if n_repeated != (n_repeat_w + n_repeat_b):
        print('ERROR!!!!', n_repeated, '!=', n_repeat_w, '+',n_repeat_b)

    for cl in listclasses:
        for e in edges_repeated[cl]:
            for (w,tl) in zip(e[2],e[3]):
                if w != len(tl)*w_step:
                    print(w,len(tl)*w_step)
        for cl1 in listclasses:
            if cl > cl1:
                for e in edges_repeated[(cl,cl1)]:
                    for (w,tl) in zip(e[2],e[3]):
                        if w != len(tl)*w_step:
                            print(w,len(tl)*w_step)

    ##### Divide edges_repeated in days
    edges_repeated_day = dict() #dictionary keyed by day including repeated links (keyed by class and pair of classes) that occur on that day.
    for day in days_selected:
        edges_repeated_day[day] = dict()
        for cl in listclasses:
            edges_repeated_day[day][cl] = []
            for cl1 in listclasses:
                if cl > cl1:
                    edges_repeated_day[day][(cl,cl1)] = []

    #repeated links within class
    for day in days_selected:
        for cl in listclasses:
            for edge in edges_repeated[cl]:
                if (classes[edge[0]]!=cl) | (classes[edge[1]]!=cl):
                    print('WHYYYYYYY %s %s %s'%(cl, classes[edge[0]],classes[edge[1]]))

                if (edge[0] in nodes_present_on_day[day]) & \
                        (edge[1] in nodes_present_on_day[day]) :
                    edges_repeated_day[day][cl].append(edge)
    
    #repeated edges between pairs of classes
    for day in days_selected:
        for cl in listclasses:
            for cl1 in listclasses:
                if cl > cl1:
                    for edge in edges_repeated[(cl, cl1)]:
                        if (edge[0] in nodes_present_on_day[day]) & \
                                (edge[1] in nodes_present_on_day[day]) :
                            edges_repeated_day[day][(cl, cl1)].append(edge)

   
    ###############################################################################
    ## CREATE SYNTHETIC COPIES
    ## generate n_copies for each day, for each triplet of the parameters
    for f,P,p_tr in fPt_values: # loop over the triplets of parameters
            kk = kk + 1
            timestamp()
            print('%d / %d :\t rep = %d, f = %f, P = %f, p_tr= %f, TOL = %f'%(\
                kk, Niter, nrep, f, P, p_tr, TOL))

            dir_out = dir_out_to_def%(len(days_selected), 100*f, 100*P, 100*p_tr) #directory where copies are saved depends on parameters.
            
            for icopy in copies:
                for day in days_selected:

                        print('DAY %d, '%day, 'COPY %02d'%icopy)
                        glob_gw = nx.Graph() #network that will contain all synthetic links between and within classes.
                        
                        ### LINKS WITHIN CLASS
                        for cl in listclasses:         
                            
                            # # we create a graph with the same size as the original graph
                            glob_gw.add_nodes_from(graphclasses[day][cl]) #add nodes present on day

                            lw = list_w[day][cl][:] #list of weights and tl within the class, on the day
                            Erepeated = edges_repeated_day[day] #repeated links (friendships) of the day.
                            
                            Gemp = [graphclasses[day], dailynets[day]]
                            list_cl = [cl]
                            G = func_both_alg_and_opt.include_repeated_and_random_edges_triangles_ptrv1(\
                                Erepeated, Gemp, list_cl, lw, \
                                P, f, p_tr, w_step, classes) #creates the synthetic network within class cl.

                            glob_gw.add_edges_from( G.edges(data = True) ) # transfer the links into glob_gw.


                        ### LINKS BETWEEN CLASSES
                        ne = 0
                        for cl in listclasses:
                            for cl1 in listclasses:
                                if cl > cl1:
                                    G = nx.Graph() #network of links between cl and cl1.
                                    lw = list_w[day][(cl,cl1)][:] #weights and tl within
                                    Erepeated = edges_repeated_day[day]
                                    Gemp = [graphclasses[day], dailynets[day]]
                                    list_cl = [cl, cl1] #pair of classes between which to generate synthetic links
                                    G = func_both_alg_and_opt.include_repeated_and_random_edges_triangles_ptrv1(\
                                        Erepeated, Gemp, list_cl, lw, \
                                        P, f, p_tr, w_step, classes)

                                    glob_gw.add_nodes_from( G )
                                    glob_gw.add_edges_from( G.edges(data=True) )


                        ### WEIGHT CORRECTION
                        Gwcorrected = nx.Graph() #will become the network with corrected weight

                        dailynets_syn = dict()
                        dailynets_syn[day] = glob_gw.copy()
                        graphclasses_syn = func_both_alg_and_opt.build_classes_subgraphs( \
                                    w_step, classes, dailynets_syn ) #dictionary of classes subgraph of synthetic contacts

                        ## within class correction
                        for cl in listclasses:
                            Gemp = graphclasses[day][cl]
                            Gsyn = graphclasses_syn[day][cl]

                            Gsyn_wc, ini_diff = func_both_alg_and_opt.weight_correction(Gemp, Gsyn, w_step, TF, TOL) #weight correction

                            Gwcorrected.add_nodes_from( Gsyn_wc )
                            Gwcorrected.add_edges_from( Gsyn_wc.edges(data=True) )

                        ## between classes
                        for cl in listclasses:
                            for cl1 in listclasses:
                                if cl > cl1:
                                    Gemp = graphclasses[day][(cl,cl1)]
                                    Gsyn = graphclasses_syn[day][(cl,cl1)]

                                    Gsyn_wc, ini_diff = func_both_alg_and_opt.weight_correction(Gemp, Gsyn, w_step, TF, TOL)

                                    Gwcorrected.add_nodes_from( Gsyn_wc )
                                    Gwcorrected.add_edges_from( Gsyn_wc.edges(data=True) )

                        glob_gw = Gwcorrected.copy()
                        
                        ###########################################################
                        ###SAVE THE CONTACTS
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
                                # print(t, e[0], e[1], classes[e[0]], classes[e[1]])
                                ## ID n1 < ID n2
                                if e[0] < e[1] :
                                    n1 = e[0]
                                    n2 = e[1]
                                else:
                                    n1 = e[1]
                                    n2 = e[0]
                                wd.write('%s\t%s\t%s\t%s\t%s\n' % (t, n1, n2, classes[n1], classes[n2]))
                        wd.close()
        
###############################################################################
###############################################################################

end = time.time()
print('\nElapsed time : ', (end - start)/60, 'min\n')

timestamp()

print('\nThe end.')