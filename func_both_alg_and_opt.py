"""
This script provides functions for the purpose of illustrating the algorithms detailed in

Calmon, L., Colosi, E., Bassignana, G., Barrat, A. and Colizza, V.
‘Preserving friendships in school contacts: an algorithm to construct
synthetic temporal networks for epidemic modelling’

(c) the authors


The present functions are used in the following files:
    
1-friendship-based.py
2-class-mixing.py
3-optimisation.ipynb
4-build-contact-sequences.py


Please cite the following references when using this code:

[1] L. Calmon, E. Colosi, G. Bassignana, A. Barrat, and V. Colizza,
‘Preserving friendships in school contacts: an algorithm to construct
synthetic temporal networks for epidemic modelling’

[2] R. Mastrandrea, J. Fournet, A. Barrat,
Contact patterns in a high school: a comparison between data collected using wearable sensors, contact diaries and friendship surveys.
PLoS ONE 10(9): e0136497 (2015)

"""



import networkx as nx
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def intervals_15min(s):
    """"
   input:
       s = timestamp in seconds over the day 
   output: the timestamp in steps of 15 minutes.
   """

    s_in_15min = 60*15
    interval=s//s_in_15min
    return interval

def build_classes_subgraphs(w_step, classes, dailynets):
    """
    Computes a dictionary of subgraphs with links within each class, and across each pair of classes.
    
    inputs: 
        w_step = the resolution, always set to 20s
        classes = dictionary containing the list of student ID in each class keyed by class.
        dailynets = dictionary of empirical daily contact networks keyed by day.
    
    
    outputs:
        graphclasses = dictionary of subgraphs keyed by day, then by class and pair of classes. 
        Entries of the type graphclasses[d][(cl)] includes a network of all links within class cl on day d.
        Entries of the type graphclasses[d][(cl1, cl2)] includes a network of all links involving a node in class cl1 and cl2 on day d.
    """
    
    days_selected = sorted(list(dailynets.keys()))

    listclasses = sorted(list(set(classes.values())))
    composition_classes = dict()
    for cl in listclasses: 
        composition_classes[cl] = set()
    for n in classes.keys():
        composition_classes[classes[n]].add(n)

    graphclasses = {}
    # graphclasses_betw = {}
    for day in days_selected:
        graphclasses[day] = {}
        # graphclasses_betw[day] = {}
        for cl in listclasses:
            graphclasses[day][cl] = dailynets[day].subgraph(composition_classes[cl]).copy()
            for cl1 in listclasses:
                if cl > cl1:
                    # graphclasses_betw[day][(cl,cl1)] = nx.Graph()
                    graphclasses[day][(cl,cl1)] = nx.Graph()
    
    for day in days_selected:
        for e in dailynets[day].edges():
            if classes[e[0]] != classes[e[1]]:
                w = dailynets[day].get_edge_data(e[0],e[1])['weight']
                tl = dailynets[day].get_edge_data(e[0],e[1])['timeline']

                if len(tl) != (w/w_step):
                    print('ERROR!!!!!! ', e[0], e[1], w, tl)

                if classes[e[0]] > classes[e[1]]:
                    cl = classes[e[0]]
                    cl1 = classes[e[1]]
                else:
                    cl1 = classes[e[0]]
                    cl = classes[e[1]]
                graphclasses[day][(cl,cl1)].add_edge(e[0], e[1], weight=w, timeline=tl)

    return graphclasses


def include_repeated_and_random_edges_triangles_ptrv1(\
    Erepeated, Gemp, list_cl, lw, P, f, p_tr, w_step, classes):
    
    """
    Creates a synthetic network between two classes or within class according to the friendship based approach.
    
    inputs: 
        Erepeated = repeated contact links and their weight + timelines keyed by day, then by class and pair of classes. Each entry is of the form [n1,n2, (w(day1), w(day2)...), (tl(day1), tl(day2)...)]
        Gemp = list, with first element the dictionary of emprical contacts in classes subgraph, and second element the empirical contact network on the base day copied.
        list_cl = [cl] for contacts to be created within cl, or [cl1, cl2] for contacts to be created between classes cl1 and cl2.
        lw = list of tuples of the form (w, tl) of all empirical contact links within the class cl or between classes cl1 and cl2.
        P =  value of p
        f = value of f
        p_tr = value of p_tr
        w_step = resolution, must be 20s
        classes = dictionary containing the list of student ID in each class keyed by class.
        
    outputs:
        G = synthetic contact network between students of class cl or between students of cl1 and cl2.
    """
    
    
    if len(list_cl) == 1: #to generate links within a class
        Gemp_day = Gemp[0][list_cl[0]] #daily net for the class
        
        if p_tr>0: #computes transitivity of the empirical network
            transi_emp = nx.transitivity(Gemp_day)
        
        Gemp_day_ci = Gemp[0][list_cl[0]]
        Gemp_day_cj = Gemp[0][list_cl[0]]
        Erepeated = Erepeated[list_cl[0]] #repeated contact links within the class we are creating contacts for
        flagBetw = False
        
    elif len(list_cl) == 2:
        Gemp_day = Gemp[0][(list_cl[0], list_cl[1])] #daily net for pair of classes
        if p_tr>0:
            transi_emp = nx.transitivity(Gemp_day)

        Gemp_day_ci = Gemp[0][list_cl[0]] #daily net of class 1
        Gemp_day_cj = Gemp[0][list_cl[1]] #daily net of class 2
        Erepeated = Erepeated[(list_cl[0], list_cl[1])] #repeated contact links between the classes
        flagBetw = True
    else:
        print('ERROR!!!!! list_cl: ', list_cl)

    G = nx.Graph() #network that will contain contact links
    if Gemp_day.number_of_edges() > 0:
        
        G.add_nodes_from( Gemp_day ) #add empirical nodes present on the day we are making a copy from.
        
        ##### 1. INCLUDE FRIENDSHIPS
        if len(Erepeated) > 0 :
            for edge in Erepeated: #loop over the repeated links
                if random.uniform(0,1) < f: #Include a fraction f of friendships on the base day.
                    if (edge[0] not in classes.keys()) | (edge[1] not in classes.keys()):
                        print('ERROR REPEATED !!!!!! i = %d, j = %d'%(edge[0], edge[1]))
                        print('classes.keys(): ', sorted(list(classes.keys())))

                    if (edge[0] in Gemp_day_ci.nodes()) & (edge[1] in Gemp_day_cj.nodes()):
                        i = edge[0]
                        j = edge[1]
                    elif (edge[1] in Gemp_day_ci.nodes()) & (edge[0] in Gemp_day_cj.nodes()):
                        i = edge[1]
                        j = edge[0]
                    else:
                        print('ERROR!!!!!!! Not in Gemp REP')

                    w, tl = make_timeline(edge, w_step) #generate a timeline for that repeated link selected.
                    G.add_edge(i, j, weight=w, timeline=tl)
        
                    
        ### 2. REACH EMPIRICAL NUMBER OF LINKS WITH RANDOM AND EMPIRICAL LINKS
        ex = 0
        nex = 0
        while G.number_of_edges() < Gemp_day.number_of_edges() : #loop until the number of links is as empirically observed.
            
            #LINK ADDED FROM EMPIRICAL LINKS
            if random.uniform(0,1) < P: #with proba p, pick a link that exists in the empirical contacts.
                esiste = True
                target = False #flag that will turn to true if the link is validated.
                if random.uniform(0,1) < p_tr: #CHECK TRANSITIVITY.
                    list_v, list_tr, list_v_all, list_tr_all = return_list_v_tr(Gemp_day, G) #return lists of links that have effect on transitivity.
                    if (nx.transitivity(G) < transi_emp): #NEED TO INCREASE TRANSIVITY
                        if len(list_tr) !=0: #continue only if it is possible to increase transitivity using empirical links.
                            e = random.choice(list_tr) #randomly picked link.
                            target= True
                            if G.has_edge(e[0], e[1]):
                                print('ERROORR')
                                
                    elif (nx.transitivity(G) > transi_emp): #NEED TO DECREASE TRANSITIVITY
                        if len(list_v) !=0: #if it is possible with empirical links: pink a link
                            e = random.choice(list_v)
                            target = True
                            if G.has_edge(e[0], e[1]):
                                print('ERROORR')
                                        
                while target == False: # case transitivities are equal or empty lists: pick a link at random regardless of transitivity
                    e = random.choice(list(Gemp_day.edges()))
                    if G.has_edge(e[0], e[1]) == False: #check it does not exist
                        target = True
                  
                if (e[0] in Gemp_day_ci.nodes()) & (e[1] in Gemp_day_cj.nodes()):
                    i = e[0]
                    j = e[1]
                elif (e[1] in Gemp_day_ci.nodes()) & (e[0] in Gemp_day_cj.nodes()):
                    i = e[1]
                    j = e[0]
                else:
                    print('ERROR!!!!!!! ')
            
            #LINK ADDED RANDOMLY
            else: #with proba 1-p, pick instead a link at random within the class or between the two classes
                esiste = False 
                target = False 
               
                if random.uniform(0,1) < p_tr: #TRANSITIVITY CHECK. 
                
                #SAME PROCESS AS ABOVE BUT USING ALL POSSIBLE PAIRS REGARDLESS OF PRESENCE IN EMPIRICAL CONTACTS.
                    list_v, list_tr, list_v_all, list_tr_all = return_list_v_tr(Gemp_day, G)
                    if (nx.transitivity(G) < transi_emp):
                        if len(list_tr_all) !=0:
                            e = random.choice(list_tr_all)
                            target= True
                            if G.has_edge(e[0], e[1]):
                                print('ERROORR')
                                
                    elif (nx.transitivity(G) > transi_emp):
                        if len(list_v_all) !=0:
                            e = random.choice(list_v_all)
                            target = True
                            if G.has_edge(e[0], e[1]):
                                print('ERROORR')
                                   
                while target == False: # case transitivities are equal or empty lists: pick randomly any pair.
                    e = [random.choice(list(Gemp_day_ci.nodes())), random.choice(list(Gemp_day_cj.nodes()))]
                    
                    if G.has_edge(e[0],e[1]) == False and e[0]!=e[1]: # check link is not there, and it is not a self link.
                        target = True
                
                if (e[0] in Gemp_day_ci.nodes()) & (e[1] in Gemp_day_cj.nodes()):
                    i = e[0]
                    j = e[1]
                elif (e[1] in Gemp_day_ci.nodes()) & (e[0] in Gemp_day_cj.nodes()):
                    i = e[1]
                    j = e[0]
                else:
                    print('ERROR!!!!!!! ')
                    
                if flagBetw == True: #make sure if between classes, the two nodes are from different classes (this should never be the case from the design of the code)
                    if classes[i] == classes[j]:
                        raise KeyboardInterrupt()
                        continue 
                    
            #CHECKS
            if target == False: #should also not be the case.
                print('false target')
                raise KeyboardInterrupt()
            
            if G.has_edge(i,j): #link already added from before, should not be the case from the design of the code
                raise KeyboardInterrupt()
                continue

            if (i not in classes.keys()) | (j not in classes.keys()):
                print('ERROR!!!!!! i = %d, j = %d'%(i,j))
                print('classes.keys(): ', sorted(list(classes.keys())))
                    
                    
            if (i not in Gemp_day_ci.nodes()) | (j not in Gemp_day_cj.nodes()):
                print('WHYYYYYYYYYYYYY nonrep')

            #SELECT A WEIGHT AND TIMELINE FOR THE LINK i, j
            ww = random.choice(lw) 
            tl = ww[1]
            w  = ww[0]
                
            if esiste == True:
                ex = ex + 1
            else:
                nex = nex + 1
            
            G.add_edge(i, j, weight = w, timeline = tl) #add the link to the synthetic network
                                                        #this process is repeated until the right number of links are addded.

            if flagBetw == True:
                if classes[i] == classes[j]:
                    print('ERROR!!!!!! ', classes[i])
                # print('\n',classes[i], classes[j])

            if (i  not in classes.keys()) | (j not in classes.keys()):
                print('ERROR!!!!!! i = %d, j = %d'%(i,j))
                print('classes.keys(): ', sorted(list(classes.keys())))
                
    return G



def make_timeline(edge, w_step):#compute timelines for a link that is repeated over multiple days.
    """
    creates timelines for the synthetic link
    
    inputs:
        edge = list of the format [n1, n2, (w(day1), w(day2)...), (tl(day1), tl(day2)...)]
        w_step = resolution, always 20s
        
    outputs:
        w = synthetic weight
        tl = synthetic timeline
    """
    w1 = int(np.min(edge[2]) / w_step )
    w2 = int(np.max(edge[2]) / w_step )
    
    w = random.randint(w1, w2)*w_step #synthetic weight selected randomly between min and max observed for that link on different days
    
    timestamps = set()
    for ii in range(len(edge[3])): #edge[3] = [timestamps[day 1], timestamps[day 2]];
        timestamps.update(edge[3][ii]) #add timestamps over the days
    
    im = edge[2].index(np.min(edge[2])) #index of the minimum weight/tl
    
    tl = set() #synthetic timeline to be generated.
    
    tl.update(edge[3][im]) #start from empirical timeline corresponding to min weight
    
    while len(tl) < w/w_step: #add timestamps until they correspond to the weight selected
        timestamps = timestamps.difference(tl) #list of timestamps excluding the ones already present in the set
        t = random.choice(list(timestamps))
        tl.add(t) 
    tl = sorted(list(tl)) #sorted timeline
    
    if len(tl)*w_step != w:
       print('ERROR!!', len(tl)*w_step,w)
    return w, tl


def return_list_v_tr(Gemp, G):
    """
    Selects links from the empirical set, and from all possible pairs based on their effect on transitivity if they were added to the synthetic network.
    
    inputs:
        Gemp = empirical network within the class or between the pair of classes considered
        G = synthetic network
        
    outputs:
        list_v: list of links from the empirical network that open a triangle (decrease transitivity) -- excluding links that already exist in the copy
        list_tr: list of links from the empirical network that close a triangle (increase transitivity) -- excluding links that already exist in the copy
        list_v_all: list of all possible pairs of nodes that open a triangle (decrease transitivity) -- excluding links that already exist in the copy
        list_tr_all: list of all possible pairs of nodes that close a triangle (increase transitivity) -- excluding links that already exist in the copy
    """
    
    list_v = []
    list_tr = []
    list_v_all = []
    list_tr_all = []
    for ii in Gemp.nodes():
        
        neigh_ii = [n for n in G[ii]]
        neigh_emp = [n for n in Gemp[ii]]
        for jj in neigh_emp:
            if jj in neigh_ii: #link (ii, jj) already exists in the copy
                continue
            if jj == ii:
                continue
            
            if jj not in neigh_ii:
                neigh_jj =  [n for n in G[jj]]
                if set(neigh_ii).intersection(set(neigh_jj)) != set(): #link (ii, jj) creates a triangle in the copy
                    list_tr.append((ii,jj))
                    list_tr_all.append((ii,jj))
                else: #link (ii, jj) does not create a triangle in the copy
                    list_v.append((ii,jj))
                    list_v_all.append((ii,jj))
                    
        for jj in [n for n in nx.non_neighbors(Gemp, ii)]:
            if jj in neigh_ii: #link (ii, jj) already exists in the copy
                continue
            if jj == ii:
                continue
            
            if jj not in neigh_ii:
                neigh_jj = [n for n in G[jj]]
                if set(neigh_ii).intersection(set(neigh_jj)) != set(): #link (ii, jj) creates a triangle in the copy
                    list_tr_all.append((ii,jj))
                else: #link (ii, jj) does not create a triangle in the copy
                    list_v_all.append((ii,jj))
                    
    for (ii,jj) in list_v:
        if G.has_edge(ii,jj):
            print('ERROR v')
    for (ii,jj) in list_v_all:
        if G.has_edge(ii,jj):
            print('ERROR vall')
    for (ii,jj) in list_tr_all:
        if G.has_edge(ii,jj):
            print('ERROR tr all') 
    for (ii,jj) in list_tr:
        if G.has_edge(ii,jj):
            print('ERROR tr')
    return list_v, list_tr, list_v_all, list_tr_all



def weight_correction(Gemp, Gsyn, w_step, TF, TOL):
    """
    Corrects weights and timelines in the synthetic networks to be within a tolerance of the empirical ones.
    
    inputs:
        Gemp = daily empirical contact network in a class or between a pair of classes
        Gsyn = corresponding synthetic contact network to be adjusted
        w_step = resolution of the timelines, 20s
        TF = Final time of interaction on the day
        TOL = tolerance (fraction) within which to keep the synthetic total interaction time.
        
        
    ouputs:
        Gsyn = synthetic network with corrected weights and timelines
        ini_diff = initial difference in weights before correction
    """
    lencutoff = 1 #threshold in nb of timestamps above which a link weight and tl can be corrected
    
    emp = 0 # will be the empirical total weight withn class or between class
    for e in Gemp.edges():
        emp = emp + Gemp.get_edge_data(e[0],e[1])['weight']

    test = 0 # will be the synthetic total weight withn class or between class
    for e in Gsyn.edges():
        test = test + Gsyn.get_edge_data(e[0],e[1])['weight']
    
    flagContinue = True
    ini_diff = test - emp #initial difference
    
    ##EXCESS WEIGHT
    while (test > (emp * (1+TOL))) & (flagContinue == True):
        ## weights of synthetic links
        a = list(nx.get_edge_attributes(Gsyn,'weight').values()) #values of all link weights
        if a == [w_step]*len(a):
            flagContinue = False
        else:
            #### PREFERENCE
            a = [] #pairs of nodes that are in contact for more than w_step
            b = [] #their time in contact 
            for edge in Gsyn.edges(): #loop over links
                curr_w = Gsyn.get_edge_data(edge[0],edge[1])['weight'] #weight of that link
                # print('curr_w = ', curr_w)
                if curr_w > (w_step*lencutoff):
                    a.append([edge[0], edge[1]])
                    b.append(curr_w)

            if len(b) == 1: #if only one link: no random selection
                e = a[0]

            else: #random selection with probability depending on the link weight
                c = []
                for i in b:
                    c.append(i/np.sum(b)) #fraction of total interaction time of each link
                
                c = np.asarray(c).astype('float64')
                i = np.random.choice(range(len(a)), size=1, replace=True, p=c) #chooses amongst all links, with higher-proba links that interact longer.
                i = i[0]
                e = a[i] #corresponding link
            
            #current timeline of the link chosen in the
            w = Gsyn.get_edge_data(e[0],e[1])['weight']
            tl = Gsyn.get_edge_data(e[0],e[1])['timeline']        
            if len(tl) > lencutoff:
                t = random.choice(tl) #choose a timestamp at random
                tl1 = tl[:]
                tl1.remove(t)
                Gsyn.add_edge(e[0], e[1], weight=w-w_step, timeline=tl1) #replace the interaction timeline with 1 less timestamp
                test -= w_step #update current synthetic weight

    ##INSUFFICIENT WEIGHT: work similarly
    while (test < (emp * (1-TOL))):
        a = []
        b = []
        
        # Select all links with their weight 
        for edge in Gsyn.edges():
            a.append([edge[0], edge[1]])
            b.append(Gsyn.get_edge_data(edge[0],edge[1])['weight'])

        c= []
        for i in b:
            c.append(i/np.sum(b))

        c = np.asarray(c).astype('float64')
       
        #### c = proportion of total time taken by each link --> proba to pick a link to increase its duration
        i = np.random.choice(range(len(a)), size=1, replace=True, p=c)
        i = i[0]
        e = a[i] #link selected
        w = Gsyn.get_edge_data(e[0],e[1])['weight']
        tl = Gsyn.get_edge_data(e[0],e[1])['timeline']        

        t = random.choice(tl) #timestamp selected
        if ((t+w_step) not in tl) & ((t+w_step) <= TF):
            tl1 = tl + [t+w_step] #extend an existing interaction by w_step
            Gsyn.add_edge(e[0], e[1], weight=w+w_step, timeline=tl1)
            test += w_step

    return Gsyn, ini_diff



def local_cosine_sim(i,g1,g2, w = 'weight'):
    """
    computes the local cosine similarity of the node i between two networks,,
    
    inputs: 
        i = node
        g1, g2: two networks of contacts (eg of different days)
        classes: dict of classes keyed by node
        w : str attribute of the nodes to take cosine similarity with
    
    output: local cosine similarity of node i in g1 and g2
    """
    if i in g1.nodes() and i in g2.nodes(): #node must be in both networks
        neigh1 = set(g1.neighbors(i)).intersection(g2.nodes()) #neighbours of node i in g1 that are also in g2
        neigh2 = set(g2.neighbors(i)).intersection(g1.nodes()) #neighbours of node i in g2 that are also in g1
        if len(neigh1)*len(neigh2) > 0: #both neighbourhoods must not be empty
            norm1 = math.sqrt(sum([g1.get_edge_data(i,j)[w]**2 for j in neigh1]))
            norm2 = math.sqrt(sum([g2.get_edge_data(i,j)[w]**2 for j in neigh2]))
            numerator = sum([g1.get_edge_data(i,j)[w]*g2.get_edge_data(i,j)[w] 
                         for j in neigh1.intersection(neigh2)])
        else:
            norm1, norm2, numerator = 1, 1, 0
    else:
        norm1, norm2, numerator = 1, 1, 0
    return numerator/(norm1*norm2)

def compute_LCS_copies_series(dizionario, classes, coppie_giorni, copie):
    """
    compute the LCS for the nodes, concatenating values obtained for all pairs of days, over the set of copies
    
    inputs: 
        dizionario = dictionary of synthetic contacts generated with one triplet of parameters, keyed by day and copy.
        classes = dict of classes keyed by node
        coppie_giorni = list of pairs of base days between which to take the LCS
        copie = list of copies ID (e.g. [1,2,3,4,5], if there are 5 copies per day)
    
    output: local cosine similarity values of all nodes, over all pairs of days, for each series of copy 1, copy 2 etc..
    """
    
    listcl = sorted(list(set(classes.values())))
    teachernodes = set()
    if 'Teachers' in listcl:
        for n in classes.keys():
            if classes[n] == 'Teachers':
                teachernodes.add(n)
            
    lcs = []
    for copy in copie:
        for days in coppie_giorni:
            g1 = dizionario[days[0]][copy]
            g2 = dizionario[days[1]][copy]
            all_nodes = set(g1).intersection(set(g2))
            
            studentnodes = all_nodes.copy()
            studentnodes = studentnodes.difference(teachernodes)

            g1nt = g1.copy()
            g2nt = g2.copy()
            g1nt = g1nt.subgraph(studentnodes)
            g2nt = g2nt.subgraph(studentnodes)
            
            for n in all_nodes:
                lcs.append(local_cosine_sim(n,g1,g2))
                
    return lcs


def get_A_all(parameters, paramtoget, paramx, paramy, df):
    """
    returns the matrix of JS from the dataframe, in a dictionary keyed by first a fixed parameter.
    
    inputs:
        parameters = dictionaries of parameter values keyed by their label
        paramtoget = label of the fixed parameter
        paramx = parameter on the x axis 
        paramy = parameter on the y axis
        df = dataframe with the JS entries per parameters
        
        
    outputs:
        A_all = matrix of JS from the dataframe, in a dictionary keyed by the value of the fixed parameter.
    """
    A_all = dict()
    for x in parameters[paramtoget]: # value of the fixed parameter
        A_all[x] = np.zeros((len(parameters[paramy]),len(parameters[paramx])))
        i1 = 0
        for i in np.flipud(range(len(parameters[paramy]))):
            for j in range(len(parameters[paramx])):
                t = df[(df['rep']==2) & (df[paramx]==parameters[paramx][j]) & (df[paramy]==parameters[paramy][i])\
                       & (df[paramtoget]==x)]
                t = t.drop_duplicates()
                A_all[x][i1][j] = float(t.JS)
            i1+=1
    return A_all


def plot_2d(A, paramfixed, paramx, paramy, parameters, title, dict_lab, saving_figure = False):
    """
    plots a series of grids from the output of get_A_all
    
    inputs:
        A = output of get_A_all
        paramfixed = label of the fixed parameter
        paramx = parameter on the x axis 
        paramy = parameter on the y axis
        title = title of the plot
        dict_lab = dictionary of labels for the x and y axis keyed by the parameters.
        saving_figure = Flag of whether the figure is to be saved
    """
    
    x = range(1,len(parameters[paramx])+1)
    if paramx == 'tr':
        xx = ['%.2f'%(i) for i in parameters[paramx]]
    else:
        xx = ['%.1f'%(i) for i in parameters[paramx]]
    
    y = range(1,len(parameters[paramy])+1)
    if paramy == 'tr':
        yy = ['%.2f'%(i) for i in np.flipud(parameters[paramy])]
      
    else: 
        yy = ['%.1f'%(i) for i in np.flipud(parameters[paramy])]
    for i in parameters[paramfixed]:
        fig, ax = plt.subplots()
        im = ax.imshow(A[i], cmap=mpl.cm.get_cmap('copper').reversed(), vmin=0, vmax=0.5)
        
        ax.set_title('%s = %s, %s'%(paramfixed,'%.2f'%(i),title))

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(x)))
        ax.set_yticks(np.arange(len(y)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(xx)
        ax.set_yticklabels(yy)
        ax.spines[:].set_color('w')
        ax.set_xticks(np.arange(A[i].shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(A[i].shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), #rotation=45, ha="right",
                 rotation_mode="anchor")

     #   Loop over data dimensions and create text annotations.
        for ii in range(len(y)):
            for jj in range(len(x)):
                text = ax.text(jj, ii, '%.3f'%round(A[i][ii, jj], 3),
                           ha="center", color = 'w', va="center", fontsize=8)
                
        plt.xlabel(dict_lab[paramx])
        plt.ylabel(dict_lab[paramy])
        
        axins = inset_axes(ax,
                        width="5%",  
                        height="100%",
                        loc='center right',
                        borderpad=-2
                       )

        cbar = fig.colorbar(im, cax=axins)#, extend = 'max')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Jensen-Shannon distance', rotation=270)
        
        if saving_figure == True:
            plt.savefig('JS_%s%s.pdf'%(paramfixed,'%.2f'%(i)), format='pdf', bbox_inches = "tight", dpi = 2200)
        plt.show()