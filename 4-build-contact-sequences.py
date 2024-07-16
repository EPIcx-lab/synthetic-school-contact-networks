"""
This script is published for the purpose of illustrating how contact sequences can be built from daily contact networks
as described in 

Calmon, L., Colosi, E., Bassignana, G., Barrat, A. and Colizza, V.
‘Preserving friendships in school contacts: an algorithm to construct
synthetic temporal networks for epidemic modelling’


(c) the authors

empirical data loaded in this script can be obtained from 
http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/.
The data was postprocessed to remove the first day (only half a day recorded), ....

sample synthetic contacts loaded in this file were generated using the files CP-3-class-mixing.py and CP-2-friendship-based.py

Please cite the following references when using this code:

[1] L. Calmon, E. Colosi, G. Bassignana, A. Barrat, and V. Colizza,
‘Preserving friendships in school contacts: an algorithm to construct
synthetic temporal networks for epidemic modelling’

[2] R. Mastrandrea, J. Fournet, A. Barrat,
Contact patterns in a high school: a comparison between data collected using wearable sensors, contact diaries and friendship surveys.
PLoS ONE 10(9): e0136497 (2015)

This script arranges pre-generated synthetic or empirical daily contacts into sequences of weekdays and weekends.

Inputs = daily contacts (empirical or synthetic) in the format ['t', 'n1', 'n2', 'c1', 'c2']
Outputs = contacts, aggregated in 15 mins steps, and covering a long period of time in format ['day','step', 'n1', 'n2', 'c1', 'c2', 'w']

Parameters:
 
type_copy = string, one of 'looped', 'class_mixing', 'friendship_based' depending on the contacts to be loaded and arranged
fPt_values = [f, p, p_tr] optimised parameters to use if friendship_based.
days_selected = list of base days to be used to generate copies or to loop over
n_copies = numbers of copies per day and triplet of parameters to generate
n_days = number of days over which to generate contacts (including weekends during which no contacts occur)

"""

import pandas as pd

import func_both_alg_and_opt
import itertools 
import os


dir_root = ''

type_copy = 'class-mixing' #or 'looped', 'friendship-based'
fPt_values = [0.8,0.4,0.75]
days_selected = [2,3,4,5]
n_days = 30
w_step = 20
s_in_day = 24 * 60 * 60
s_in_h = 60 * 60
s_in_15min = 15 * 60

dir_out = dir_root + 'sequences-%ddays/'%n_days
dir_data = dir_root + 'data_in/'
if type_copy == 'class-mixing':
    dir_copies = dir_root + 'copies-class-mixing-%dd/'%len(days_selected)
    
elif type_copy == 'friendship-based':
    dir_copies = dir_root + 'copies-friendship-based-%dd/copies-f%03d-P%03d-t%03d/'%(len(days_selected),100*fPt_values[0], 100*fPt_values[1],100*fPt_values[2])

elif type_copy == 'looped':
    dir_data = dir_root + 'data_in/'
    
else:
    print('unknown copy type')
    raise KeyboardInterrupt()


dict_classes = {'2BIO1': 11,'2BIO2':12, '2BIO3': 13,'MP':21, 'MP*1': 22, 'MP*2':23, 'PC':31, 'PC*':32, 'PSI*':33, 'Teachers':6}
str_header = ['t', 'n1', 'n2', 'c1', 'c2', 'day', 'w', 't_day', 't_all_days']

#empirical contacts
df_emp = pd.read_table(dir_data +'CP-tidy-2345.csv', \
                delimiter='\t', names = str_header)
df_emp = df_emp.astype({"t":int, "n1":int, "n2":int, "c1":str, "c2":str, \
    "day":int, "w":int, "t_day":int, "t_all_days":int})

#aggregate in 15 mins step the empirical networks, and collect the daily total interaction weight
weight_tot_day = dict()
for day in set(df_emp.day):
    df = df_emp.copy().query('day == @day')
    df['step'] = df['t_day'].apply(func_both_alg_and_opt.intervals_15min)
    df = df.groupby(['n1','n2','c1','c2','step']).size()
    df = df.reset_index()
    df.columns = ['n1','n2','c1','c2','step','num']
    df['w'] = df['num'] * w_step
    df['c1'] = df['c1'].apply(lambda x: dict_classes[x])
    df['c2'] = df['c2'].apply(lambda x: dict_classes[x])
    df['w'] = df['w'].apply(lambda x: x/(s_in_15min))
        
    df_day = df.copy()
    print(sum(df_day.w))
    weight_tot_day[day] = sum(df_day.w)
    
weight_tot_day[1] = weight_tot_day[2]


#sequences of base days keyed by number of base days. 
days_to_cop = dict()
days_to_cop[4] = [2,2,3,4,5]
days_to_cop[3] = [2,2,3,4,3]
days_to_cop[2] = [2,2,3,2,3]
days_to_cop[1] = [2,2,2,2,2]

#%% ARRANGE THE COPIES
copy_number = dict() #will count how many base days is copied over the sequence.
for day in days_selected:
    copy_number[day] = 1


day = 1 # day counter
copy_i = 1 #copy number used, initialised to 1.

days_copied = itertools.cycle(days_to_cop[len(days_selected)]) #days in the base data used.

if not os.path.exists(dir_out):          
    os.makedirs(dir_out)
    
with open(dir_out+'contacts-%s-%dd-%d-days.txt'%(type_copy, len(days_selected),n_days), 'w') as f: 
    while day <= n_days:
        day_week = (day-1) % 7  +1
            
        if day_week == 6 or day_week == 7:  #skip weekeends
            day+=1
            print('weekend')
            continue
        
        day_i = next(days_copied)
        if type_copy == 'looped': #select the empirical contact on the base days and aggregate them over 15 mins steps
            df = df_emp.copy().query('day == @day_i')
            df['step'] = df['t_day'].apply(func_both_alg_and_opt.intervals_15min)
            df = df.groupby(['n1','n2','c1','c2','step']).size()
            df = df.reset_index()
            df.columns = ['n1','n2','c1','c2','step','num']
            df['w'] = df['num'] * w_step
            df['c1'] = df['c1'].apply(lambda x: dict_classes[x])
            df['c2'] = df['c2'].apply(lambda x: dict_classes[x])
            df['w'] = df['w'].apply(lambda x: x/(s_in_15min))
        else: #load the synthetic contacts of the base day, with the right copy number.
            dir_day = 'DAY%d/'%day_i
            
            df = pd.read_table(dir_copies+dir_day+'dynamic_copy_%02d.csv'%copy_number[day_i], \
                            delimiter='\t', names = ['t', 'n1', 'n2', 'c1', 'c2'])
            print(dir_copies+dir_day+'dynamic_copy_%02d.csv'%copy_number[day_i])
            #aggregate them in 15 mins steps
            df['step'] = df['t'].apply(func_both_alg_and_opt.intervals_15min)
            
            df = df.groupby(['n1','n2','c1','c2','step']).size()
            df = df.reset_index()
            df.columns = ['n1','n2','c1','c2','step','num']
            df['w'] = df['num'] * w_step
            df['c1'] = df['c1'].apply(lambda x: dict_classes[x])
            df['c2'] = df['c2'].apply(lambda x: dict_classes[x])
            df['w'] = df['w'].apply(lambda x: x/(s_in_15min))
        
        #rescale the total weight
        fact = weight_tot_day[day_week]/sum(df.w)
        df['w'] = df['w'].apply(lambda x: x*fact)
        df['day'] = day #day
        df = df.sort_values(by=['step'])
        df = df.reindex(columns=['day','step', 'n1', 'n2', 'c1', 'c2', 'w'])
        copy_number[day_i]=copy_number[day_i]+1
        day+=1
        df.to_csv(f, sep="\t", header = None, index = None)

#%% save nodes present on the days
# nodes_inc = set()
# for day in days_to_cop[len(days_selected)]:
#     nodes_inc = nodes_inc.union(set(df_emp.query('day == @day').copy().n1))
#     nodes_inc = nodes_inc.union(set(df_emp.query('day == @day').copy().n2))
    
# with open(dir_out + 'metadata_%dd.txt'%len(days_selected),'w') as f:
#     for node in nodes_inc:
#         f.write('%d\n'%node)
                
    
    
    

