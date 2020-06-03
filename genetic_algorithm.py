from data_utils import xml_to_csv, generate_pbtxt, make_config
from tensorboard.backend.event_processing import event_accumulator

import matplotlib.pyplot as plt
import numpy as np

import os

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
import array
import math


def model_train(num_steps=200000, initial_learning_rate=0.004, decay_factor=0.95, momentum_optimizer_value=0.9, decay=0.9, epsilon=1.0):
    
    cur_dir = os.getcwd()
    PATH_TO_DATA = 'dataset'
    os.chdir(PATH_TO_DATA)
    
    for directory in ['valid_annot','train_annot']:
        image_path = os.path.join(os.getcwd(), directory )
        xml_df = xml_to_csv.xml_to_csv(image_path)
        xml_df.to_csv('{}_labels.csv'.format(directory), index=None)
        print('{} was successfully converted xml to csv.'.format(directory))
        
        if directory == 'train_annot':
            with open('model_label_map.pbtxt', "w") as file:
                file.write(generate_pbtxt.generate_pbtxt(xml_df))
                print('pbtxt-file was created!')
        
        
    os.system("python ..\\data_utils\\generate_tfrecord.py --csv_input=..\\dataset\\train_annot_labels.csv  --output_path=..\\dataset\\train.record --image_dir=..\\dataset\\train_images")
    os.system("python ..\\data_utils\\generate_tfrecord.py --csv_input=..\\dataset\\valid_annot_labels.csv  --output_path=..\\dataset\\valid.record --image_dir=..\\dataset\\valid_images")
    os.chdir('../models')
    if 'ssdmn_fine_tuned' in os.listdir():
        os.system('rmdir ssdmn_fine_tuned /s /q')
    os.system('mkdir ssdmn_fine_tuned')
    os.chdir('ssdmn_fine_tuned')
    os.system('tar xzvf ../ssd_mobilenet_v1_coco_11_06_2017.tar.gz')
    os.chdir('..')
    os.system('copy ssd_mobilenet_v1_coco.config ssdmn_fine_tuned')
    config_path = 'ssdmn_fine_tuned/ssd_mobilenet_v1_coco.config'
    config_data = make_config.make_config(config_path, len(xml_df['class'].unique()),num_steps=num_steps, initial_learning_rate=initial_learning_rate, decay_factor=decay_factor, momentum_optimizer_value=momentum_optimizer_value, decay=decay, epsilon=epsilon)
    with open(config_path, "w") as file:
        file.write(config_data)
    
    os.chdir('..')
    if 'genetic_algorithm_training' in os.listdir():
        os.system('rmdir genetic_algorithm_training /s /q')
        os.system('mkdir genetic_algorithm_training')
    os.system('python train.py --logtostderr --train_dir=genetic_algorithm_training --pipeline_config_path=models/ssdmn_fine_tuned/ssd_mobilenet_v1_coco.config')
    
    os.chdir('genetic_algorithm_training')
    ga_dir_lst = os.listdir()
    events_name = list(filter(lambda x: 'events' in x, ga_dir_lst))[0]
    
    ea = event_accumulator.EventAccumulator(events_name,
    size_guidance={ # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    })

    ea.Reload()

    data = ea.Scalars('Losses/TotalLoss')
    
    steps = [el[1] for el in data][1::]
    values = np.array([el[2] for el in data][1::])
    
    os.chdir(cur_dir)
    
    return steps, values
    

def fitness_function(ind):
    if (ind[0]>1.0) or (ind[1]>1.0) or (ind[2]>1.0) or (ind[3]>1.0) or (ind[4]>1.0):
        return 999999,
    else:
        _, values = model_train(num_steps=400, initial_learning_rate=ind[0], decay_factor=ind[1], momentum_optimizer_value=ind[2], decay=ind[3], epsilon=ind[4])
        print(values)
        if values==[]:
            return 999999,
        else:
            return values[-1],


def choice_based(ind, indpb):
    new_ind = toolbox.clone(ind)
    for i in range(0, len(ind)):
        if np.random.uniform() < indpb:
            if i==0:
                new_ind[i] = np.random.uniform(LR_START, LR_END)
            if i==1:
                new_ind[i] = np.random.uniform(DF_START, DF_END)
            if i==2:
                new_ind[i] = np.random.uniform(MOV_START, MOV_END)
            if i==3:
                new_ind[i] = np.random.uniform(D_START, D_END)
            if i==4:
                new_ind[i] = np.random.uniform(E_START, E_END)
    return new_ind,

IND_SIZE = 5
LR_START = 0.00001
LR_END = 0.15
DF_START = 0.7
DF_END = 0.99
MOV_START = 0.6
MOV_END = 0.99
D_START = 0.7
D_END = 0.99
E_START = 0.5
E_END = 1.0

N_POP = 8
NGEN = 5
CXPB = 0.8
MUTPB = 0.05

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("LR", np.random.uniform, LR_START, LR_END)
toolbox.register("DF", np.random.uniform, DF_START, DF_END)
toolbox.register("MOV", np.random.uniform, MOV_START, MOV_END)
toolbox.register("D", np.random.uniform, D_START, D_END)
toolbox.register("E", np.random.uniform, E_START, E_END)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.LR, toolbox.DF, toolbox.MOV, toolbox.D, toolbox.E), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.2)
toolbox.register("mutate", choice_based, indpb=0.9)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

pop = toolbox.population(n=N_POP)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

'''
all_graphs_values = []
for i in range(0,5):
    steps, values = model_train(num_steps=2000)
    all_graphs_values.append(values)
    
    
all_graphs_values = [el[0:4] for el in all_graphs_values]
mean_values = np.mean(all_graphs_values, axis=0)
std_values = np.std(all_graphs_values, axis=0)

y_up = mean_values + std_values
y_down = mean_values - std_values

plt.plot(steps, mean_values, 'r', steps, y_up, 'c', steps, y_down, 'c')

'''