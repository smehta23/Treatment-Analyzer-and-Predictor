import pickle
import pandas as pd
import numpy as np
from math import*
import collections as col
import matplotlib.pyplot as plt


def euclidean_distance (x,y):
    return sqrt(abs(sum(pow(a-b, 2) for a, b in zip(x,y))))

#converting .p files to python dicts
actions_in = open('actions_discretized_vaso4_fluid4_1hr_bpleq65.p', 'rb')
actions = pickle.load(actions_in)

states_in = open('states_1hr.p','rb')
states = pickle.load(states_in)

#states dict to states dataframe
statesdf = pd.DataFrame.from_dict(states, orient='index')


#list of Patient IDs based on keys of the original states dict model
statesListOfPatientIDs = list(states.keys())
statesdf['PatientIDs'] = statesListOfPatientIDs

statesList = statesdf.values.tolist()

statesData = list(states.values()) #states dict into states list

#weeds out patients with no state-action correlation
noCorPatients = []
x = 0
while x < len(statesData):
    PID = statesList[x][1]
    if(PID not in actions):
        noCorPatients.append(PID)
        del statesList[x] 
        del statesData[x]
    else: 
        x = x + 1
        
        
#length of states data in rows is the number patients with state-action correlation in the data
numOfPatients = len(statesData)
        

#inputting all the time steps of states of every patient into this array
totalTimeSteps = []
totalTimeStepsWithPatientID = []
i = 0
j = 0
for i in range(numOfPatients): #first insert a test number less than 5, then input numOfPatients as program will take a LONG time
    numOfTimeSteps = len(statesData[i]) #number of timesteps
    for j in range (numOfTimeSteps):
        totalTimeSteps.append(statesData[i].iloc[j,:])
        totalTimeStepsWithPatientID.append(statesData[i].iloc[j,:].append(pd.Series(statesList[i][1])))
    j = 0
    
    
    

#2D list to calculate the distance between time steps (numpy array can't do due to its 32 x 32 limitation)
distanceMatrix = [[-1 for g in range(len(totalTimeSteps))] for h in range(len(totalTimeSteps))]
FinalKNNmatrix = []
FinalOutputMatrix = []
neighbors = []
proportions = []
TimeStepLabel = []
TimeStepSameAction = []
EDcalculations = len(totalTimeSteps)
distance = 0


#k is which row it's in, l is which column
k = 0
for k in range (EDcalculations): 
    #l initalizes a k as we are taking advantage of matrix symmetry when calculating Euclidean Distance
    l = k
    for l in range (EDcalculations): 
        
        ED = euclidean_distance(totalTimeSteps[k], totalTimeSteps[l])
        PatientID1 = totalTimeStepsWithPatientID[k][0] #comparer patient (constant)
        PatientID2 = totalTimeStepsWithPatientID[l][0] #compared to the comparer patient
        time_comparedTo = totalTimeSteps[k].get(key='Times') #comparer timestep of the constant patient
        time_compared = totalTimeSteps[l].get(key='Times') #timestep being compared
        distanceMatrix[k][l] = [ED, PatientID1, time_comparedTo, PatientID2, time_compared]
        distanceMatrix[l][k] = [ED, PatientID2, time_compared, PatientID1, time_comparedTo]  
        
        
        
    KNNmatrix = []
    ActionMatrix = []
    l = 0   
    distance = 100 #arbitrarily set as maximum ED
    for l in range (EDcalculations):
        if distanceMatrix[k][l][0] < distance: 
            patientData = actions.get(distanceMatrix[k][l][3])
            timeData = int(distanceMatrix[k][l][4])
            actionSeries = patientData.iloc[timeData-1]
            ActionMatrix = [actionSeries.get(key='FLUID_ID'), actionSeries.get(key='VASO_ID')]
            if(l == k): ActionListTimestep = ActionMatrix 
            #logic if a patient in the actions data set does not have the same amount of timesteps as in the states data set
            if ActionMatrix is None:
                KNNmatrix.append("Patient " + str(distanceMatrix[k][l][3]) + " has no action correlation at time " + str(timeData))
            else:    
                KNNmatrix.append(ActionMatrix)
                
    #generating statistics from the K nearest neighbors of specific timestep (k)
    a = 0
    #lists are unhashable, so they are mapped into tuples for future calculations
    testableKNN1 = map(tuple, KNNmatrix) 
    testableKNN2 = map(tuple, KNNmatrix)
    treatmentMeasure_stat = []
    typeAction_stat = []
    output_statistics = []
    occurenceOfTimestepAction = 0
    zeroTreatment = 0
    vasoTreatment = 0
    fluidTreatment = 0 
    bothTreatments = 0
    for a in range (len(KNNmatrix)):
        #to measure how many times the treatment of the particular timestep (fluid AND vasopressor) is seen throughout all the timesteps
        if(ActionListTimestep == KNNmatrix[a]): occurenceOfTimestepAction = occurenceOfTimestepAction + 1
        #to measure how many times no treatment is given, only fluid/only vasopressor treatment is given, and both treatments are given across all timesteps
        if(KNNmatrix[a][0] == 0 and KNNmatrix[a][1] == 0):
            zeroTreatment= zeroTreatment + 1
        elif (KNNmatrix[a][0] == 0 and KNNmatrix[a][1] > 0):
            vasoTreatment = vasoTreatment + 1
        elif (KNNmatrix[a][0] > 0 and KNNmatrix[a][1] == 0):
            fluidTreatment = fluidTreatment + 1
        elif (KNNmatrix[a][0] > 0 and KNNmatrix[a][1] > 0):
            bothTreatments = bothTreatments + 1
    
    #occurenceOfTreatmentMode counts how often the mode treatment (vaso + fluid) occurs through the k-nearest neighbors of a particular timestep
    occurenceOfTreatmentMode = col.Counter(testableKNN1).most_common(1)[0][1]
    #treatmentMeasure_stat is a list that stores the most seen treatment (mode treatment), how many times it occurred, and the proportion/percentage of the time it occured amongst the k-nearest neighbors
    treatmentMeasure_stat = [col.Counter(testableKNN2).most_common(1)[0][0], occurenceOfTreatmentMode, occurenceOfTreatmentMode/len(KNNmatrix)]
    #stores how often each type of treatment was seen (no treatment, vaso treatment, etc.)
    treatment = [zeroTreatment, vasoTreatment, fluidTreatment, bothTreatments]
    #typeActionMode stores the most seen type of treatment and occurenceOfTypeAction stores how often that type of treatment was seen amongst the timestep's k nearest neighbors
    if (treatment[0] == max(treatment)): 
        typeActionMode = "No Treatment"
        occurenceOfTypeAction = treatment[0]
    elif (treatment[1] == max(treatment)): 
        typeActionMode = "Vasopressor Treatment ONLY"
        occurenceOfTypeAction = treatment[1]
    elif (treatment[2] == max(treatment)): 
        typeActionMode = "Fluid Treatment ONLY"
        occurenceOfTypeAction = treatment[2]
    elif (treatment[3] == max(treatment)): 
        typeActionMode = "BOTH Treatments"
        occurenceOfTypeAction = treatment[3]
        
    typeAction_stat = [treatment, typeActionMode, occurenceOfTypeAction/len(KNNmatrix)]
    output_statistics = [len(KNNmatrix), treatmentMeasure_stat, typeAction_stat]
    
    
    FinalKNNmatrix.append(KNNmatrix)
    FinalOutputMatrix.append(output_statistics)
    neighbors.append(len(KNNmatrix))
    proportions.append(occurenceOfTreatmentMode/len(KNNmatrix))
    TimeStepSameAction.append(occurenceOfTimestepAction/len(KNNmatrix))
            
            
#TimeStepLabel is used for the x-axis of the graphs
TimeStepLabel = list(range(1, EDcalculations + 1))

#graphs using matplotlib
fig1 = plt.figure(1)
plt.bar(TimeStepLabel, neighbors) 
fig1.suptitle('Neighbors')
plt.ylabel('Number of Nearest Neighbors as per euclidean distance ' + str(distance) + ' (k)')
plt.xlabel('Timestep out of total number of timesteps (' + str(EDcalculations) + ')')

fig2 = plt.figure(2)
plt.bar(TimeStepLabel, proportions)
fig2.suptitle('Proportion')
plt.ylabel('Proportion of Neighbors to Timestep with the most seen (mode) treatment')
plt.xlabel('Timestep out of total number of timesteps (' + str(EDcalculations) + ')')

fig3 = plt.figure(3)
fig3.suptitle('Proportion')
plt.bar(TimeStepLabel, TimeStepSameAction)
plt.ylabel('Proportion of Neighbors to Timestep with the same treatment as Timestep')
plt.xlabel('Timestep out of total number of timesteps (' + str(EDcalculations) + ')')