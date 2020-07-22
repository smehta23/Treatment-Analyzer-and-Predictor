#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:40:49 2020

@author: owner
"""

import time
import pickle
import pandas as pd
import numpy as np
from math import*
import collections as col
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import csv 
import operator
import random



def euclidean_distance (x,y):
    ED = 0
    for i in range (1, len(x)-1):
        ED = ED + (x[i] - y[i])**2
    ED = sqrt(ED)
    return ED


#-----------------------------PREPROCESSING THE STATES (CONDITIONS) AND ACTIONS (TREATMENTS) DATA IN PICKLE FILES-----------------------------------#

#converting .p files to python dicts
actions_in = open('actions_discretized_vaso4_fluid4_1hr_bpleq65.p', 'rb')
actions = pickle.load(actions_in)
states_in = open('states_1hr.p','rb')
states = pickle.load(states_in)


#actions dict to actions dataframe (actionsdf is a dataframe of dataframes)
actionsdf = pd.DataFrame.from_dict(actions, orient='index')
#list of keys of action dictionary -- keys are the patient IDs
actionKeys = list(actions.keys());
#adding a column 'PatientIDs' containing list of patient IDs to actionsdf
actionsdf['PatientsIDs'] = actionKeys
#transforming the actionsdf (dataframe) to actionsList (now a list of lists -- [[200028, Dataframe]...]
actionsList = actionsdf.values.tolist()
#actions dict into actions list (list of dataframes)
actionsData = list(actions.values())


#states dict to states dataframe (statesdf)
statesdf = pd.DataFrame.from_dict(states, orient='index')
#list of Patient IDs based on keys of the original states dict model
statesListOfPatientIDs = list(states.keys())
#adding a column 'PatientIDs' containing list of patient IDs to statesdf
statesdf['PatientIDs'] = statesListOfPatientIDs
#transforming the statesdf (dataframe) to statesList
statesList = statesdf.values.tolist()
#states dict into states list (list of dataframes)
statesData = list(states.values()) 


#---------------------------------WEEDING OUT PATIENTS IN THE STATESDATA (LIST OF DATAFRAMES) W/O STATE-ACTION CORRELATION--------------------------#
 

noCorPatients = []

x = 0
while x < len(statesData):
    PID = statesList[x][1]
    if(PID!=actionKeys[x]):
        noCorPatients.append(PID)
        del statesList[x] 
        del statesData[x]
        del statesListOfPatientIDs[x]  
    else: 
       x = x + 1
 

#------------------------------------SELECTING RANDOM SAMPLE OF PATIENTS THE PROGRAM WILL BE RUNNING WITH------------------------------------------#

                 
percentForTestingProgram = 0.03; #should be 1 when running for real       
percentTraining = 0.7


TOTAL_PATIENTS = int(percentForTestingProgram * len(statesData))
patients_set_IDs = random.sample(actionKeys, TOTAL_PATIENTS) #actionKeys = statesListOfPatientIDs
patients_set = []


for patient_ID in patients_set_IDs:
    patients_set.append(states.get(patient_ID)) #appending all selected patient dataframes to the patients_set list




   
#------------------------------------------------------ADDING PREVIOUS ACTION TO DATAFRAME--------------------------------------------------------#   

for x in range (0, TOTAL_PATIENTS): 
    vasoSeries = []
    fluidSeries = []
    previousVaso = []
    previousFluid = []
    
    print("Adding Previous Action: Patient " + str(x + 1))

    correspondingActionDataframe = actions.get(patients_set_IDs[x]) #patients_set_IDs <--> patients_set
    
    vasoSeries = correspondingActionDataframe['RAW_VASO'][0:len(patients_set[x]) - 1]
    fluidSeries = correspondingActionDataframe['RAW_FLUID'][0:len(patients_set[x]) - 1]
    
    previousVaso = [0] #0 because there is no action taken before the zeroth timestep in states
    previousFluid = [0] #0 because there is no action taken before the zeroth timestep in states
    previousVaso[1:] = vasoSeries
    previousFluid[1:] = fluidSeries
    
    
    patients_set[x]['Previous Vasopressin'] = previousVaso
    patients_set[x]['Previous Fluid'] = previousFluid 
        
    
    patients_set[x]['Previous Vasopressin'].fillna(0)
    patients_set[x]['Previous Fluid'].fillna(0)

#------------------------------------------------------FEATURE SCALING W/SCI-KIT LEARN----------------------------------------------------------#

      
for i, patient in enumerate(patients_set, 1): 
    a = 1 #starting at 1 so that the 0th column, 'Times' is not feature scaled
    while a < patient.shape[1]: 
        print("Feature Scaling: Patient " + str(i))
        z = np.array([patient.iloc[:, a]])
        y = z.reshape(len(z[0]), len(z))
        standardScaling = preprocessing.StandardScaler()
        scaledColumn = standardScaling.fit_transform(y)
        patient.iloc[:, a] = scaledColumn
        a = a + 1
    
        
    
#---------------------------------------PARTIONING TOTAL PATIENTS INTO TESTING & TRAINING SETS-------------------------------------------------#


#zipping patients_set and patients_set_IDs to a dictionary
patientStatesDictionary = dict(zip(patients_set_IDs, patients_set))

TOTAL_PATIENTS_TRAINING = int(percentTraining * TOTAL_PATIENTS)
TOTAL_PATIENTS_TESTING = TOTAL_PATIENTS - TOTAL_PATIENTS_TRAINING

patients_training_set_IDs = patients_set_IDs[0:TOTAL_PATIENTS_TRAINING]
patients_testing_set_IDs = patients_set_IDs[TOTAL_PATIENTS_TRAINING: len(patients_set_IDs)]

patients_training_set = []
patients_testing_set = []



for a in range (0, TOTAL_PATIENTS_TRAINING):
    patients_training_set.append(patientStatesDictionary.get(patients_training_set_IDs[a]))
    patients_training_set[a]['Patient ID'] = patients_training_set_IDs[a]

for b in range (0, TOTAL_PATIENTS_TESTING):
    patients_testing_set.append(patientStatesDictionary.get(patients_testing_set_IDs[b]))
    patients_testing_set[b]['Patient ID'] = patients_testing_set_IDs[b]

#----------------------------------------------------------AGGREGATING TRAINING & TESTING TIMESTEPS------------------------------------------#

""" MAIN K-NEAREST NEIGHBOR ALGORITHM BEGINS HERE """

start_time = time.time()

totalTSTraining = 0
totalTSTesting = 0

for i in range(0, TOTAL_PATIENTS_TRAINING): 
    patients_training_set[i]['Previous Vasopressin'].fillna(0, inplace=True)
    patients_training_set[i]['Previous Fluid'].fillna(0, inplace=True)    
    patients_training_set[i].to_csv ('training_timesteps.csv', header=False, index = False, chunksize = 100000, mode = 'a')
    print("Aggregating: Training Patient " + str(i))
    totalTSTraining = totalTSTraining + len(patients_training_set[i])


for i in range(0, TOTAL_PATIENTS_TESTING): 
    patients_testing_set[i]['Previous Vasopressin'].fillna(0, inplace=True)
    patients_testing_set[i]['Previous Fluid'].fillna(0, inplace=True)    
    patients_testing_set[i].to_csv ('testing_timesteps.csv', header=False, index = False, chunksize = 100000, mode = 'a')
    print("Aggregating: Test Patient " + str(i))
    totalTSTesting = totalTSTesting + len(patients_testing_set[i])
    
#---------------------------------------------------------------DECLARING MAX ED AND K-----------------------------------------------------# 


K = int(sqrt(totalTSTraining))
maxEuclideanDistance = 53 #two columns have been removed 


#---------------------------CALCULATING EUCLIDEAN DISTANCES BETWEEN INDIVIDUAL TIMESTEPS AND ALL OTHERS and KNN ITSELF------------------------------------#



with open("testing_timesteps.csv", "r") as f: 
        
        constantReader = csv.reader(f)
        
        
        #the constant loop (timestep that is being compared to all others ==> should be from the testing set)

        for rowNum, row in enumerate(constantReader, 1):
            euclidean_distances_list = []
            nearest_neighbors_raw_treatments = []
            nearest_neighbors_ID_treatments = []
            K_neighbors_raw_treatments = []
            K_neighbors_ID_treatments = []

            row = [float(j) for j in row] 
            PatientID1 = row[len(row) - 1]
            timestep1 = row[0]
            
            testPatientActualDataframe = actions.get(PatientID1)
            testPatientActualIDFluid = testPatientActualDataframe['FLUID_ID'][timestep1 - 1]
            testPatientActualIDVaso = testPatientActualDataframe['VASO_ID'][timestep1 - 1]
            testPatientActualRawFluid = testPatientActualDataframe['RAW_FLUID'][timestep1 - 1]
            testPatientActualRawVaso = testPatientActualDataframe['RAW_VASO'][timestep1 - 1]
            
            testPatientActualIDTreatment = [testPatientActualIDFluid, testPatientActualIDVaso]
            testPatientActualRawTreatment = [testPatientActualRawFluid, testPatientActualRawVaso]
            
            
            
            with open("training_timesteps.csv", "r") as g, open(
                "euclidean_calculations.csv", "w") as h, open(
                        "euclidean_calculations.csv", "r") as readCalc, open(
                                "euclidean_calculations.csv","w") as writeCalc, open(
                                         "euclidean_calculations.csv", "r") as neighbors_nearest, open(
                                                 "euclidean_calculations.csv", "r") as neighbors_k: 
            
                varyingReader = csv.reader(g)
                calcWriter = csv.writer(writeCalc)
                calcReader = csv.reader(readCalc)
                euclideanCalculationsWriter = csv.writer(h)
                gettingNearestNeighbors = csv.reader(neighbors_nearest)
                gettingKNeighbors = csv.reader(neighbors_k)
            #the varying loop (timestep being compared to constant timestep)
            
                for lineNum, line in enumerate(varyingReader, 1):
                    
                    line = [float(k) for k in line] 
                    PatientID2 = line[len(row) - 1]
                    timestep2 = line[0]
                    print("Line " + str(lineNum) + ": " + str(euclidean_distance(line, row)))
                   
                    euclidean_distances_list.append(
                            [rowNum, lineNum, euclidean_distance(line, row), PatientID1, timestep1, PatientID2, timestep2]
                            #   (0)         (1)                    (2)                 (3)        (4)         (5)         (6)
                    )
                    
                    
    
                
    
                #sorting the column of calculated Euclidean distance
                sort = sorted(euclidean_distances_list, key=operator.itemgetter(2))
                for eachListNum, eachList in enumerate(sort, 1):
                    print("Inserting Sorted Row " + str(eachListNum))
                    calcWriter.writerow(eachList)
                    
                    
                    
    
                
                #MAX_DISTANCE VARIANT: instead of a static K, K will vary depending on how many state points fall within the max euclidean distance
                for calculatedRowNum, calculatedRow in enumerate(gettingNearestNeighbors, 1):
                    print("Within Max Euclidean Distance - Row " + str(calculatedRowNum))
                    if (float(calculatedRow[2]) < maxEuclideanDistance):
                        patientID_forActionQuery = int(float(calculatedRow[5]))
                        timestepNum_forActionQuery = int(float(calculatedRow[6]))
                        
                        patientActionDataframe = actions.get(patientID_forActionQuery)
                        
                        rawFluidTreatment = patientActionDataframe['RAW_FLUID'][timestepNum_forActionQuery - 1] #-1 because index is one less than the timestep value 
                        rawVasoTreatment = patientActionDataframe['RAW_VASO'][timestepNum_forActionQuery - 1]#-1 because index is one less than the timestep value 
                        IDFluidTreatment = patientActionDataframe['FLUID_ID'][timestepNum_forActionQuery - 1]
                        IDVasoTreatment = patientActionDataframe['VASO_ID'][timestepNum_forActionQuery - 1]
                        
                        rawTreatment = [rawFluidTreatment, rawVasoTreatment]
                        IDTreatment = [IDFluidTreatment, IDVasoTreatment]
                        
                        nearest_neighbors_raw_treatments.append(rawTreatment)
                        nearest_neighbors_ID_treatments.append(IDTreatment)
                        
                        
                        
                    else:
                        break
                    
                
                for neighborNum, neighbor in enumerate(gettingKNeighbors, 1):
                    print("Current K (Line): " + str(neighborNum))
                    if (neighborNum <= K):
                        patientID_forActionQuery = int(float(neighbor[5]))
                        timestepNum_forActionQuery = int(float(neighbor[6]))
                        
                        patientActionDataframe = actions.get(patientID_forActionQuery)
                        
                        rawFluidTreatment = patientActionDataframe['RAW_FLUID'][timestepNum_forActionQuery - 1] #-1 because index is one less than the timestep value 
                        rawVasoTreatment = patientActionDataframe['RAW_VASO'][timestepNum_forActionQuery - 1]#-1 because index is one less than the timestep value 
                        IDFluidTreatment = patientActionDataframe['FLUID_ID'][timestepNum_forActionQuery - 1]
                        IDVasoTreatment = patientActionDataframe['VASO_ID'][timestepNum_forActionQuery - 1]
                        
                        rawTreatment = [rawFluidTreatment, rawVasoTreatment]
                        IDTreatment = [IDFluidTreatment, IDVasoTreatment]
                        
                        K_neighbors_raw_treatments.append(rawTreatment)
                        K_neighbors_ID_treatments.append(IDTreatment)

                    else:
                    	break
                    
                



                #acquiring results
                #finding mode treatment of nearest neighbors (Raw and ID); max euclidean distance method
                #if-else used here as max euclidean distance method could potentially return 0 nearest neighbors if max value is too small
                if (len(nearest_neighbors_raw_treatments) > 0):
                    TUPLE_nearest_neighbors_raw_treatments =  map(tuple, nearest_neighbors_raw_treatments)
                    modeTreatment = list(col.Counter(TUPLE_nearest_neighbors_raw_treatments).most_common(1)[0])
                    predictedRawTreatment_nearest_raw = modeTreatment[0]
                    predictedRawTreatment_nearest_raw_occurence = modeTreatment[1]
                    
                else:
                    predictedRawTreatment_nearest_raw = "null (no nearest neighbors found with given maximum ED)"
                
                if (len(nearest_neighbors_ID_treatments) > 0):
                    TUPLE_nearest_neighbors_ID_treatments =  map(tuple, nearest_neighbors_ID_treatments)
                    modeTreatment = list(col.Counter(TUPLE_nearest_neighbors_ID_treatments).most_common(1)[0])
                    predictedIDTreatment_nearest_ID = modeTreatment [0]
                    predictedIDTreatment_nearest_ID_occurence = modeTreatment [1]

                
                else:
                    predictedIDTreatment_nearest_ID = "null (no nearest neighbors found with given maximum ED)"
                
                
                TUPLE_K_neighbors_raw_treatments =  map(tuple, K_neighbors_raw_treatments)
                modeTreatment = list(col.Counter(TUPLE_K_neighbors_raw_treatments).most_common(1)[0])
                predictedRawTreatment_K_raw = modeTreatment
                
                TUPLE_nearest_neighbors_ID_treatments =  map(tuple, nearest_neighbors_ID_treatments)
                modeTreatment = list(col.Counter(TUPLE_nearest_neighbors_ID_treatments).most_common(1)[0])
                predictedRawTreatment_K_ID = modeTreatment
            

                #results in the form [A, B] ==> [Fluid, Vaso]
                with open('results.csv', 'a') as resultsFile:
                    results = csv.writer(resultsFile)
                    if (rowNum == 1v                        results.writerow(["Patient ID", "Timestep #", "Nearest Neighbors Amount (Raw)", "K Neighbors Amount (Raw)", 
                                            "Nearest Neighbors Amount (ID)", "K Neighbors Amount (ID)", 
                                            "MaxED: Raw Treatment", "MaxED: Occurence in Neighbors",
                                            "MaxED: ID Treatment", "MaxED: Occurence in Neighbors", 
                                            "K: Raw Treatment", "K: Occurence in Neighbors", 
                                            "K: ID Treatment", "K: Occurence in Neighbors", 
                                            "Actual Raw Treatment", "Occurence (MaxED Raw)", "Occurence (K Raw)", 
                                             "Actual ID Treatment", "Occurence (MaxED Raw)", "Occurence (K Raw)"])
                    resultData = [PatientID1, timestep1, len(nearest_neighbors_raw_treatments), len(K_neighbors_raw_treatments), 
                                    len(nearest_neighbors_ID_treatments), len(K_neighbors_ID_treatments),
                                    predictedRawTreatment_nearest_raw, predictedRawTreatment_nearest_raw_occurence,
                                    predictedIDTreatment_nearest_ID, predictedIDTreatment_nearest_ID_occurence, 
                                    predictedRawTreatment_K_raw[0], predictedRawTreatment_K_raw[1], 
                                    predictedRawTreatment_K_ID[0], predictedRawTreatment_K_ID[1], 
                                    testPatientActualRawTreatment, nearest_neighbors_raw_treatments.count(testPatientActualRawTreatment), 
                                    K_neighbors_raw_treatments.count(testPatientActualRawTreatment), testPatientActualIDTreatment, 
                                    nearest_neighbors_ID_treatments.count(testPatientActualIDTreatment), 
                                    K_neighbors_ID_treatments.count(testPatientActualIDTreatment)]
                    results.writerow(resultData)


                
                
               
            if (rowNum >= 1000):
                break
                
#            break
