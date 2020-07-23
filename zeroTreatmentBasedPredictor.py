#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 09 20:40:49 2020

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



def euclidean_distance (x,y, parameters):
    ED = 0
    for i in range (1, len(x)-1):
        if ("_ind" not in parameters[i]):  #taking out _ind parameters from Euclidean distance calculations
            ED = ED + (x[i] - y[i])**2
    ED = sqrt(ED)
    return ED

def findOccurenceOfActualRawTreatment(actual_raw_treatment, raw_treatments_of_neighbors):
    actual_raw_treatment_integers = [round(specific_treatment) for specific_treatment in actual_raw_treatment]
    for neighbors_treatment_list in raw_treatments_of_neighbors:
        neighbors_treatment_list = [round(specific_treatment) for specific_treatment in neighbors_treatment_list]

    occurence = 0
    for i in range (len(raw_treatments_of_neighbors)):
        if (raw_treatments_of_neighbors[i] == actual_raw_treatment_integers):
            occurence = occurence + 1





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
 

#-------------------------REMOVING PATIENT IDS THAT HAVE FEWER THAN reducedDimensions TIMESTEPS FOR PCA---------------------------------------------#

reducedDimensions = 52 #originally 20, when PCA was done

n = len(actionKeys) - 1
while n >= 0:
    if(len(states.get(actionKeys[n])) < reducedDimensions):
        actionKeys.pop(n)
        statesListOfPatientIDs.pop(n)
    n = n - 1

#------------------------------------SELECTING RANDOM SAMPLE OF PATIENTS THE PROGRAM WILL BE RUNNING WITH------------------------------------------#

                 
percentForTestingProgram = 0.03; #should be 1 when running for real       
percentTraining = 0.7

TOTAL_PATIENTS = int(percentForTestingProgram * len(statesData))
random.seed(24)
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

#------------------------------------------------------FEATURE SCALING W/SCI-KIT LEARN----------------------------------------------------------#

patient_states_parameters = patients_set[0].columns.copy()
patient_states_timesteps = []
      
for i, patient in enumerate(patients_set, 0): #feature scaling all columns including the times column 
    patient_states_timesteps.append(states.get(patients_set_IDs[i])['Times'].copy(deep = True))
    print("Feature Scaling: Patient " + str(i + 1))
    minMaxScale = preprocessing.MinMaxScaler() #effects of using different scalers?
    patient[patient_states_parameters] = minMaxScale.fit_transform(patient[patient_states_parameters])
    patient.rename(columns={'Times':'Timestep'}, inplace=True)
    patient = patient.insert(loc = 0, column = 'Times', value = patient_states_timesteps[i])


#-------------------------PRINCIPAL COMPONENT ANALYSIS TO BRING DIMENSIONS OF DATA TO reducedDimensions-----------------------------------------#
# from sklearn import decomposition

# for i, patientID in enumerate(patients_set_IDs, 0):
#     patient_states = patients_set[i]
#     pca = decomposition.PCA(n_components = reducedDimensions)
#     columnHeaders = [("Column" + str(c+1)) for c in range (0, reducedDimensions)]

#     patient_states_transformed = pca.fit_transform(patient_states)
#     sklearn_pca_patient_states = pd.DataFrame(data = patient_states_transformed, columns = columnHeaders)

#     sklearn_pca_patient_states.insert(loc = 0, column = 'Times', value = patient_states_timesteps[i])
#     patients_set[i] = sklearn_pca_patient_states #i-1 because index is being accessed


        
    
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


patient_states_parameters = patients_training_set[0].columns.copy()

#----------------------------------------------------AGGREGATING TRAINING & TESTING TIMESTEPS TO CSV----------------------------------------#

""" MAIN K-NEAREST NEIGHBOR ALGORITHM BEGINS HERE """

start_time = time.time()

totalTSTraining = 0
totalTSTesting = 0

for i in range(0, TOTAL_PATIENTS_TRAINING): 
    patients_training_set[i].to_csv ('training_timesteps.csv', header=False, index = False, chunksize = 100000, mode = 'a')
    print("Aggregating: Training Patient " + str(i))
    totalTSTraining = totalTSTraining + len(patients_training_set[i])


for i in range(0, TOTAL_PATIENTS_TESTING): 
    patients_testing_set[i].to_csv ('testing_timesteps.csv', header=False, index = False, chunksize = 100000, mode = 'a')
    print("Aggregating: Test Patient " + str(i))
    totalTSTesting = totalTSTesting + len(patients_testing_set[i])
    

#---------------------------------------------------------------DECLARING MAX ED AND K-----------------------------------------------------# 


K = int(sqrt(totalTSTraining))
maxEuclideanDistance = int(sqrt(sqrt(53))) #two columns have been removed, now 53



#---------------------------SELECTING X TEST TIMESTEPS FROM THE TESTING_TIMESTEPS CSV------------------------------------------------------#

num_of_test_timesteps = 5

with open('testing_timesteps.csv', 'r') as ttfR:
    reader = csv.reader(ttfR)
    random.seed(24)
    randomlyChosenTestTimesteps = random.sample(list(reader), num_of_test_timesteps)

with open('random_selected_testing_timesteps.csv', 'w') as ttfW:
    writer = csv.writer(ttfW)
    for timestepNum, timestep in enumerate(randomlyChosenTestTimesteps, 1):
            print("Selecting Test Timestep" + str(timestepNum))
            writer.writerow(timestep)

#--------------------CALCULATING EUCLIDEAN DISTANCES BETWEEN TEST TIMESTEPS AND THOSE IN TRAINING_TIMESTEPS and KNN ITSELF-----------------#



with open("random_selected_testing_timesteps.csv", "r") as f: 
        
        constantReader = csv.reader(f)
        
        
        #the constant loop (timestep that is being compared to all others ==> should be from the testing set)

        for rowNum, row in enumerate(constantReader, 1):
            euclidean_distances_list = []
            nearest_neighbors_raw_treatments = []
            nearest_neighbors_ID_treatments = []
            nearest_neighbors_overall_treatments = []
            K_neighbors_raw_treatments = []
            K_neighbors_ID_treatments = []
            K_neighbors_overall_treatments = []

            row = [float(j) for j in row] 
            PatientID1 = row[len(row) - 1]
            timestep1 = row[0]
            
            testPatientActualDataframe = actions.get(PatientID1)
            testPatientActualIDFluid = testPatientActualDataframe['FLUID_ID'][timestep1 - 1]
            testPatientActualIDVaso = testPatientActualDataframe['VASO_ID'][timestep1 - 1]
            testPatientActualRawFluid = testPatientActualDataframe['RAW_FLUID'][timestep1 - 1]
            testPatientActualRawVaso = testPatientActualDataframe['RAW_VASO'][timestep1 - 1]
            testPatientActualOverall = testPatientActualDataframe['OVERALL_ACTION_ID'][timestep1 - 1]
            
            testPatientActualIDTreatment = [testPatientActualIDFluid, testPatientActualIDVaso]
            testPatientActualRawTreatment = [testPatientActualRawFluid, testPatientActualRawVaso]
            testPatientActualOverallTreatment = testPatientActualOverall
            
            if (testPatientActualIDTreatment == [0, 0]):
                rowNum-=1
                continue
            elif (testPatientActualRawTreatment == [0.0, 0.0]):
                rowNum-=1
                continue
            
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
                    print("Line " + str(lineNum) + ": " + str(euclidean_distance(line, row, patient_states_parameters)))
                   
                    euclidean_distances_list.append(
                            [rowNum, lineNum, euclidean_distance(line, row, patient_states_parameters), PatientID1, timestep1, PatientID2, timestep2]
                            #   (0)         (1)                    (2)                 (3)        (4)         (5)         (6)
                    )
                    
                    
    
                
    
                #sorting the column of calculated Euclidean distance
                sort = sorted(euclidean_distances_list, key=operator.itemgetter(2))
                for eachListNum, eachList in enumerate(sort, 1):
                    print("Inserting Sorted Row " + str(eachListNum))
                    calcWriter.writerow(eachList)
                    
                    
                    
    
                
                #MAX_DISTANCE VARIANT: instead of a static K, K will vary depending on how many state points fall within the max euclidean distance
                nearestNeighborsAmount = 0
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
                        overallTreatment = patientActionDataframe['OVERALL_ACTION_ID'][timestepNum_forActionQuery - 1]

                        rawTreatment = [rawFluidTreatment, rawVasoTreatment]
                        IDTreatment = [IDFluidTreatment, IDVasoTreatment]
                        overallTreatment = overallTreatment
                        
                        nearest_neighbors_raw_treatments.append(rawTreatment)
                        nearest_neighbors_ID_treatments.append(IDTreatment)
                        nearest_neighbors_overall_treatments.append(overallTreatment)

                        nearestNeighborsAmount = nearestNeighborsAmount + 1
                        
                        
                    else:
                        break
                    
                kNeighborsAmount = 0
                for neighborNum, neighbor in enumerate(gettingKNeighbors, 1):
                    print("Current K (Line): " + str(neighborNum))
                    if (kNeighborsAmount <= K):
                        patientID_forActionQuery = int(float(neighbor[5]))
                        timestepNum_forActionQuery = int(float(neighbor[6]))
                        
                        patientActionDataframe = actions.get(patientID_forActionQuery)
                        
                        rawFluidTreatment = patientActionDataframe['RAW_FLUID'][timestepNum_forActionQuery - 1] #-1 because index is one less than the timestep value 
                        rawVasoTreatment = patientActionDataframe['RAW_VASO'][timestepNum_forActionQuery - 1]#-1 because index is one less than the timestep value 
                        IDFluidTreatment = patientActionDataframe['FLUID_ID'][timestepNum_forActionQuery - 1]
                        IDVasoTreatment = patientActionDataframe['VASO_ID'][timestepNum_forActionQuery - 1]
                        overallTreatment = patientActionDataframe['OVERALL_ACTION_ID'][timestepNum_forActionQuery - 1]

                        rawTreatment = [rawFluidTreatment, rawVasoTreatment]
                        IDTreatment = [IDFluidTreatment, IDVasoTreatment]
                        overallTreatment = overallTreatment

                        if (IDTreatment == [0, 0] or rawTreatment == [0.0, 0.0]):
                            continue
                        
                        K_neighbors_raw_treatments.append(rawTreatment)
                        K_neighbors_ID_treatments.append(IDTreatment)
                        K_neighbors_overall_treatments.append(overallTreatment)

                        kNeighborsAmount = kNeighborsAmount + 1

                    else:
                        break
                    
                



                #acquiring results
                #finding mode treatment of nearest neighbors (Raw and ID); max euclidean distance method
                #if-else used here as max euclidean distance method could potentially return 0 nearest neighbors if max value is too small
                if (len(nearest_neighbors_raw_treatments) > 0):
                    TUPLE_nearest_neighbors_raw_treatments =  map(tuple, nearest_neighbors_raw_treatments)
                    modeTreatment = list(col.Counter(TUPLE_nearest_neighbors_raw_treatments).most_common())
                    for i in range (len(modeTreatment)):
                        if (list(modeTreatment[i][0]) == [0, 0]):
                            continue
                        else:
                            predictedTreatment_nearest_raw = list(modeTreatment[i][0])
                            predictedTreatment_nearest_raw_occurence = modeTreatment[i][1]
                            break
                    
                    
                else:
                    predictedTreatment_nearest_raw = "null (no nearest neighbors found with given maximum ED)"
                    predictedTreatment_nearest_raw_occurence = "null"

                

                if (len(nearest_neighbors_ID_treatments) > 0):
                    TUPLE_nearest_neighbors_ID_treatments =  map(tuple, nearest_neighbors_ID_treatments)
                    modeTreatment = list(col.Counter(TUPLE_nearest_neighbors_ID_treatments).most_common())
                    for i in range (len(modeTreatment)):
                        if (list(modeTreatment[i][0]) == [0, 0]):
                            continue
                        else:
                            predictedTreatment_nearest_ID = list(modeTreatment[i][0])
                            predictedTreatment_nearest_ID_occurence = modeTreatment[i][1]
                            break
                    


                else:
                    predictedTreatment_nearest_ID = "null (no nearest neighbors found with given maximum ED)"
                    predictedTreatment_nearest_ID_occurence = "null"


                
                if(len(nearest_neighbors_overall_treatments) > 0):
                    #TUPLE_nearest_neighbors_overall_treatments = map(tuple, nearest_neighbors_overall_treatments)
                    modeTreatment = list(col.Counter(nearest_neighbors_overall_treatments).most_common())
                    for i in range (len(modeTreatment)):
                        if (modeTreatment[i][0] == 0):
                            continue
                        else:
                            predictedTreatment_nearest_overall = modeTreatment[i][0]
                            predictedTreatment_nearest_overall_occurence = modeTreatment[i][1]
                            break
                    

                else:
                    predictedTreatment_nearest_overall = "null (no nearest neighbors found with given maximum ED)"
                    predictedTreatment_nearest_overall_occurence = "null"

                
                
                TUPLE_K_neighbors_raw_treatments =  map(tuple, K_neighbors_raw_treatments)
                modeTreatment = list(col.Counter(TUPLE_K_neighbors_raw_treatments).most_common())
                for i in range (len(modeTreatment)):
                    if (list(modeTreatment[i][0]) == [0, 0]):
                        continue
                    else:
                        predictedTreatment_K_raw = list(modeTreatment[i][0])
                        predictedTreatment_K_raw_occurence = modeTreatment[i][1]
                        break

                
                TUPLE_K_neighbors_ID_treatments =  map(tuple, K_neighbors_ID_treatments)
                modeTreatment = list(col.Counter(TUPLE_K_neighbors_ID_treatments).most_common())
                for i in range (len(modeTreatment)):
                    if (list(modeTreatment[i][0]) == [0, 0]):
                        continue
                    else:
                        predictedTreatment_K_ID = list(modeTreatment[i][0])
                        predictedTreatment_K_ID_occurence = modeTreatment[i][1]
                        break


                #TUPLE_K_neighbors_overall_treatments = map(tuple, K_neighbors_overall_treatments)
                modeTreatment = list(col.Counter(K_neighbors_overall_treatments).most_common())
                for i in range (len(modeTreatment)):
                    if (modeTreatment[i][0] == 0):
                        continue
                    else:
                        predictedTreatment_K_overall = modeTreatment[i][0]
                        predictedTreatment_K_overall_occurence = modeTreatment[i][1]
                        break
            

                #results in the form [A, B] ==> [Fluid, Vaso]
                with open('results.csv', 'a') as resultsFile:
                    results = csv.writer(resultsFile)
                    if (rowNum == 1):
                        results.writerow(["Patient ID", "Timestep #", "K", "Max ED",  
                                            "Number of Neighbors - MaxED (dynamic)", "Number of Neighbors - K (static)", 
                                            "MaxED: Raw Treatment Found", "MaxED: Occurence in Neighbors",
                                            "MaxED: ID Treatment Found", "MaxED: Occurence in Neighbors", 
                                            "MaxED: Overall Treatment Found", "MaxED: Occurence in Neighbors", 
                                            "K: Raw Treatment Found", "K: Occurence in Neighbors", 
                                            "K: ID Treatment Found", "K: Occurence in Neighbors", 
                                            "K: Overall Treatment Found", "K: Occurence in Neighbors",
                                            "Actual Raw Treatment", "Occurence (MaxED Raw)", "Occurence (K Raw)", 
                                             "Actual ID Treatment", "Occurence (MaxED Raw)", "Occurence (K Raw)",
                                             "Actual Overall Treatment", "Occurence (MaxED Raw)", "Occurence (K Raw)"])
                    resultData = [PatientID1, timestep1, K, maxEuclideanDistance,
                                    nearestNeighborsAmount, kNeighborsAmount,
                                    predictedTreatment_nearest_raw, predictedTreatment_nearest_raw_occurence,
                                    predictedTreatment_nearest_ID, predictedTreatment_nearest_ID_occurence, 
                                    predictedTreatment_nearest_overall, predictedTreatment_nearest_overall_occurence, 
                                    predictedTreatment_K_raw, predictedTreatment_K_raw_occurence, 
                                    predictedTreatment_K_ID, predictedTreatment_K_ID_occurence, 
                                    predictedTreatment_K_overall, predictedTreatment_K_overall_occurence, 
                                    testPatientActualRawTreatment, findOccurenceOfActualRawTreatment(testPatientActualRawTreatment, nearest_neighbors_raw_treatments), 
                                    findOccurenceOfActualRawTreatment(testPatientActualRawTreatment, K_neighbors_raw_treatments), 
                                    testPatientActualIDTreatment, nearest_neighbors_ID_treatments.count(testPatientActualIDTreatment), 
                                    K_neighbors_ID_treatments.count(testPatientActualIDTreatment), 
                                    testPatientActualOverallTreatment, nearest_neighbors_overall_treatments.count(testPatientActualOverallTreatment), 
                                    K_neighbors_overall_treatments.count(testPatientActualOverallTreatment)]
                    results.writerow(resultData)


                
                

                
