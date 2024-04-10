
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from sklearn.cluster import KMeans

def getDirectoriesInDir(directory):
    all_files_and_dirs = os.listdir(directory)
    directories = [d for d in all_files_and_dirs if os.path.isdir(os.path.join(directory, d))]
    return directories

def processData(directories):
    firstRunTime = []
    firstRunObjectPosition = []
    firstRunRayCast = []
    firstRunWaypoint = []
    secondRunTime = []
    secondRunObjectPosition = []
    secondRunRayCast = []
    secondRunWaypoint = []
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            if os.path.splitext(file)[2] == '.meta':
                pass
            elif file == 'recordedData00.csv':
                df = pd.read_csv(directory + '/' + file)
                firstRunTime.append(df['Time'].values)
                firstRunObjectPosition.append([df['ObjectPositionX'].values, df['ObjectPositionY'].values, df['ObjectPositionZ'].values])
                firstRunRayCast.append(df['RayCastX'].values, df['RayCastY'].values, df['RayCastZ'].values)
                firstRunWaypoint.append(df['BlueSphereX'].values, df['BlueSphereY'].values, df['BlueSphereZ'].values)
            elif file == 'recordedData01.csv':
                df = pd.read_csv(directory + '/' + file)
                secondRunTime.append(df['Time'].values)
                secondRunObjectPosition.append([df['ObjectPositionX'].values, df['ObjectPositionY'].values, df['ObjectPositionZ'].values])
                secondRunRayCast.append(df['RayCastX'].values, df['RayCastY'].values, df['RayCastZ'].values)
                secondRunWaypoint.append(df['BlueSphereX'].values, df['BlueSphereY'].values, df['BlueSphereZ'].values)

def getAverageTime(timeList):
    timeTotal = 0
    for times in timeList:
        timeTotal += times[len(times)-1] - times[0]
    timeAverage = timeTotal / len(timeList)
    return timeAverage

def getAverageData(dataList):
    dataTot = 0
    dataAvg = 0
    for dataSet in dataList:
        dataTot = 0
        for data in dataSet:
            dataTot += np.sqrt(data[0]**2 + data[1]**2 + data[2]**2)
        dataAvg += dataTot / len(dataSet)
    dataAvg = dataAvg / len(dataList)
    return dataAvg

def getNumberAndCoordinatesOfWaypoints(df):
    # THIS GETS NUMBER OF WAYPOINTS PLACED ON PLACENTA
    # df is the data frame, feed in the raw df from the csv file
    rows_list = []
    numTimes = 0
    for i in range(0,len(df['BlueSphereX'])-2):
        if df['BlueSphereX'].iloc[i] != df['BlueSphereX'].iloc[i+1]:
            rows_list.append(df.iloc[i+1])
            print(i)
            numTimes+=1
    spot_df = pd.DataFrame(rows_list, columns=df.columns)
    for i in range(0,len(spot_df['RayCastX'])-1):
        if df['BlueSphereX'].iloc[i] == 0:
            # this is when the waypoint has been placed and the person has also placed a waypoint
            # we don't have to do anything here or we can do something here
            pass
    
    return numTimes, spot_df

def getLazerClusters(df, variable, centroid_bool = False):
    # THIS GETS THE LOCATION OF THE LAZER POINTS ON THE PLACENTA FOR ERROR ANALYSIS
    variableX = variable + 'X'
    variableY = variable + 'Y'
    variableZ = variable + 'Z'
    mask = (df[variableX] != 0) & (df[variableY] != 0) & (df[variableZ] != 0)
    new_df = df[mask]
    num = None
    # ONLY NEED TO LOOK AT X AND Z COORDINATES
    # if time:
        # data = new_df[['Time', variableX, variableZ]]
    data = new_df[[variableZ,variableX]]
    wcss = []
    for i in range(1, 30):
        kmeans = KMeans(n_clusters=i, n_init=12, random_state=0).fit(data)
        wcss.append(kmeans.inertia_)
    
    for i in range(0,len(wcss)-1):
        if (wcss[i]-wcss[i+1])/wcss[i] < 0.2:
            num = i
            # print(f"Num Clusters {variable}",i)
            break
    kmeans = KMeans(n_clusters=num, n_init=12, random_state=0).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return num, centroids, wcss

def errorAnalysis(waypoints, lasers):
    min_distance = 1000000000
    min_pair = None
    min_pair_list = []
    totalError = 0
    for waypoint in waypoints:
        for laser in lasers:
            distance = np.sqrt((waypoint[0] - laser[0])**2 + (waypoint[1] - laser[1])**2)
            if distance < min_distance:
                min_distance = distance
                min_pair = [waypoint, laser]    
        min_pair_list.append(min_pair)
        totalError += min_distance
    
    return totalError, min_pair_list

def getPathSmoothness(positionX, positionZ, time):

    cumulativeVelo = 0
    currentVelo = 0
    veloList = []
    for i in range(0, len(positionX)-1):
        currentVelo = np.sqrt((positionX[i+1] - positionX[i])**2 + (positionZ[i+1] - positionZ[i])**2) / (time[i+1] - time[i])
        cumulativeVelo += currentVelo
        veloList.append(currentVelo)
    averageVelo = cumulativeVelo / (len(veloList))

    currentAcceleration = 0
    accList = []
    cumulativeAcceleration = 0
    for i in range(0, len(veloList)-1):
        currentAcceleration = np.sqrt((veloList[i+1] - veloList[i])**2) / (time[i+1] - time[i])
        cumulativeAcceleration += currentAcceleration
        accList.append(currentAcceleration)
    averageAcceleration = cumulativeAcceleration / (len(accList))

    currentJerk = 0
    jerkList = []
    cumulativeJerk = 0
    for i in range(0, len(accList)-1):
        currentJerk = np.sqrt((accList[i+1] - accList[i])**2) / (time[i+1] - time[i])
        cumulativeJerk += currentJerk
        jerkList.append(currentJerk)
    averageJerk = cumulativeJerk / (len(jerkList))

    return averageVelo, averageAcceleration, averageJerk

def removeZeros(csv, var):
    # this takes a csv file and returns a dataframe
    # var will either be 'BlueSphere' or 'RayCast' or 'ObjectPosition
    df = pd.read_csv(csv)
    mask = (df[var + 'X'] != 0) & (df[var + 'Y'] != 0) & (df[var + 'Z'] != 0)
    new_df = df[mask].reset_index(drop=True)
    return new_df

def getCsvData(directory, case):
    # case is a string and is either: haptic, hololens, nothing, everything, or minimap
    caseUsers = directory + '\\' + case

    files = os.listdir(caseUsers)
    folders = []
    analysisData01 = []
    analysisData02 = []
    for i in range(0, len(files)):
        folders.append(caseUsers + '\\' + files[i])
    for i in range(0, len(folders)):
        csv_and_meta_files = os.listdir(folders[i])
        for j in csv_and_meta_files:
            if 'meta' in j:
                pass
            elif j == 'recordedData01.csv':
                print('processing recordedData01')
                csvPath = folders[i] + '\\' + j
                print(csvPath)
                laser_df01 = removeZeros(csvPath, 'RayCast')
                laser_clust01 = getLazerClusters(laser_df01, 'RayCast')
                laser_centroids01 = laser_clust01[1]

                waypoint_df01 = removeZeros(csvPath, 'BlueSphere')
                waypoint_clust01 = getLazerClusters(waypoint_df01, 'BlueSphere')
                waypoint_centroids01 = waypoint_clust01[1]
                
                numWaypoints01 = getNumberAndCoordinatesOfWaypoints(waypoint_df01)[0]

                error01 = errorAnalysis(waypoint_centroids01, laser_centroids01)[0]
                averageVelo01 = getPathSmoothness(laser_df01['RayCastX'], laser_df01['RayCastZ'], laser_df01['Time'])[0]
                averageAcc01 = getPathSmoothness(laser_df01['RayCastX'], laser_df01['RayCastZ'], laser_df01['Time'])[1]
                averageJerk01 = getPathSmoothness(laser_df01['RayCastX'], laser_df01['RayCastZ'], laser_df01['Time'])[2]
                analysisData01.append([numWaypoints01, error01, averageVelo01, averageAcc01, averageJerk01])
                print('\nnumber of waypoints: ', numWaypoints01, "\nnumber of waypoint centroids: ", len(waypoint_centroids01))

            elif j == 'recordedData02.csv':
                print('processing recordedData02')
                csvPath = folders[i] + '\\' + j
                print(csvPath)
                laser_df02 = removeZeros(csvPath, 'RayCast')
                laser_clust02 = getLazerClusters(laser_df02, 'RayCast')
                laser_centroids02 = laser_clust02[1]

                waypoint_df02 = removeZeros(csvPath, 'BlueSphere')
                waypoint_clust02 = getLazerClusters(waypoint_df02, 'BlueSphere')
                waypoint_centroids02 = waypoint_clust02[1]

                numWaypoints02 = getNumberAndCoordinatesOfWaypoints(waypoint_df02)[0]

                error02 = errorAnalysis(waypoint_centroids02, laser_centroids02)[0]
                averageVelo02 = getPathSmoothness(laser_df02['RayCastX'], laser_df02['RayCastZ'], laser_df02['Time'])[0]
                averageAcc02 = getPathSmoothness(laser_df02['RayCastX'], laser_df02['RayCastZ'], laser_df02['Time'])[1]
                averageJerk02 = getPathSmoothness(laser_df02['RayCastX'], laser_df02['RayCastZ'], laser_df02['Time'])[2]
                analysisData02.append([numWaypoints02, error02, averageVelo02, averageAcc02, averageJerk02])
                print('\nnumber of waypoints: ', numWaypoints02, "\nnumber of waypoint centroids: ", len(waypoint_centroids02))

    avgNumWaypoints01 = 0
    avgError01 = 0
    avgVelo01 = 0
    avgAcc01 = 0
    avgJerk01 = 0
    for i in range(0, len(analysisData01)):
        avgNumWaypoints01 += analysisData01[i][0]
        avgError01 += analysisData01[i][1]
        avgVelo01 += analysisData01[i][2]
        avgAcc01 += analysisData01[i][3]
        avgJerk01 += analysisData01[i][4]
    avgNumWaypoints01 = avgNumWaypoints01 / len(analysisData01)
    avgError01 = avgError01 / len(analysisData01)
    avgVelo01 = avgVelo01 / len(analysisData01)
    avgAcc01 = avgAcc01 / len(analysisData01)
    avgJerk01 = avgJerk01 / len(analysisData01)

    avgNumWaypoints02 = 0
    avgError02 = 0
    avgVelo02 = 0
    avgAcc02 = 0
    avgJerk02 = 0
    for i in range(0, len(analysisData02)):
        avgNumWaypoints02 += analysisData02[i][0]
        avgError02 += analysisData02[i][1]
        avgVelo02 += analysisData02[i][2]
        avgAcc02 += analysisData02[i][3]
        avgJerk02 += analysisData02[i][4]
    avgNumWaypoints02 = avgNumWaypoints02 / len(analysisData02)
    avgError02 = avgError02 / len(analysisData02)
    avgVelo02 = avgVelo02 / len(analysisData02)
    avgAcc02 = avgAcc02 / len(analysisData02)
    avgJerk02 = avgJerk02 / len(analysisData02)
    avgs01 = [avgNumWaypoints01, avgError01, avgVelo01, avgAcc01, avgJerk01]
    avgs02 = [avgNumWaypoints02, avgError02, avgVelo02, avgAcc02, avgJerk02]
    return avgs01, avgs02