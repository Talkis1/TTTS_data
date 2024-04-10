
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from sklearn.cluster import KMeans
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def getDirectoriesInDir(directory):
    all_files_and_dirs = os.listdir(directory)
    directories = [d for d in all_files_and_dirs if os.path.isdir(os.path.join(directory, d))]
    return directories

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
    mask1 = (df['BlueSphereX'] != 0) & (df['BlueSphereY'] != 0) & (df['BlueSphereZ'] != 0)
    new_df = df[mask1].reset_index(drop=True).round(4)
    new_df = new_df.drop_duplicates(subset=['BlueSphereX', 'BlueSphereY', 'BlueSphereZ'], keep='first')
    return len(new_df), new_df

def getBSpheres(df):
    mask1 = (df['BlueSphereX'] != 0) & (df['BlueSphereY'] != 0) & (df['BlueSphereZ'] != 0)
    new_df = df[mask1].reset_index(drop=True).round(4)
    new_df = new_df.drop_duplicates(subset=['BlueSphereX', 'BlueSphereY', 'BlueSphereZ'], keep='first')
    return len(new_df), new_df

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
        n_clusters = len(set(kmeans.labels_)) - (1 if -1 in kmeans.labels_ else 0)
        # print(f'Number of distinct clusters for {i} clusters: ', n_clusters)
    
    for i in range(0,len(wcss)-1):
        if (wcss[i]-wcss[i+1])/wcss[i] < 0.2:
            if i+1 > n_clusters:
                num = n_clusters
            else:
                num = i + 1
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
    totalDistance = 0
    for i in range(0, len(positionX)-1):
        totalDistance += np.sqrt((positionX[i+1] - positionX[i])**2 + (positionZ[i+1] - positionZ[i])**2)
    
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

    return averageVelo, averageAcceleration, averageJerk, totalDistance

def removeZeros(csv, var):
    # this takes a csv file and returns a dataframe
    # var will either be 'BlueSphere' or 'RayCast' or 'ObjectPosition
    df = pd.read_csv(csv)
    mask = (df[var + 'X'] != 0) & (df[var + 'Y'] != 0) & (df[var + 'Z'] != 0)
    new_df = df[mask].reset_index(drop=True)
    return new_df

def getTimeTotal(csv):
    df = pd.read_csv(csv)
    time = df['Time'].iloc[0] - df['Time'].iloc[-1]
    return time

def getSingleCsvData(csv):
    op_time01 = getTimeTotal(csv)
    op_laser_df01 = removeZeros(csv, 'RayCast')
    op_laser_clust01 = getLazerClusters(op_laser_df01, 'RayCast')
    op_laser_centroids01 = op_laser_clust01[1]

    op_waypoint_df01 = removeZeros(csv, 'BlueSphere')
    op_waypoint_clust01 = getLazerClusters(op_waypoint_df01, 'BlueSphere')
    op_waypoint_centroids01 = op_waypoint_clust01[1]
    
    op_numWaypoints01 = getNumberAndCoordinatesOfWaypoints(op_waypoint_df01)[0]

    op_error01 = errorAnalysis(op_waypoint_centroids01, op_laser_centroids01)[0]
    op_pathSmoothness01 = getPathSmoothness(op_laser_df01['RayCastX'], op_laser_df01['RayCastZ'], op_laser_df01['Time'])
    op_averageVelo01 = op_pathSmoothness01[0]
    op_averageAcc01 = op_pathSmoothness01[1]
    op_averageJerk01 = op_pathSmoothness01[2]
    op_distance01 = op_pathSmoothness01[3]
    optimalDataComp01 = [op_numWaypoints01, op_error01, op_averageVelo01, op_averageAcc01, op_averageJerk01, op_distance01, op_time01]
    return optimalDataComp01

def getCsvData(directory, case):
    # case is a string and is either: haptic, hololens, nothing, everything, or minimap
    caseUsers = directory + '\\' + case
    optimalCsvFile01 = directory + '\\tristanOptimalPath\\recordedData01.csv'
    optimalCsvFile02 = directory + '\\tristanOptimalPath\\recordedData02.csv'
    files = os.listdir(caseUsers)
    folders = []
    analysisData01 = []
    analysisData02 = []

    # OPTIMAL FIRST RUN DATA
    op_time01 = getTimeTotal(optimalCsvFile01)
    op_laser_df01 = removeZeros(optimalCsvFile01, 'RayCast')
    op_laser_clust01 = getLazerClusters(op_laser_df01, 'RayCast')
    op_laser_centroids01 = op_laser_clust01[1]

    op_waypoint_df01 = removeZeros(optimalCsvFile01, 'BlueSphere')
    op_waypoint_clust01 = getLazerClusters(op_waypoint_df01, 'BlueSphere')
    op_waypoint_centroids01 = op_waypoint_clust01[1]
    
    op_numWaypoints01 = getNumberAndCoordinatesOfWaypoints(op_waypoint_df01)[0]

    op_error01 = errorAnalysis(op_waypoint_centroids01, op_laser_centroids01)[0]
    op_pathSmoothness01 = getPathSmoothness(op_laser_df01['RayCastX'], op_laser_df01['RayCastZ'], op_laser_df01['Time'])
    op_averageVelo01 = op_pathSmoothness01[0]
    op_averageAcc01 = op_pathSmoothness01[1]
    op_averageJerk01 = op_pathSmoothness01[2]
    op_distance01 = op_pathSmoothness01[3]
    optimalDataComp01 = [op_numWaypoints01, op_error01, op_averageVelo01, op_averageAcc01, op_averageJerk01, op_distance01, op_time01]

    print('\nnumber of waypoints: ', op_numWaypoints01, "\nnumber of waypoint centroids: ", len(op_waypoint_centroids01), 
          "\nnumber of laser centroids: ", len(op_laser_centroids01))
    # OPTIMAL SECOND RUN DATA
    op_time02 = getTimeTotal(optimalCsvFile02)
    op_laser_df02 = removeZeros(optimalCsvFile02, 'RayCast')
    op_laser_clust02 = getLazerClusters(op_laser_df02, 'RayCast')
    op_laser_centroids02 = op_laser_clust02[1]

    op_waypoint_df02 = removeZeros(optimalCsvFile02, 'BlueSphere')
    op_waypoint_clust02 = getLazerClusters(op_waypoint_df02, 'BlueSphere')
    op_waypoint_centroids02 = op_waypoint_clust02[1]

    op_numWaypoints02 = getNumberAndCoordinatesOfWaypoints(op_waypoint_df02)[0]

    op_error02 = errorAnalysis(op_waypoint_centroids02, op_laser_centroids02)[0]
    op_pathSmoothness02 = getPathSmoothness(op_laser_df02['RayCastX'], op_laser_df02['RayCastZ'], op_laser_df02['Time'])
    op_averageVelo02 = op_pathSmoothness02[0]
    op_averageAcc02 = op_pathSmoothness02[1]
    op_averageJerk02 = op_pathSmoothness02[2]
    op_distance02 = op_pathSmoothness02[3]
    optimalDataComp02 = [op_numWaypoints02, op_error02, op_averageVelo02, op_averageAcc02, op_averageJerk02, op_distance02, op_time02]

    print('\nnumber of waypoints: ', op_numWaypoints02, "\nnumber of waypoint centroids: ", len(op_waypoint_centroids02), 
      "\nnumber of laser centroids: ", len(op_laser_centroids02))
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
                time01 = getTimeTotal(csvPath)
                laser_df01 = removeZeros(csvPath, 'RayCast')
                laser_clust01 = getLazerClusters(laser_df01, 'RayCast')
                laser_centroids01 = laser_clust01[1]

                waypoint_df01 = removeZeros(csvPath, 'BlueSphere')
                waypoint_clust01 = getLazerClusters(waypoint_df01, 'BlueSphere')
                waypoint_centroids01 = waypoint_clust01[1]
                
                numWaypoints01 = getNumberAndCoordinatesOfWaypoints(waypoint_df01)[0]

                error01 = errorAnalysis(waypoint_centroids01, laser_centroids01)[0]
                pathSmoothness01 = getPathSmoothness(op_laser_df01['RayCastX'], op_laser_df01['RayCastZ'], op_laser_df01['Time'])
                averageVelo01 = pathSmoothness01[0]
                averageAcc01 = pathSmoothness01[1]
                averageJerk01 = pathSmoothness01[2]
                distance01 = pathSmoothness01[3]
                waypoint_errorComp01 = errorAnalysis(op_waypoint_centroids01, waypoint_centroids01)[0]
                laser_errorComp01 = errorAnalysis(op_laser_centroids01, laser_centroids01)[0]
                analysisData01.append([numWaypoints01, error01, averageVelo01, averageAcc01, averageJerk01, distance01, time01, waypoint_errorComp01, laser_errorComp01])
                print('\nnumber of waypoints: ', numWaypoints01, "\nnumber of waypoint centroids: ", len(waypoint_centroids01), 
                    "\nnumber of laser centroids: ", len(laser_centroids01))

            elif j == 'recordedData02.csv':
                print('processing recordedData02')
                csvPath = folders[i] + '\\' + j
                print(csvPath)
                time02 = getTimeTotal(csvPath)
                laser_df02 = removeZeros(csvPath, 'RayCast')
                laser_clust02 = getLazerClusters(laser_df02, 'RayCast')
                laser_centroids02 = laser_clust02[1]

                waypoint_df02 = removeZeros(csvPath, 'BlueSphere')
                waypoint_clust02 = getLazerClusters(waypoint_df02, 'BlueSphere')
                waypoint_centroids02 = waypoint_clust02[1]

                numWaypoints02 = getNumberAndCoordinatesOfWaypoints(waypoint_df02)[0]

                error02 = errorAnalysis(waypoint_centroids02, laser_centroids02)[0]
                pathSmoothness02 = getPathSmoothness(op_laser_df02['RayCastX'], op_laser_df02['RayCastZ'], op_laser_df02['Time'])
                averageVelo02 = pathSmoothness02[0]
                averageAcc02 = pathSmoothness02[1]
                averageJerk02 = pathSmoothness02[2]
                distance02 = pathSmoothness02[3]
                waypoint_errorComp02 = errorAnalysis(op_waypoint_centroids02, waypoint_centroids02)[0]
                laser_errorComp02 = errorAnalysis(op_laser_centroids02, laser_centroids02)[0]
                analysisData02.append([numWaypoints02, error02, averageVelo02, averageAcc02, averageJerk02, distance02, time02, waypoint_errorComp02, laser_errorComp02])
                print('\nnumber of waypoints: ', numWaypoints02, "\nnumber of waypoint centroids: ", len(waypoint_centroids02), 
                    "\nnumber of laser centroids: ", len(laser_centroids02))

    avgNumWaypoints01 = 0
    avgError01 = 0
    avgVelo01 = 0
    avgAcc01 = 0
    avgJerk01 = 0
    avgDistance01 = 0
    avgTime01 = 0
    avgWaypointErrorComp01 = 0
    avgLaserErrorComp01 = 0
    for i in range(0, len(analysisData01)):
        avgNumWaypoints01 += analysisData01[i][0]
        avgError01 += analysisData01[i][1]
        avgVelo01 += analysisData01[i][2]
        avgAcc01 += analysisData01[i][3]
        avgJerk01 += analysisData01[i][4]
        avgDistance01 += analysisData01[i][5]
        avgTime01 += analysisData01[i][6]
        avgWaypointErrorComp01 += analysisData01[i][7]
        avgLaserErrorComp01 += analysisData01[i][8]

    avgNumWaypoints01 = avgNumWaypoints01 / len(analysisData01)
    avgError01 = avgError01 / len(analysisData01)
    avgVelo01 = avgVelo01 / len(analysisData01)
    avgAcc01 = avgAcc01 / len(analysisData01)
    avgJerk01 = avgJerk01 / len(analysisData01)
    avgDistance01 = avgDistance01 / len(analysisData01)
    avgTime01 = avgTime01 / len(analysisData01)
    avgWaypointErrorComp01 = avgWaypointErrorComp01 / len(analysisData01)
    avgLaserErrorComp01 = avgLaserErrorComp01 / len(analysisData01)

    avgNumWaypoints02 = 0
    avgError02 = 0
    avgVelo02 = 0
    avgAcc02 = 0
    avgJerk02 = 0
    avgDistance02 = 0
    avgTime02 = 0
    avgWaypointErrorComp02 = 0
    avgLaserErrorComp02 = 0
    for i in range(0, len(analysisData02)):
        avgNumWaypoints02 += analysisData02[i][0]
        avgError02 += analysisData02[i][1]
        avgVelo02 += analysisData02[i][2]
        avgAcc02 += analysisData02[i][3]
        avgJerk02 += analysisData02[i][4]
        avgDistance02 += analysisData02[i][5]
        avgTime02 += analysisData02[i][6]
        avgWaypointErrorComp02 += analysisData02[i][7]
        avgLaserErrorComp02 += analysisData02[i][8]

    avgNumWaypoints02 = avgNumWaypoints02 / len(analysisData02)
    avgError02 = avgError02 / len(analysisData02)
    avgVelo02 = avgVelo02 / len(analysisData02)
    avgAcc02 = avgAcc02 / len(analysisData02)
    avgJerk02 = avgJerk02 / len(analysisData02)
    avgDistance02 = avgDistance02 / len(analysisData02)
    avgTime02 = avgTime02 / len(analysisData02)
    avgWaypointErrorComp02 = avgWaypointErrorComp02 / len(analysisData02)
    avgLaserErrorComp02 = avgLaserErrorComp02 / len(analysisData02)


    avgs01 = [avgNumWaypoints01, avgError01, avgVelo01, avgAcc01, avgJerk01, avgDistance01, avgTime01, avgWaypointErrorComp01, avgLaserErrorComp01]
    avgs02 = [avgNumWaypoints02, avgError02, avgVelo02, avgAcc02, avgJerk02, avgDistance02, avgTime02, avgWaypointErrorComp02, avgLaserErrorComp02]
    return avgs01, avgs02, optimalDataComp01, optimalDataComp01

def plotData(data, title):
     for i in range(0, len(files)):
        folders.append(caseUsers + '\\' + files[i])
    for i in range(0, len(folders)):
        csv_and_meta_files = os.listdir(folders[i])
        for j in csv_and_meta_files:
            if 'meta' in j:
                pass
            elif j == 'recordedData01.csv':