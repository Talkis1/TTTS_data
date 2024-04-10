
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from sklearn.cluster import KMeans
import warnings
from sklearn.exceptions import ConvergenceWarning
import math

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
    data = new_df[[variableX,variableZ]]
    num_samples = data.shape[0]
    max_clusters = min(num_samples, 30)
    wcss = []
        
    for i in range(1, max_clusters):
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
    time =  df['Time'].iloc[-1] - df['Time'].iloc[0]
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
                df = pd.read_csv(csvPath)
                if not df.empty and len(df['Time']) > 0:
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
                df = pd.read_csv(csvPath)
                if not df.empty and len(df['Time']) > 0:
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

    # Assuming analysisData01 is your dataset and is already defined
    n = len(analysisData01)  # Number of observations

    # Initialize sums for averages
    sumNumWaypoints01 = sumError01 = sumVelo01 = sumAcc01 = sumJerk01 = sumDistance01 = sumTime01 = sumWaypointErrorComp01 = sumLaserErrorComp01 = 0

    # Initialize sums for standard deviation calculations (sum of squared deviations)
    sq_dev_NumWaypoints01 = sq_dev_Error01 = sq_dev_Velo01 = sq_dev_Acc01 = sq_dev_Jerk01 = sq_dev_Distance01 = sq_dev_Time01 = sq_dev_WaypointErrorComp01 = sq_dev_LaserErrorComp01 = 0

    # Calculate sums for averages
    for i in range(n):
        sumNumWaypoints01 += analysisData01[i][0]
        sumError01 += analysisData01[i][1]
        sumVelo01 += analysisData01[i][2]
        sumAcc01 += analysisData01[i][3]
        sumJerk01 += analysisData01[i][4]
        sumDistance01 += analysisData01[i][5]
        sumTime01 += analysisData01[i][6]
        sumWaypointErrorComp01 += analysisData01[i][7]
        sumLaserErrorComp01 += analysisData01[i][8]

    # Calculate averages
    avgNumWaypoints01 = sumNumWaypoints01 / n
    avgError01 = sumError01 / n
    avgVelo01 = sumVelo01 / n
    avgAcc01 = sumAcc01 / n
    avgJerk01 = sumJerk01 / n
    avgDistance01 = sumDistance01 / n
    avgTime01 = sumTime01 / n
    avgWaypointErrorComp01 = sumWaypointErrorComp01 / n
    avgLaserErrorComp01 = sumLaserErrorComp01 / n

    # Calculate squared deviations from the mean
    for i in range(n):
        sq_dev_NumWaypoints01 += (analysisData01[i][0] - avgNumWaypoints01) ** 2
        sq_dev_Error01 += (analysisData01[i][1] - avgError01) ** 2
        sq_dev_Velo01 += (analysisData01[i][2] - avgVelo01) ** 2
        sq_dev_Acc01 += (analysisData01[i][3] - avgAcc01) ** 2
        sq_dev_Jerk01 += (analysisData01[i][4] - avgJerk01) ** 2
        sq_dev_Distance01 += (analysisData01[i][5] - avgDistance01) ** 2
        sq_dev_Time01 += (analysisData01[i][6] - avgTime01) ** 2
        sq_dev_WaypointErrorComp01 += (analysisData01[i][7] - avgWaypointErrorComp01) ** 2
        sq_dev_LaserErrorComp01 += (analysisData01[i][8] - avgLaserErrorComp01) ** 2

    # Calculate standard deviations
    stdNumWaypoints01 = math.sqrt(sq_dev_NumWaypoints01 / n)
    stdError01 = math.sqrt(sq_dev_Error01 / n)
    stdVelo01 = math.sqrt(sq_dev_Velo01 / n)
    stdAcc01 = math.sqrt(sq_dev_Acc01 / n)
    stdJerk01 = math.sqrt(sq_dev_Jerk01 / n)
    stdDistance01 = math.sqrt(sq_dev_Distance01 / n)
    stdTime01 = math.sqrt(sq_dev_Time01 / n)
    stdWaypointErrorComp01 = math.sqrt(sq_dev_WaypointErrorComp01 / n)
    stdLaserErrorComp01 = math.sqrt(sq_dev_LaserErrorComp01 / n)
    
    # Assuming analysisData02 is your dataset and is already defined
    n = len(analysisData02)  # Number of observations

    # Initialize sums for averages
    sumNumWaypoints02 = sumError02 = sumVelo02 = sumAcc02 = sumJerk02 = sumDistance02 = sumTime02 = sumWaypointErrorComp02 = sumLaserErrorComp02 = 0

    # Initialize sums for standard deviation calculations (sum of squared deviations)
    sq_dev_NumWaypoints02 = sq_dev_Error02 = sq_dev_Velo02 = sq_dev_Acc02 = sq_dev_Jerk02 = sq_dev_Distance02 = sq_dev_Time02 = sq_dev_WaypointErrorComp02 = sq_dev_LaserErrorComp02 = 0

    # Calculate sums for averages
    for i in range(n):
        sumNumWaypoints02 += analysisData02[i][0]
        sumError02 += analysisData02[i][1]
        sumVelo02 += analysisData02[i][2]
        sumAcc02 += analysisData02[i][3]
        sumJerk02 += analysisData02[i][4]
        sumDistance02 += analysisData02[i][5]
        sumTime02 += analysisData02[i][6]
        sumWaypointErrorComp02 += analysisData02[i][7]
        sumLaserErrorComp02 += analysisData02[i][8]

    # Calculate averages
    avgNumWaypoints02 = sumNumWaypoints02 / n
    avgError02 = sumError02 / n
    avgVelo02 = sumVelo02 / n
    avgAcc02 = sumAcc02 / n
    avgJerk02 = sumJerk02 / n
    avgDistance02 = sumDistance02 / n
    avgTime02 = sumTime02 / n
    avgWaypointErrorComp02 = sumWaypointErrorComp02 / n
    avgLaserErrorComp02 = sumLaserErrorComp02 / n

    # Calculate squared deviations from the mean
    for i in range(n):
        sq_dev_NumWaypoints02 += (analysisData02[i][0] - avgNumWaypoints02) ** 2
        sq_dev_Error02 += (analysisData02[i][1] - avgError02) ** 2
        sq_dev_Velo02 += (analysisData02[i][2] - avgVelo02) ** 2
        sq_dev_Acc02 += (analysisData02[i][3] - avgAcc02) ** 2
        sq_dev_Jerk02 += (analysisData02[i][4] - avgJerk02) ** 2
        sq_dev_Distance02 += (analysisData02[i][5] - avgDistance02) ** 2
        sq_dev_Time02 += (analysisData02[i][6] - avgTime02) ** 2
        sq_dev_WaypointErrorComp02 += (analysisData02[i][7] - avgWaypointErrorComp02) ** 2
        sq_dev_LaserErrorComp02 += (analysisData02[i][8] - avgLaserErrorComp02) ** 2

    # Calculate standard deviations
    stdNumWaypoints02 = math.sqrt(sq_dev_NumWaypoints02 / n)
    stdError02 = math.sqrt(sq_dev_Error02 / n)
    stdVelo02 = math.sqrt(sq_dev_Velo02 / n)
    stdAcc02 = math.sqrt(sq_dev_Acc02 / n)
    stdJerk02 = math.sqrt(sq_dev_Jerk02 / n)
    stdDistance02 = math.sqrt(sq_dev_Distance02 / n)
    stdTime02 = math.sqrt(sq_dev_Time02 / n)
    stdWaypointErrorComp02 = math.sqrt(sq_dev_WaypointErrorComp02 / n)
    stdLaserErrorComp02 = math.sqrt(sq_dev_LaserErrorComp02 / n)

    avgs01 = [[avgNumWaypoints01,stdNumWaypoints01], [avgError01,stdError01], [avgVelo01, stdVelo01]
              , [avgAcc01,stdAcc01], [avgJerk01,stdJerk01], [avgDistance01,stdDistance01]
              , [avgTime01,stdTime01], [avgWaypointErrorComp01,stdWaypointErrorComp01]
              , [avgLaserErrorComp01,stdLaserErrorComp01]]
    avgs02 = [[avgNumWaypoints02,stdNumWaypoints02], [avgError02,stdError02], [avgVelo02, stdVelo02]
                , [avgAcc02,stdAcc02], [avgJerk02,stdJerk02], [avgDistance02,stdDistance02]
                , [avgTime02,stdTime02], [avgWaypointErrorComp02,stdWaypointErrorComp02]
                , [avgLaserErrorComp02,stdLaserErrorComp02]]
    # avgs02 = [avgNumWaypoints02, avgError02, avgVelo02, avgAcc02, avgJerk02, avgDistance02, avgTime02, avgWaypointErrorComp02, avgLaserErrorComp02]
    return avgs01, avgs02, optimalDataComp01, optimalDataComp02

def plotData(directory, case):
    curdir = os.getcwd()
    print(curdir)
    caseUsers = directory + '\\' + case
    optimalCsvFile01 = curdir + '\\testSubjecPathData\\tristanOptimalPath\\recordedData01.csv'
    optimalCsvFile02 = curdir + '\\testSubjecPathData\\tristanOptimalPath\\recordedData02.csv'
    files = os.listdir(caseUsers)
    folders = []
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    iteration = 0
    df1s = []
    df1s_wp = []
    df2s = []
    df2s_wp = []
    for i in range(0, len(files)):
        folders.append(caseUsers + '\\' + files[i])
    for i in range(0, len(folders)):
        csv_and_meta_files = os.listdir(folders[i])
        for j in csv_and_meta_files:
            if 'meta' in j:
                pass
            elif j == 'recordedData01.csv':
                iteration +=1
                csvPath = folders[i] + '\\' + j
                df1 = pd.read_csv(csvPath)
                if not df1.empty and len(df1['Time']) > 0:
                    mask1 = (df1['RayCastX'] != 0) & (df1['RayCastY'] != 0) & (df1['RayCastZ'] != 0)
                    new_df1 = df1[mask1].reset_index(drop=True)
                    new_df1.loc[:, 'RayCastX'] = round(new_df1.loc[:, 'RayCastX'], 3)
                    # Assuming 'df' is your DataFrame and 'col' is the column where you want to remove duplicates
                    new_df1 = new_df1.drop_duplicates(subset='RayCastX', keep='first')
                    spheres = getBSpheres(df1)[1]
                    df1s.append(new_df1)
                    df1s_wp.append(spheres)

                # ax1.scatter(new_df1['RayCastX'], new_df1['RayCastZ'],label = 'novice')
            elif j == 'recordedData02.csv':
                csvPath = folders[i] + '\\' + j
                df2 = pd.read_csv(csvPath)
                if not df2.empty and len(df2['Time']) > 0:
                    mask2 = (df2['RayCastX'] != 0) & (df2['RayCastY'] != 0) & (df2['RayCastZ'] != 0)
                    new_df2 = df2[mask2].reset_index(drop=True)
                    new_df2.loc[:, 'RayCastX'] = round(new_df2.loc[:, 'RayCastX'], 3)
                    # Assuming 'df' is your DataFrame and 'col' is the column where you want to remove duplicates
                    new_df2 = new_df2.drop_duplicates(subset='RayCastX', keep='first')
                    new_df1 = new_df1.drop_duplicates(subset='RayCastX', keep='first')
                    spheres = getBSpheres(df2)[1]               
                    df2s.append(new_df2)
                    df2s_wp.append(spheres)

    df1 = pd.read_csv(optimalCsvFile01)
    mask1 = (df1['RayCastX'] != 0) & (df1['RayCastY'] != 0) & (df1['RayCastZ'] != 0)
    new_df1 = df1[mask1].reset_index(drop=True)
    coordinatesList = []
    # for df in df1s:
    #     print(isinstance(df, pd.DataFrame))

    all_dfs = pd.concat(df1s)

    # Group by 'x' and calculate the mean of 'y' values
    grouped = all_dfs.groupby('RayCastX', as_index=False)['RayCastZ'].mean()

    # Create the list of (x, y) pairs
    xy_pairs = list(grouped.itertuples(index=False, name=None))
    xs, ys = zip(*xy_pairs)

    allSpheres = pd.concat(df1s_wp)
    sphereCentroids = getLazerClusters(allSpheres, 'BlueSphere')[1]

    # ax1.scatter(xs_wp,ys_wp, label = 'waypoints')
    sphereCentroidsExpert = getBSpheres(df1)[1]
    
    ax1.scatter(xs,ys, label = 'Averaged Novice Path')
    ax1.scatter(new_df1['RayCastX'], new_df1['RayCastZ'], label = 'Expert Optimal Path')
    ax1.scatter(sphereCentroids[:,0],sphereCentroids[:,1], label = 'Novice Waypoints')
    ax1.scatter(sphereCentroidsExpert['BlueSphereX'],sphereCentroidsExpert['BlueSphereZ'], label = 'Expert Waypoints')

    ax1.set_xlabel('Horizontal Position on Placenta')  # Add your X-axis label here
    ax1.set_ylabel('Vertical Position on Placenta')  # Add your Y-axis label here
    ax1.set_title('First Trial Laser Path (No Assistance)')  # Add your title here
    ax1.legend()  # This will add the legend to the plot

    all_dfs = pd.concat(df2s)

    # Group by 'x' and calculate the mean of 'y' values
    grouped = all_dfs.groupby('RayCastX', as_index=False)['RayCastZ'].mean()

    # Create the list of (x, y) pairs
    xy_pairs = list(grouped.itertuples(index=False, name=None))
    xs, ys = zip(*xy_pairs)

    allSpheres = pd.concat(df2s_wp)
    sphereCentroids = getLazerClusters(allSpheres, 'BlueSphere')[1]


    # ax1.scatter(xs_wp,ys_wp, label = 'waypoints')
    
    # ax2.scatter(xs_wp,ys_wp, label = 'waypoints')
    ax2.scatter(xs,ys, label = 'Averaged Novice Path')
    df2 = pd.read_csv(optimalCsvFile02)
    mask2 = (df2['RayCastX'] != 0) & (df2['RayCastY'] != 0) & (df2['RayCastZ'] != 0)
    new_df2 = df2[mask2].reset_index(drop=True)
    sphereCentroidsExpert = getBSpheres(df2)[1]

    ax2.scatter(new_df2['RayCastX'], new_df2['RayCastZ'], label = 'Expert Optimal Path')
    ax2.scatter(sphereCentroids[:,0],sphereCentroids[:,1], label = 'Novice Waypoints')
    ax2.scatter(sphereCentroidsExpert['BlueSphereX'],sphereCentroidsExpert['BlueSphereZ'], label = 'Expert Waypoints')
    ax2.set_xlabel('Horizontal Position on Placenta')  # Add your X-axis label here
    ax2.set_ylabel('Vertical Position on Placenta')  # Add your Y-axis label here
    ax2.set_title(f'Second Trial Laser Path ({case} assistance)')  # Add your title here
    ax2.legend()  # This will add the legend to the plot
    
    return df1s, df2s