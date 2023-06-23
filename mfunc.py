import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy import interpolate

def get_cartesian(lon,lat):
#     '''''
#     convert (longitude, latitude) points to 2D plane coordinate (X, Y).
#     1 degree of latidute = 111.32 km
#     1 degree of longitude = 2*\pi*R*\cos(lat)/360 where R ~ 6371 km
#     Step:
#     1. Choose centre points of given coordinate as the origin of coordinate (lon_c,lat_c) -> (0,0)
#     2. Corresponding point (x,y) for the given geographical coord (lon,lat) will be
#         x = (lon-lon_c) * 2*\pi*6371*10^3*\cos(lat)/360 [m]
#         y = (lat - lat_c) * 111.32 * 10^3 [m]
#     '''''
    # prepare params
    delta_lat = 111.32 * np.power(10,3) # [meter]
    R_earth = 6371 * np.power(10,3) # [meter]
    lon_c = np.median(np.array(lon))  # [degree]
    lat_c = np.median(np.array(lat))  # [degree]
    # conversion
    X = (np.array(lon)-lon_c) * 2.0*np.pi*R_earth*np.cos(lat_c*np.pi/180)/360 # [meter]
    Y = (np.array(lat)-lat_c) * delta_lat # [meter]
    return X, Y, [lon_c,lat_c]

def file_reader(path,CONVERT_COORD=False):
    df_table = pd.read_table(path,header=None)
    data = pd.DataFrame()
#     data['x'] = df_table[0] # longitude
    x_mod = np.zeros(len(df_table[0].values))
    for i,x in enumerate(df_table[0].values):
        if x > 0:
            x_mod[i] = x
        if x < 0:
            x_mod[i] = 360 + x
    data['x'] = x_mod[np.where(np.isnan(df_table[2].values)==False)]
    data['y'] = df_table[1].values[np.where(np.isnan(df_table[2].values)==False)] # latitude
    data['z'] = df_table[2].values[np.where(np.isnan(df_table[2].values)==False)] # elevation
    if CONVERT_COORD:
        xc,yc, center = get_cartesian(data['x'],data['y'])
        data['x'] = xc
        data['y'] = yc
        return data, center
    else:
        return data

def elevation_percentileRange_from_path_list(path_list,minp=5,maxp=95):
    minE = []
    maxE = []
    for path in path_list:
        data = file_reader(path)
        minE.append(np.percentile(data['z'],minp))
        maxE.append(np.percentile(data['z'],maxp))
    return (np.min(minE),np.max(maxE))

def channel_interpolator(data,sigma=1,kind='cubic'):
    Z = data.pivot_table(index='x', columns='y', values='z').T.values # Elevation
    X_unique = np.sort(data.x.unique()) # X_axis 
    Y_unique = np.sort(data.y.unique()) # Y_axis
    X, Y = np.meshgrid(X_unique, Y_unique) # grid
    grad = np.gradient(gaussian_filter(Z,sigma=sigma)) # grid of local slope(Sx,Sy)
    mag = np.sqrt(grad[0]**2 + grad[1]**2) # local slope grid (Sxy)
    fe = interpolate.interp2d(X_unique, Y_unique, gaussian_filter(Z,sigma=sigma), kind='cubic') # Interepolated elevation
    fm = interpolate.interp2d(X_unique, Y_unique, mag, kind='cubic') # Interpolated slope
    return (X,Y,Z,mag), (fe, fm)

def cumulative_distance(CX,CY,CZ):
    distance_array = np.zeros(len(CX))
    for i in range(len(CX)):
        if i == 0:
            distance_array[0] = 0
        else:
            distance_array[i] = np.sqrt((CX[i]-CX[i-1])**2 + (CY[i]-CY[i-1])**2 + (CZ[i]-CZ[i-1])**2)
    CD = np.cumsum(distance_array)
    return CD

def read_centreline_fromXYZ(path,grids,fields,cut=True):
    X, Y, Z, mag = grids # Grid data
    fe, fm = fields # interpolated grid of elevation and slope
    df = pd.read_table(path)
    CX = np.array(df['Longitude'].values)
    x_mod = np.zeros(len(CX))
    for i,x in enumerate(CX):
        if x > 0:
            x_mod[i] = x
        if x < 0:
            x_mod[i] = 360 + x
    CX = x_mod
    CY = np.array(df['Latitude'].values)
    if cut:
        CX = CX[np.where(CX <= np.max(X))]; CY = CY[np.where(CX <= np.max(X))]
        CX = CX[np.where(CX >= np.min(X))]; CY = CY[np.where(CX >= np.min(X))]
        CX = CX[np.where(CY <= np.max(Y))]; CY = CY[np.where(CY <= np.max(Y))]
        CX = CX[np.where(CY >= np.min(Y))]; CY = CY[np.where(CY >= np.min(Y))]
    CX = list(CX); CY = list(CY)
    CZ = []
    for i in range(len(CX)):
        CZ.append(fe(CX[i],CY[i]))
    CD = cumulative_distance(CX,CY,CZ)
    DEV = np.nan
    return (CX,CY,CZ,CD,DEV)

def detect_centreline(data, fields, radius, step, debug_params = (2,1500,5,95,0), hM_std = 2.5, sight_angle = np.pi/4, resolution = 12, moment_num = 3, start_position = 'None', start_loc = [np.nan,np.nan],max_iteration = 1000):   
    X, Y, Z, mag = data # Grid data
    fe, fm = fields # interpolated grid of elevation and slope
    rad_array = np.linspace(-sight_angle,sight_angle,resolution) # radian 
    debug_radius_ratio, debug_std, hpl, hph, sigma = debug_params # expand rate of radius, threshold of minimum std, lower and upper height-weight percentile
    Z = gaussian_filter(Z,sigma=sigma)
    # Define start point
    if start_position == 'None':
        minz_array = np.ones(5)
        start_array = np.ones(5)
        pos_array = ['left','right','top','bottom','topright']
    #     pos_array = ['right','left','bottom','top']
        #left
        X_array = X[:,0]; Y_array = Y[:,0]; Z_array = Z[:,0]; S_array = mag[:,0]
        minz_array[0] = np.min(Z_array)
        start_array[0] = np.argmin(Z_array)
        #right
        X_array = X[:,-1]; Y_array = Y[:,-1]; Z_array = Z[:,-1]; S_array = mag[:,-1]
        minz_array[1] = np.min(Z_array)
        start_array[1] = np.argmin(Z_array)
        #top
        X_array = X[-1,:]; Y_array = Y[-1,:]; Z_array = Z[-1,:]; S_array = mag[-1,:]
        minz_array[2] = np.min(Z_array)
        start_array[2] = np.argmin(Z_array)
        #bottom
        X_array = X[0,:]; Y_array = Y[0,:]; Z_array = Z[0,:]; S_array = mag[0,:]
        minz_array[3] = np.min(Z_array)
        start_array[3] = np.argmin(Z_array)
        
        # estimate position    
        position = pos_array[np.argsort(minz_array)[1]]
        start = start_array[np.argsort(minz_array)[1]].astype(int)
    
    position = start_position
    if position == 'left':
        Z_array = Z[:,0];
        start = np.argmin(Z_array)
        clp_ind = [start,0] # define current location index
    elif position == 'right':
        Z_array = Z[:,-1];
        start = np.argmin(Z_array)
        clp_ind = [start,-1] # define current location index
        rad_array += np.pi
    elif position == 'righttop':
        Z_array = Z[int(len(Z[:,-1])/2):,-1];
        start = np.argmin(Z_array) + int(len(Z[:,-1])/2)
        clp_ind = [start,-1] # define current location index
        rad_array += np.pi
    elif position == 'bottom':
        Z_array = Z[0,:];
        start = np.argmin(Z_array)
        clp_ind = [0,start] # define current location index
        rad_array += np.pi/2
    elif position == 'bottomleft':
        Z_array = Z[0,:int(len(Z[0,:])/2)];
        start = np.argmin(Z_array) 
        clp_ind = [0,start] # define current location index
        rad_array += np.pi/2
    elif position == 'top':
        Z_array = Z[-1,:];
        start = np.argmin(Z_array)
        clp_ind = [-1,start] # define current location index
        rad_array += -np.pi/2
    elif position == 'topright':
        Z_array = Z[-1,int(len(Z[-1,:])/2):];
        start = np.argmin(Z_array) + int(len(Z[-1,:])/2)
        clp_ind = [-1,start] # define current location index
        rad_array += -np.pi/2
    elif position == 'toprightright':
        Z_array = Z[-1,int(3*len(Z[-1,:])/4):];
        start = np.argmin(Z_array) + int(3*len(Z[-1,:])/4)
        clp_ind = [-1,start] # define current location index
        rad_array += -np.pi/2
    
    if np.any(np.isnan(start_loc)):
        clp_x = X[clp_ind[0],clp_ind[1]];    clp_y = Y[clp_ind[0],clp_ind[1]];    clp_z = Z[clp_ind[0],clp_ind[1]]
    else:
        clp_x = start_loc[0]; clp_y = start_loc[1]; clp_z = fe(clp_x,clp_y)

    clp = [clp_x,clp_y]
    CX = [clp_x];    CY = [clp_y];    CZ = [clp_z];   CD = [0];
    DEV = [0]
    def pointChecker(clp,count,max_iteration):
        if clp[0] < np.min(X[0,:]) or clp[0] > np.max(X[0,:]) or clp[1] < np.min(Y[:,0]) or clp[1] > np.max(Y[:,0]) or count > max_iteration:
            return False
        else:
            count += 1
            return True
    count = 0
    direc_array = rad_array
    dev_array = np.zeros(len(rad_array)) # deviation of slope
    hM_array = np.zeros(len(rad_array)) # maximum elevation 
#     distance_array = [0]
    direc_temp = np.ones(moment_num)*np.mean(rad_array)
    direc = np.mean(direc_array)
    while(pointChecker(clp,count,max_iteration)):
        count += 1
        ind_array = np.arange(len(direc_array))
        for i,rad in enumerate(direc_array): 
#             # short range
            lx = np.cos(rad) * radius + clp_x; ly = np.sin(rad) * radius + clp_y
            lx = np.linspace(clp_x,lx,20); ly = np.linspace(clp_y,ly,20)
            if lx[-1] > np.max(X) or lx[-1] < np.min(X) or ly[-1] > np.max(Y) or ly[-1] < np.min(Y):
                lx = np.linspace(clp_x,lx[-1],20); ly = np.linspace(clp_y,ly[-1],20)
                lx = lx[np.where(lx <= np.max(X))]; ly = ly[np.where(lx <= np.max(X))]
                lx = lx[np.where(lx >= np.min(X))]; ly = ly[np.where(lx >= np.min(X))]
                lx = lx[np.where(ly <= np.max(Y))]; ly = ly[np.where(ly <= np.max(Y))]
                lx = lx[np.where(ly >= np.min(Y))]; ly = ly[np.where(ly >= np.min(Y))]
                lx = np.linspace(clp_x,lx[-1],20); ly = np.linspace(clp_y,ly[-1],20)
            lz = fe(lx,ly); lm = fm(lx,ly)
#             lz = np.array([fe(lx[k],ly[k]) for k in range(len(lx))]); lm = np.array([fm(lx[k],ly[k]) for k in range(len(lx))])
            H_eval = np.percentile(lz,hph)/np.percentile(lz,hpl)
            m_eval = np.sum(lm[np.where(lz >= clp_z)]*H_eval)
            m_dev = np.std(lm[np.where(lz >= clp_z)])
            hM_array[i] = np.max(lz) - clp_z
            dev_array[i] = m_eval
#             dev_array[i] = m_eval * (1 + m_dev)
        if np.std(dev_array) < debug_std:
            for i,rad in enumerate(direc_array): # -pi/2 < rad < pi/2
                lx = np.cos(rad) * radius*debug_radius_ratio + clp_x 
                ly = np.sin(rad) * radius*debug_radius_ratio + clp_y
                lx = np.linspace(clp_x,lx,10); ly = np.linspace(clp_y,ly,10)
                if lx[-1] > np.max(X) or lx[-1] < np.min(X) or ly[-1] > np.max(Y) or ly[-1] < np.min(Y):
                    lx = np.linspace(clp_x,lx[-1],20); ly = np.linspace(clp_y,ly[-1],20)
                    lx = lx[np.where(lx <= np.max(X))]; ly = ly[np.where(lx <= np.max(X))]
                    lx = lx[np.where(lx >= np.min(X))]; ly = ly[np.where(lx >= np.min(X))]
                    lx = lx[np.where(ly <= np.max(Y))]; ly = ly[np.where(ly <= np.max(Y))]
                    lx = lx[np.where(ly >= np.min(Y))]; ly = ly[np.where(ly >= np.min(Y))]
                    lx = np.linspace(clp_x,lx[-1],20); ly = np.linspace(clp_y,ly[-1],20)
                lz = np.array([fe(lx[k],ly[k]) for k in range(len(lx))]);
                lm = np.array([fm(lx[k],ly[k]) for k in range(len(lx))])
                H_eval = np.percentile(lz,hph)/np.percentile(lz,hpl)
                m_eval = np.sum(lm[np.where(lz >= clp_z)]*H_eval)
                m_dev = np.std(lm[np.where(lz >= clp_z)])
                hM_array[i] = np.max(lz) - clp_z
                dev_array[i] = m_eval        
        DEV.append( np.std(hM_array) )
        if np.std(hM_array) < hM_std:
            direc = direc_array[np.argmin(dev_array)]
            direc_temp[count%moment_num] = direc
        else:
            direc = direc_array[np.argmin(hM_array)]
            direc_temp[count%moment_num] = direc

        rad_array = np.linspace(-sight_angle,sight_angle,resolution) # radian 
        direc_array = rad_array + np.mean(direc_temp)
        lx = np.cos(direc) * radius + clp_x
        ly = np.sin(direc) * radius + clp_y
        lx = np.linspace(clp_x,lx,100)
        ly = np.linspace(clp_y,ly,100)
        dx = np.cos(direc) * step
        dy = np.sin(direc) * step
        dz = clp_z - fe(clp_x+dx,clp_y+dy)
        # assign
        clp_x += np.cos(direc) * step
        clp_y += np.sin(direc) * step
        clp_z = fe(clp_x,clp_y)
        CX.append(clp_x)
        CY.append(clp_y)
        CZ.append(clp_z)
        clp = [clp_x,clp_y]
    CD = cumulative_distance(CX,CY,CZ)
    return (CX,CY,CZ,CD,DEV)


