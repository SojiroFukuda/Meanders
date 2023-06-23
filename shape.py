import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import os
import re
from sklearn.decomposition import PCA
import random

# Meander object
class meander(object):
    def __init__(self):
        super(meander, self).__init__()
    def read_path(self,path):
        csv_format = '.csv'
        X = 'X [-]' # name of header 
        Y = 'Y [-]' # name of header
        S = 'S [-]' # name of header 
        C = 'C [-]' # name of header
        self.data_     = pd.read_csv(path)
        self.path_name = path
        ids            = path.split(os.sep)
        self.file_name = ids[-1]
        self.name = self.file_name[0:-len(csv_format)]
        self.x = np.array(self.data_[X])
        self.y = np.array(self.data_[Y])
        self.s = np.array(self.data_[S])
        self.c = np.array(self.data_[C])

    
def get_cartesian(lon,lat,lonc='None',latc='None'):
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
    if lonc == 'None':
        lon_c = np.median(np.array(lon))  # [degree]
    else:
        lon_c = lonc
    if latc == 'None':
        lat_c = np.median(np.array(lat))  # [degree]
    else: 
        lat_c = latc
    # conversion
    X = (np.array(lon)-lon_c) * 2.0*np.pi*R_earth*np.cos(lat_c*np.pi/180)/360 # [meter]
    Y = (np.array(lat)-lat_c) * delta_lat # [meter]
    return X, Y
    
    
    
# shift start point to the origin of coordinate
def normalise(x,y,width=1):
#     ''''
#     convert (x0,y0) -> (0,0)
#     Normalise the coordinate by the charateristic width of channel
#     Reflect coordinate so that all meanders go from left to right
#     ''''
    x_std =  (np.array(x) - x[0]) / width  
    y_std =  (np.array(y) - y[0]) / width
    return x_std, y_std

# Rotate the shape (theta [radian])
def rotate_coord(x,y,theta):
	x_rot = x*np.cos(theta) - y*np.sin(theta)
	y_rot = x*np.sin(theta) + y*np.cos(theta)
	return x_rot, y_rot

# Calc the radian of slope between (0,0) and (x,y)
def slope(x,y):
	tan = y/x
	rad = np.arctan(tan)
	return rad

# Calc the radian of slope between (0,0) and (x,y)
def mean_direction(x,y):
# 	tan = y/x
# 	rad = np.arctan(tan)
    rad = np.arctan2(np.sum(y),np.sum(x))
    return rad

def smoothen(x,y,window=31,poly=3):
    from scipy.signal import savgol_filter
    xhat = savgol_filter(x, window, poly)
    yhat = savgol_filter(y, window, poly)
    return xhat,yhat


def distance(x_array,y_array):
    coord = np.array([(xi,yi) for xi,yi in zip(x_array,y_array)])
    temp = np.zeros([len(coord)+1,2])
    temp[1:] = coord
    temp[0] = coord[0]
    dist_temp = coord - temp[:-1]
    dist = [np.linalg.norm(dis) for dis in dist_temp]
    array = np.cumsum(dist) # cumlative distance
    return array

def streamwiseDirection(x,y):
    x_temp = np.zeros(len(x)+1)
    y_temp = np.zeros(len(y)+1)
    x_temp[1:] = x
    y_temp[1:] = y
    x_temp[0] = x[0]
    y_temp[0] = y[0]
    x_dif = x - x_temp[:-1]
    y_dif = y - y_temp[:-1]
    t = np.arctan2(y_dif,x_dif)
    return t

def splitmeander(md,window=50,delt=0.001,T=1):
    mds = []
    temp_p = [0,0]
    start_indice = [0]
    end_indice = []
    st_ind = 0
    flag = 0
    def dist(p1,p2):
        return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
    for i,(x,y) in enumerate(zip(md.x,md.y)):
        if dist(temp_p,[x,y]) >= window:
            temp_p = [x,y]
            if flag == 0:
                end_indice.append(i)
                st_ind = i
                flag += 1
            else:
                start_indice.append(st_ind)
                end_indice.append(i)
                st_ind = i
    flag = 0
    if dist([md.x[end_indice[-1]],md.y[end_indice[-1]]],[md.x[-1],md.y[-1]] ) > window/2.0:
        for i in range(len(md.x)):
            if dist([md.x[-1],md.y[-1]],[md.x[-(i+1)],md.y[-(i+1)]]) >= window:
                if flag == 0:
                    start_indice.append(len(md.x)-(i+1))
                    end_indice.append(len(md.x)-1)
                    flag += 1
 
    for i, (start_ind, end_ind) in enumerate(zip(start_indice,end_indice)):
        md_temp = meander()
        x = md.x[start_ind:end_ind] - md.x[start_ind]
        y = md.y[start_ind:end_ind] - md.y[start_ind]
        x_rot,y_rot = rotate_coord(x,y,-slope(x[-1],y[-1]))
        xp,yp = coord_time(x_rot,y_rot,delt,T)
        md_temp.x = xp
        md_temp.y = yp
        md_temp.start = [md.x[start_ind],md.y[start_ind]]
        md_temp.id = md.id + '_' + str(i+1)
        md_temp.setting = md.setting
        md_temp.site     = md.site        
        mds.append(md_temp)
    return mds

def detrendmeander_quad(x:list,y:list):
    params = np.polyfit(x,y,2)
    fx = np.poly1d( params )
    return fx, params
    

def ellipseCoord(x,y,ratio=1,delt=0.001,T=1,equal_step=False):
    xpr = x*ratio
    radius = np.abs(xpr[0] - xpr[-1])/(2*np.pi)
    theta = xpr/radius
    rt = y + radius
    if equal_step:
        xp,yp = coord_time(rt*np.cos(theta),rt*np.sin(theta),delt,T)
    else:
        xp = rt*np.cos(theta)
        yp = rt*np.sin(theta)
    return xp,yp

# Export Elliptic Fourier Descriptors into CSV files
def exportEFDs(name,N,efds):
    if len(efds) != N:
        print('N does not match efds_array')
    header_list = ['Name','N']
    if len(efds[0]) == 4:
        for i in range(N):
            header_list.append('a_'+str(i+1))
            header_list.append('b_'+str(i+1))
            header_list.append('c_'+str(i+1))
            header_list.append('d_'+str(i+1))
    elif len(efds[0]) == 2:
        for i in range(N):
            header_list.append('c_'+str(i+1))
            header_list.append('d_'+str(i+1))
    df = pd.DataFrame(columns=header_list)
    data = [name,N]
    if len(efds[0]) == 4:
        for i in range(N):
            n   = i+1
            efd = []
            data.append(efds[i,0])
            data.append(efds[i,1])
            data.append(efds[i,2])
            data.append(efds[i,3])
    elif len(efds[0]) == 2:
        for i in range(N):
            n   = i+1
            efd = []
            data.append(efds[i,0])
            data.append(efds[i,1])
    df.loc[name] = data
    return df

# Import EFDs and meta data from CSV file
def importEFDs(path):
    df       = pd.read_csv(path)
    name     = df['Name'].values.astype(str)[0]
    N        = df['N'].values.astype(int)[0]
    efd_list = np.zeros((N,4))
    for i in range(int(N)):
        efd_list[i] = [ df[ 'a_'+str(i+1)].values.astype(float)[0], 
                       df[  'b_'+str(i+1)].values.astype(float)[0], 
                       df[  'c_'+str(i+1)].values.astype(float)[0], 
                       df[  'd_'+str(i+1)].values.astype(float)[0] ]
    return name, N, efd_list

def importEFDs_x(path):
    df       = pd.read_csv(path)
    name     = df['Name'].values.astype(str)[0]
    N        = df['N'].values.astype(int)[0]
    efd_list = np.zeros((N,2))
    for i in range(int(N)):
        efd_list[i] = [  df[  'c_'+str(i+1)].values.astype(float)[0], 
                       df[  'd_'+str(i+1)].values.astype(float)[0] 
                      ]
    return name, N, efd_list



def coord_time(x,y,delt,T):
    # t = np.arange(0,T,delt)
    t = np.linspace(0,T,int(T/delt)+1)
    dt = np.zeros( len( x ) - 1 ) # length along the countour between each coordinate.
    cum = np.zeros( len( x ) ) # cumlative length of dt.
    cum[0] = 0
    for i in range(len( x ) - 1):
        dt[i] = np.sqrt( (x[i+1]-x[i])*(x[i+1]-x[i]) + (y[i+1]-y[i])*(y[i+1]-y[i]) )
        cum[i+1] = cum[i] + dt[i]
    #create func 
    inter_func_X = interp1d( (cum/cum[-1])*T,x,kind='cubic')
    inter_func_Y = interp1d( (cum/cum[-1])*T,y,kind='cubic')
    # align the coordinates evenly along the contour
    x_p = inter_func_X(t)
    y_p = inter_func_Y(t)
    return x_p, y_p

def get_Coords(x,y,delt=0.001,T=1.0):
    t = np.linspace(0,T,int(T/delt)+1)
    dt = np.zeros( len( x ) - 1 ) # length along the countour between each coordinate.
    cum = np.zeros( len( x ) ) # cumlative length of dt.
    cum[0] = 0
    for i in range(len( x ) - 1):
        dt[i] = np.sqrt( (x[i+1]-x[i])*(x[i+1]-x[i]) + (y[i+1]-y[i])*(y[i+1]-y[i]) )
        cum[i+1] = cum[i] + dt[i]
    #create func 
    inter_func_X = interp1d( (cum/cum[-1])*T,x,kind='cubic')
    inter_func_Y = interp1d( (cum/cum[-1])*T,y,kind='cubic')
    # align the coordinates evenly along the contour
    xp = inter_func_X(t)
    yp = inter_func_Y(t)
    x_temp = np.zeros(len(xp)+1); y_temp = np.zeros(len(yp)+1)
    x_temp[1:] = xp; y_temp[1:] = yp
    x_temp[0] = xp[0]; y_temp[0] = yp[0]
    x_dif = xp - x_temp[:-1]; y_dif = yp - y_temp[:-1]
    theta = np.arctan2(y_dif,x_dif)
    s = t * cum[-1]
    return t, xp, yp, s, theta

def align_x(x,y,delt,T):
    x_p = np.arange(0,T,delt)
    #create func 
    inter_func_X = interp1d( x, y, kind='cubic' )
    # align the coordinates evenly along the contour
    y_p = inter_func_X(x_p)
    return x_p, y_p

# sort file from Folda which include 'key'
def file_list(FOLDA_DIR,key):
    file_list = []
    files = os.listdir(FOLDA_DIR)
    for file in files:
        index = re.search(key,file)
        if index:
            file_list.append(file)
    return file_list

# transfrom a curve into a closed contour
# method 1: fold x coordinate 
def periodic_fold(x_ori):
    t = len(x_ori)
    width = max(x_ori)
    ind = np.where(x_ori>(width/2))
    x_return = width - x_ori[ind[0][0]:-1]
    x_periodic = np.concatenate((x_ori[0:ind[0][0]+1],x_return))
    return x_periodic

def periodic_mirror(x,y):
    x_mirror = np.flip(np.array(x))
    y_mirror = np.flip(np.array(y)*-1.0)
    x_per    = np.concatenate( ( np.array(x), np.array(x_mirror[1:-1]) )  )
    y_per    = np.concatenate( ( np.array(y), np.array(y_mirror[1:-1]) )  )
    return x_per, y_per 

def periodic_meander(x,y,delt,T):
    # t = np.arange(0,T,delt)
    t = np.linspace(0,T,int(T/delt)+1)
    x_t, y_t = coord_time(x,y,delt,T)
    x_per = x_t-t*np.max(x_t)
    y_per = y_t
#     x_per, y_per = coord_time(x_per, y_per, delt,T)
    return x_per, y_per, t

    
def periodic_abs(x,y):
    x_mirror = np.flip(np.array(x))
    y_mirror = np.flip(np.abs(y)*-1.0)
    x_per    = np.concatenate( ( np.array(x), np.array(x_mirror[1:-1]) )  )
    y_per    = np.concatenate( ( np.abs(y), np.array(y_mirror[1:-1]) )  )
    return x_per, y_per 


def adjustXYCoord(x,y):
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    x_center = x_min+np.abs(x_max-x_min)/2
    y_center = y_min+np.abs(y_max-y_min)/2
    x_a = (x-x_center)
    y_a = -(y-y_center)
    return x_a,y_a



def fourierApproximation(x_ori,y_ori,N,delt=0.001,T=1.0):
    efd_list = [] # original elliptic Fourier descriptors (EFDs)
    # t = np.arange(0,T,delt)
    t = np.linspace(0,T,int(T/delt)+1)
    x_r = np.zeros(len(t)) #X-coordinate of reconstructed shape from EFDs
    y_r = np.zeros(len(t)) #Y-coordinate of reconstructed shape from EFDs

    # x,y = adjustXYCoord(x_ori,y_ori)
    x = x_ori
    y = y_ori
    # x,y = getXYCoord(cnt)
    dt = np.zeros( len( x ) - 1 ) # length along the countour between each coordinate.
    cum = np.zeros( len( x ) ) # cumlative length of dt.
    cum[0] = 0
    for i in range(len( x ) - 1):
        dt[i] = np.sqrt( (x[i+1]-x[i])*(x[i+1]-x[i]) + (y[i+1]-y[i])*(y[i+1]-y[i]) )
        cum[i+1] = cum[i] + dt[i]
    #create func 
    inter_func_X = interp1d(cum/cum[-1],x,kind='linear')
    inter_func_Y = interp1d(cum/cum[-1],y,kind='linear')
    # align the coordinates evenly along the contour
    x_p = inter_func_X(t)
    y_p = inter_func_Y(t)
    N_list = []
    for i in range(N):
        #calculate EFDs
        an, bn, cn, dn = efd(T,x_p,y_p,t,delt,i+1)
        efd_list.append(np.array([an,bn,cn,dn]))
        #Reconstruction
        x_r += an*np.cos(2*(i+1)*np.pi*t/T) + bn*np.sin(2*(i+1)*np.pi*t/T)
        y_r += cn*np.cos(2*(i+1)*np.pi*t/T) + dn*np.sin(2*(i+1)*np.pi*t/T)
        N_list.append([np.copy(x_r),np.copy(y_r)])
    return N_list,efd_list,x_p,y_p,t

def reconstContourCoord_efd(efd_list,N,t,T):
    x_t = np.zeros(len(t)) #X-coordinate
    y_t = np.zeros(len(t)) #Y-coordinate
    for n in range(1,N+1):
        efd = efd_list[n-1]
        an = efd[0]
        bn = efd[1]
        cn = efd[2]
        dn = efd[3]
        x_t += an*np.cos(2*n*np.pi*t/T) + bn*np.sin(2*n*np.pi*t/T)
        y_t += cn*np.cos(2*n*np.pi*t/T) + dn*np.sin(2*n*np.pi*t/T)
    np.append(x_t,x_t[0])
    np.append(y_t,y_t[0])
    return x_t, y_t

def reconstContourCoord_efdx(efdx_list,N,t,T):
    y_t = np.zeros(len(t)) #Y-coordinate
    for n in range(1,N+1):
        efd = efdx_list[n-1]
        cn = efd[0]
        dn = efd[1]
        y_t += cn*np.cos(2*n*np.pi*t/T) + dn*np.sin(2*n*np.pi*t/T)
    np.append(y_t,y_t[0])
    return y_t

def efd(T,x_p,y_p,t_p,dt,n):
    an = 0
    bn = 0
    cn = 0
    dn = 0
    for i in range( 1, len(x_p) ):
        del_xp = x_p[i]-x_p[i-1]
        del_yp = y_p[i]-y_p[i-1]
        del_t_test = np.sqrt(del_xp*del_xp+del_yp*del_yp)
        pi = np.pi
        an +=  ( del_xp / dt ) * ( np.cos(2*n*pi*t_p[i]/T) - np.cos(2*n*pi*t_p[i-1]/T) )
        bn +=  ( del_xp / dt ) * ( np.sin(2*n*pi*t_p[i]/T) - np.sin(2*n*pi*t_p[i-1]/T) )
        cn +=  ( del_yp / dt ) * ( np.cos(2*n*pi*t_p[i]/T) - np.cos(2*n*pi*t_p[i-1]/T) )
        dn +=  ( del_yp / dt ) * ( np.sin(2*n*pi*t_p[i]/T) - np.sin(2*n*pi*t_p[i-1]/T) )
    an = an* (T/(2*n*n*pi*pi))
    bn = bn* (T/(2*n*n*pi*pi))
    cn = cn* (T/(2*n*n*pi*pi))
    dn = dn* (T/(2*n*n*pi*pi))
    return an,bn,cn,dn

def efd_x(T,y_p,t_p,dt,n):
    cn = 0
    dn = 0
    for i in range( 1, len(t_p) ):
        del_yp = y_p[i]-y_p[i-1]
        pi = np.pi
        cn +=  ( del_yp / dt ) * ( np.cos(2*n*pi*t_p[i]/T) - np.cos(2*n*pi*t_p[i-1]/T) )
        dn +=  ( del_yp / dt ) * ( np.sin(2*n*pi*t_p[i]/T) - np.sin(2*n*pi*t_p[i-1]/T) )
    cn = cn* (T/(2*n*n*pi*pi))
    dn = dn* (T/(2*n*n*pi*pi))
    return cn,dn

def efds(T,x_p,y_p,t_p,dt,N):
    efd_list = np.zeros((N,4))
    for i in range(N):
        an,bn,cn,dn = efd(T,x_p,y_p,t_p,dt,i+1)
        efd_list[i] = [an,bn,cn,dn]
    return efd_list

def efds_x(T,y_p,t_p,dt,N):
    efd_list = np.zeros((N,2))
    for i in range(N):
        cn,dn = efd_x(T,y_p,t_p,dt,i+1)
        efd_list[i] = [cn,dn]
    return efd_list



def FPS_fromEFDs(EFD_list):
    FPS_list = np.sum(np.array(EFD_list)**2,axis=1)
    return FPS_list/2

def efd_norm(efd_list,N,t,T):
    efd_star_list = []
    x_r, y_r = reconstContourCoord_efd(efd_list,N,t,T)
    a1, b1, c1, d1 = efd_list[0]
    x1 = x_r[0]
    y1 = y_r[0]
    atan = np.arctan2( (2 * ( a1*b1 + c1*d1 )) , ( a1*a1 + c1*c1 - b1*b1 - d1*d1 ) )
    if atan < 0:
        atan += 2*np.pi
    theta = 0.5 * atan

    a1_star = a1 * np.cos(theta) + b1 * np.sin(theta)
    c1_star = c1 * np.cos(theta) + d1 * np.sin(theta)
    b1_star = -1 * a1 * np.sin(theta) + b1 * np.cos(theta)
    d1_star = -1 * c1 * np.sin(theta) + d1 * np.cos(theta)

    psi_1 = np.arctan2( c1_star , a1_star )
    if psi_1 < 0:
        psi_1 += 2*np.pi

    E = np.sqrt( a1_star*a1_star + c1_star*c1_star )
    psi_mat = np.array([[np.cos(psi_1),np.sin(psi_1)],[-1*np.sin(psi_1),np.cos(psi_1)]])
    x_star = np.zeros(len(t)) #X-coordinate of reconstructed shape from normalized EFDs
    y_star = np.zeros(len(t)) #Y-coordinate of reconstructed shape from normalized EFDs
    harmonics = []
    for j in range(N):
        aj = efd_list[j][0]
        bj = efd_list[j][1]
        cj = efd_list[j][2]
        dj = efd_list[j][3]
        efd_n = np.array([[aj,bj],[cj,dj]])
        theta_mat = np.array([[np.cos((j+1)*theta),-1*np.sin((j+1)*theta)],[np.sin((j+1)*theta),np.cos((j+1)*theta)]])
        efd_star = np.dot( np.dot(psi_mat,efd_n), theta_mat)
        efd_star_array = np.array([efd_star[0][0],efd_star[0][1],efd_star[1][0],efd_star[1][1]])
        efd_star_array = efd_star_array / E
        efd_star_list.append(efd_star_array) # acquire normalized EFDs
    return efd_star_list


def FPS_analysis(efd_list,x_r,y_r,N,name,group):
    print(name)
    T = 1
    # t = np.arange(0,T,0.001)
    t = np.linspace(0,T,int(T/0.001)+1)
    FPS_HEADER = []
    df_pre = pd.DataFrame({ 'FileName_' :[name]})

#     if re.search('cmRun',name):
#         df_pre['group'] = ['cmRun']
#     elif re.search('mm',name):
#         df_pre['group'] = ['mm']
#     elif re.search('peterman',name):
#         df_pre['group'] = ['peterman']
    df_pre['group'] = group

    for i in range(N):
        FPS_HEADER.append("FPS"+str(i+1))
    for i in range(N):
        FPS_HEADER.append("a"+str(i+1))
        FPS_HEADER.append("b"+str(i+1))
        FPS_HEADER.append("c"+str(i+1))
        FPS_HEADER.append("d"+str(i+1))

    FPS_matrix = np.zeros(5*N)
    efd_star_list = []
    ## Normalization (size, axial rotation, starting point)
    a1, b1, c1, d1 = efd_list[0]
    x1 = x_r[0]
    y1 = y_r[0]
    atan = np.arctan2( (2 * ( a1*b1 + c1*d1 )) , ( a1*a1 + c1*c1 - b1*b1 - d1*d1 ) )
    if atan < 0:
        atan += 2*np.pi
    theta = 0.5 * atan

    a1_star = a1 * np.cos(theta) + b1 * np.sin(theta)
    c1_star = c1 * np.cos(theta) + d1 * np.sin(theta)
    b1_star = -1 * a1 * np.sin(theta) + b1 * np.cos(theta)
    d1_star = -1 * c1 * np.sin(theta) + d1 * np.cos(theta)

    psi_1 = np.arctan2( c1_star , a1_star )
    if psi_1 < 0:
        psi_1 += 2*np.pi

    E = np.sqrt( a1_star*a1_star + c1_star*c1_star )
    psi_mat = np.array([[np.cos(psi_1),np.sin(psi_1)],[-1*np.sin(psi_1),np.cos(psi_1)]])
    x_star = np.zeros(len(t)) #X-coordinate of reconstructed shape from normalized EFDs
    y_star = np.zeros(len(t)) #Y-coordinate of reconstructed shape from normalized EFDs
    for i in range(N):
        aj = efd_list[i][0]
        bj = efd_list[i][1]
        cj = efd_list[i][2]
        dj = efd_list[i][3]
        efd_n = np.array([[aj,bj],[cj,dj]])
        theta_mat = np.array([[np.cos((i+1)*theta),-1*np.sin((i+1)*theta)],[np.sin((i+1)*theta),np.cos((i+1)*theta)]])
        efd_star = np.dot( np.dot(psi_mat,efd_n), theta_mat)
        efd_star_array = np.array([efd_star[0][0],efd_star[0][1],efd_star[1][0],efd_star[1][1]])
        efd_star_array = efd_star_array / E
        efd_star_list.append(efd_star_array) # acquire normalized EFDs
        fps_value = (efd_star_array[0]*efd_star_array[0]+efd_star_array[1]*efd_star_array[1]+efd_star_array[2]*efd_star_array[2]+efd_star_array[3]*efd_star_array[3])/2

        FPS_matrix[i] = fps_value
        # traditional EFDs
        FPS_matrix[4*i+N+0] = efd_star_array[0]
        FPS_matrix[4*i+N+1] = efd_star_array[1]
        FPS_matrix[4*i+N+2] = efd_star_array[2]
        FPS_matrix[4*i+N+3] = efd_star_array[3]    
    df_fps = pd.DataFrame([FPS_matrix],columns=FPS_HEADER)
    df_concat = pd.concat([df_pre,df_fps],axis=1)
    return df_concat

def conductPCA_correlation(df,N,isCor):
    # df = pd.read_csv(csv_path)
#     df = csv_path
    # Normalization: case for Correlation matrix
    FPS_loc = df.columns.get_loc('FPS1')
    # take log FPS2–N 
#     df.iloc[:,( FPS_loc + 1 ): (FPS_loc + N )] = -1*np.log(df.iloc[:,( FPS_loc + 1 ):(FPS_loc + N )])
    
    scale_array = np.std(df.iloc[:,(FPS_loc):(FPS_loc + N )])
    center_array= np.mean(df.iloc[:,(FPS_loc):(FPS_loc + N )])
    
    # Correlation Matrix
    # print(isFPS)
    
    if isCor:
        dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    else:
        dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean()), axis=0)
    
    #PCA
    pca = PCA()
    feature = pca.fit(dfs)
    feature = pca.transform(dfs)
    dfs.to_csv("test.csv")

    PC_SCORE = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    result_df = pd.concat([df,PC_SCORE],axis=1)

    #Contribute Rate
    contribution = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    #EigenValue
    eigenvalues = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    #EigenVector Rotation
    csv_rot = pd.DataFrame(pca.components_, columns=df.columns[FPS_loc:(FPS_loc + N )], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    # Standard Dev
    stdv_array = np.std(PC_SCORE)
    #Rotation Mat
    csv_rot = csv_rot.T
    rot_data = np.array(csv_rot.values.flatten())
    rot_array = np.reshape(rot_data,(csv_rot.shape[0],csv_rot.shape[1]))
    rot_mat = np.matrix(rot_array).astype(float)
    inv_rot = np.linalg.inv(rot_mat) #inverse matrix of the rotation.
    #Scale Mat
    scale_data = np.array(scale_array.values.flatten())
    #Center Mat
    center_data = np.array(center_array.values.flatten()) 
    scale_mat = np.diag(np.reshape(scale_data.astype(float),N))
    center_mat = np.reshape(center_data.astype(float),N)    
    #Stdv array
    stdv_array = np.array(stdv_array.values.flatten())
    cont = contribution
    cont['Cum.'] = np.cumsum(pca.explained_variance_ratio_)
    cont['Dev.'] = stdv_array
    cont.columns = ["Cont.","Cum.","Dev."]

    return result_df,cont,eigenvalues,csv_rot,scale_mat,center_mat,stdv_array,inv_rot,N


def conductPCA_correlation_xy(df,N,start_loc,isCor):
    # df = pd.read_csv(csv_path)
#     df = csv_path
    # Normalization: case for Correlation matrix
#     FPS_loc = df.columns.get_loc('FPS1')
    FPS_loc = start_loc
    # take log FPS2–N 
#     df.iloc[:,( FPS_loc + 1 ): (FPS_loc + N )] = -1*np.log(df.iloc[:,( FPS_loc + 1 ):(FPS_loc + N )])
    
    scale_array = np.std(df.iloc[:,(FPS_loc):(FPS_loc + N )])
    center_array= np.mean(df.iloc[:,(FPS_loc):(FPS_loc + N )])
    
    # Correlation Matrix
    # print(isFPS)
    
    if isCor:
        dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    else:
        dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean()), axis=0)
    
    #PCA
    pca = PCA()
    feature = pca.fit(dfs)
    feature = pca.transform(dfs)
    dfs.to_csv("test.csv")

    PC_SCORE = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    result_df = pd.concat([df,PC_SCORE],axis=1)

    #Contribute Rate
    contribution = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    #EigenValue
    eigenvalues = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    #EigenVector Rotation
    csv_rot = pd.DataFrame(pca.components_, columns=df.columns[FPS_loc:(FPS_loc + N )], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    # Standard Dev
    stdv_array = np.std(PC_SCORE)
    #Rotation Mat
    csv_rot = csv_rot.T
    rot_data = np.array(csv_rot.values.flatten())
    rot_array = np.reshape(rot_data,(csv_rot.shape[0],csv_rot.shape[1]))
    rot_mat = np.matrix(rot_array).astype(float)
    inv_rot = np.linalg.inv(rot_mat) #inverse matrix of the rotation.
    #Scale Mat
    scale_data = np.array(scale_array.values.flatten())
    #Center Mat
    center_data = np.array(center_array.values.flatten()) 
    scale_mat = np.diag(np.reshape(scale_data.astype(float),N))
    center_mat = np.reshape(center_data.astype(float),N)    
    #Stdv array
    stdv_array = np.array(stdv_array.values.flatten())
    cont = contribution
    cont['Cum.'] = np.cumsum(pca.explained_variance_ratio_)
    cont['Dev.'] = stdv_array
    cont.columns = ["Cont.","Cum.","Dev."]

    return result_df,cont,eigenvalues,csv_rot,scale_mat,center_mat,stdv_array,inv_rot,N

# def conductPCA_correlation(csv_path,isFPS,isCor):
#     # df = pd.read_csv(csv_path)
#     df = csv_path
#     # Normalization: case for Correlation matrix
#     FPS_loc = df.columns.get_loc('FPS1')
#     A1_loc = df.columns.get_loc('a1')
#     N = int((len(df.columns) - FPS_loc)/5.0)
#     # take log FPS2–N 
#     df.iloc[:,( FPS_loc + 1 ): (FPS_loc + N )] = -1*np.log(df.iloc[:,( FPS_loc + 1 ):(FPS_loc + N )])
#     # df.iloc[:,( FPS_loc + 5*N + 1 ):] = -1*np.log(df.iloc[:,( FPS_loc + 5*N + 1 ):])
#     # df.iloc[:,( FPS_loc + 1 ): (FPS_loc + N )] = np.log(df.iloc[:,( FPS_loc + 1 ):(FPS_loc + N )])  #### koko
#     # df.iloc[:,( FPS_loc + N + 4 ):] = -1*np.log(1+df.iloc[:,( FPS_loc + N + 4 ):])
#     if isFPS == 0: # FPS
#         scale_array = np.std(df.iloc[:,(FPS_loc):(FPS_loc + N )])
#         center_array= np.mean(df.iloc[:,(FPS_loc):(FPS_loc + N )])
#     elif isFPS == 2: # EFD
#         scale_array = np.std(df.iloc[:,(FPS_loc + N):(FPS_loc + 5*N)])
#         center_array= np.mean(df.iloc[:,(FPS_loc + N):(FPS_loc + 5*N)])
#     # else: # AMp
#     #     scale_array = np.std(df.iloc[:,(FPS_loc + 5*N):])
#     #     center_array= np.mean(df.iloc[:,(FPS_loc + 5*N):])
#     # Correlation Matrix
#     # print(isFPS)
#     if isFPS == 0:
#         if isCor:
#             dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean())/x.std(), axis=0)
#         else:
#             dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean()), axis=0)
#     elif isFPS == 2: #EFD
#         if isCor:
#             dfs = df.iloc[:,FPS_loc + N:FPS_loc + 5*N].apply(lambda x: (x-x.mean())/x.std(), axis=0)
#         else:
#             dfs = df.iloc[:,FPS_loc + N:FPS_loc + 5*N].apply(lambda x: (x-x.mean()), axis=0)
#     # else:
#     #     if isCor:
#     #         dfs = df.iloc[:,FPS_loc + 5*N:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
#     #     else:
#     #         dfs = df.iloc[:,FPS_loc + 5*N:].apply(lambda x: (x-x.mean()), axis=0)
#     #PCA
#     pca = PCA()
#     feature = pca.fit(dfs)
#     feature = pca.transform(dfs)
#     dfs.to_csv("test.csv")

#     PC_SCORE = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
#     result_df = pd.concat([df,PC_SCORE],axis=1)

#     #Contribute Rate
#     contribution = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
#     #EigenValue
#     eigenvalues = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
#     #EigenVector Rotation
#     if isFPS == 0: # FPS
#         csv_rot = pd.DataFrame(pca.components_, columns=df.columns[FPS_loc:(FPS_loc + N )], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
#     elif isFPS == 2: # EFD
#         csv_rot = pd.DataFrame(pca.components_, columns=df.columns[(FPS_loc + N):(FPS_loc + 5*N)], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
#     # else:
#     #     csv_rot = pd.DataFrame(pca.components_, columns=df.columns[(FPS_loc + 5*N):], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    
#     # Standard Dev
#     stdv_array = np.std(PC_SCORE)
#     #Rotation Mat
#     csv_rot = csv_rot.T
#     rot_data = np.array(csv_rot.values.flatten())
#     rot_array = np.reshape(rot_data,(csv_rot.shape[0],csv_rot.shape[1]))
#     rot_mat = np.matrix(rot_array).astype(float)
#     inv_rot = np.linalg.inv(rot_mat) #inverse matrix of the rotation.
#     #Scale Mat
#     scale_data = np.array(scale_array.values.flatten())
#     #Center Mat
#     center_data = np.array(center_array.values.flatten())

#     if isFPS == 0: # FPS    
#         scale_mat = np.diag(np.reshape(scale_data.astype(float),N))
#         center_mat = np.reshape(center_data.astype(float),N)
#     elif isFPS == 2: # EFD
#         scale_mat = np.diag(np.reshape(scale_data.astype(float),4*N))
#         center_mat = np.reshape(center_data.astype(float),4*N)
#     # else:
#     #     scale_mat = np.diag(np.reshape(scale_data.astype(float),2*N))
#     #     center_mat = np.reshape(center_data.astype(float),2*N)
#     #Stdv array
#     stdv_array = np.array(stdv_array.values.flatten())
#     cont = contribution
#     cont['Cum.'] = np.cumsum(pca.explained_variance_ratio_)
#     cont['Dev.'] = stdv_array
#     cont.columns = ["Cont.","Cum.","Dev."]

#     return result_df,cont,eigenvalues,csv_rot,scale_mat,center_mat,stdv_array,inv_rot,N




def fps(pcscore,inv_rot,scale_mat,center_mat):
    fps_log = np.zeros(len(pcscore))
    fps = np.zeros(len(pcscore))
    fps_log = (pcscore.dot(inv_rot).dot(scale_mat)+center_mat)
    # fps_log = (pcscore.dot(inv_rot)+center_mat)
#     for i in range(0,len(pcscore)):
#         if i == 0:
#             fps[i] = fps_log[0,i]
#         else:
#             fps[i]=np.exp(-1*fps_log[0,i])
    for i in range(0,len(pcscore)):
        fps[i] = np.exp(-1*fps_log[0,i])  
    return fps

# def fps(isFPS,pcscore,inv_rot,scale_mat,center_mat):
#     fps_log = np.zeros(len(pcscore))
#     fps = np.zeros(len(pcscore))
#     fps_log = (pcscore.dot(inv_rot).dot(scale_mat)+center_mat)
#     # fps_log = (pcscore.dot(inv_rot)+center_mat)
#     if isFPS == 0: # FPS mode
#         for i in range(0,len(pcscore)):
#             if i == 0:
#                 fps[i] = fps_log[0,i]
#             else:
#                 fps[i]=np.exp(-1*fps_log[0,i])
#                 # fps[i]=np.exp(fps_log[0,i])   ####koko
#     elif isFPS == 2: # EFD mode
#         for i in range(0,len(pcscore)):
#             if i == 0:
#                 fps[i] = fps_log[0,i]
#             else:
#                 fps[i] = fps_log[0,i]
#     else:
#          for i in range(0,len(pcscore)):
#             if i == 0:
#                 fps[i] = fps_log[0,i]
#             else:
#                 fps[i]=np.exp(-1*fps_log[0,i])
#     return fps

def efdgenerator(fps):
    abcd=2.0*fps
    # determine the ratio between a^2 + b^2 and c^2 + d^2
    ab = random.random() * abcd
    cd = abcd - ab
    aa = random.random() * ab
    bb = ab - aa
    cc = random.random() * cd
    dd = cd - cc
    a = np.sqrt(aa)*(-1)**random.randint(1,2)
    b = np.sqrt(bb)*(-1)**random.randint(1,2)
    c = np.sqrt(cc)*(-1)**random.randint(1,2)
    d = np.sqrt(dd)*(-1)**random.randint(1,2)
    return (a,b,c,d)

def efdgenerator_x(fps):
    cd=2.0*fps
    # determine the ratio between a^2 + b^2 and c^2 + d^2
    cc = random.random() * cd
    dd = cd - cc
    c = np.sqrt(cc)*(-1)**random.randint(1,2)
    d = np.sqrt(dd)*(-1)**random.randint(1,2)
    return (c,d)

def efdgenerator_amp(xam,yam):
    ab=2.0*xam
    cd=2.0*yam
    # determine the ratio between a^2 + b^2 and c^2 + d^2
    aa = random.random() * ab
    bb = ab - aa
    cc = random.random() * cd
    dd = cd - cc
    a = np.sqrt(aa)*(-1)**random.randint(1,2)
    b = np.sqrt(bb)*(-1)**random.randint(1,2)
    c = np.sqrt(cc)*(-1)**random.randint(1,2)
    d = np.sqrt(dd)*(-1)**random.randint(1,2)
    return (a,b,c,d)


def reconstContourCoord(N,fps):
    T = 1.0
    # t = np.arange(0,T,0.001)
    t = np.linspace(0,T,int(T/0.001)+1)
    x_t = np.zeros(len(t)) #X-coordinate
    y_t = np.zeros(len(t)) #Y-coordinate
    efd_list=[]
    for n in range(1,N+1):
        efd = efdgenerator(fps[n-1])
        an = efd[0]
        bn = efd[1]
        cn = efd[2]
        dn = efd[3]
        efd_list.append(efd)
        x_t += an*np.cos(2*n*np.pi*t/T) + bn*np.sin(2*n*np.pi*t/T)
        y_t += cn*np.cos(2*n*np.pi*t/T) + dn*np.sin(2*n*np.pi*t/T)
    np.append(x_t,x_t[0])
    np.append(y_t,y_t[0])
    return x_t, y_t, efd_list


def reconstContourCoord_x(N,window,fps):
    T = window
    s = np.arange(0,T,0.1)
    c = np.zeros(len(s)) #Y-coordinate
    efd_list=[]
    
    for n in range(1,N+1):
        efd = efdgenerator_x(fps[n-1])
        cn = efd[0]
        dn = efd[1]
        efd_list.append(efd)
        c += cn*np.cos(2*n*np.pi*s/T) + dn*np.sin(2*n*np.pi*s/T)
    return s, c, efd_list


def convertContour2Meander(x,y):
    x_min = min(x)
    min_ind = np.argmin(x)
    x_m = x - min(x)
    half = int(len(x)/2)
    x_m = np.append( np.flip(x_m[0:min_ind]),x_m[min_ind:]+x_m[0])
    y_m = y
    return x_m,y_m

def reconstMeanderfromFPS(fps,R,N,t,T):
    efd1 = np.sqrt(fps[0])
    x_r,y_r, efd_list = reconstContourCoord(N,fps) # closed contour
    efd_list[0] = [efd1,0,0,efd1] # Force the first harmonics to be a circle with Radius = R
    x_r,y_r = reconstContourCoord_efd(efd_list,N,t,T)
    y = np.sqrt(x_r**2+y_r**2) - R
    x = R * np.arctan2(y_r/(R+y),x_r/(R+y))
    for i,xt in enumerate(x):
        if i != 0:
            xtp = x[i-1]
            if np.abs(xtp-xt) > np.pi*R:
                x[i] = x[i]+50*(xtp-xt)/np.abs(xt-xtp)
    return x - x[0], y - y[0]
# def reconstMeanderfromFPS(fps,R,N,t,T):
#     # efd1 = np.sqrt(fps[0])
#     x_r,y_r, efd_list = reconstContourCoord(N,fps) # closed contour
#     # efd_list[0] = [efd1,0,0,efd1] # Force the first harmonics to be a circle with Radius = R
#     # x_r,y_r = reconstContourCoord_efd(efd_list,N,t,T)
#     # y = np.sqrt(x_r**2+y_r**2) - R
#     # x = R * np.arctan2(y_r/(R+y),x_r/(R+y))
#     # for i,xt in enumerate(x):
#         # if i != 0:
#             # xtp = x[i-1]
#             # if np.abs(xtp-xt) > np.pi*R:
#                 # x[i] = x[i]+50*(xtp-xt)/np.abs(xt-xtp)
    # return x - x[0], y - y[0]


def dif_array(va,vb,tei,num):
        diff = np.absolute(va-vb) + 1
        diff_array = np.power(tei, np.linspace(0,np.log(diff)/np.log(tei),num)) - 1
        if va < vb:
            return diff_array + va
        else:
            return diff_array*(-1) + va


def generateSequenceImages(x_a,y_a,x_b,y_b,SAVE_PATH,FILE_NAME):
    T = 1.0
    dt = 0.001
    N = 500
    t = np.arange(0,T,dt)

    aa = np.zeros(N)
    ba = np.zeros(N)
    ca = np.zeros(N)
    da = np.zeros(N)
    efd_list_a = []

    ab = np.zeros(N)
    bb = np.zeros(N)
    cb = np.zeros(N)
    db = np.zeros(N)
    efd_list_b = []

    for i in range(N):
        aa[i],ba[i],ca[i],da[i] = efd(T,x_a,y_a,t,dt,i+1)
        efd_list_a.append( np.array( [aa[i],ba[i],ca[i],da[i]] ) )

        ab[i],bb[i],cb[i],db[i] = efd(T,x_b,y_b,t,dt,i+1)
        efd_list_b.append( np.array( [ab[i],bb[i],cb[i],db[i]] ) )

    efd_list_a_n = efd_norm(efd_list_a, N, t, T)
    efd_list_b_n = efd_norm(efd_list_b, N, t, T)

    frame = 50
    efd_list_dt = []
    for i in range(N):
        efd_a = efd_list_a_n[i]
        efd_b = efd_list_b_n[i]
        efd_dif = ((np.log(efd_b)-np.log(efd_a))/np.log(2+i))/frame
        efd_dif = np.exp(efd_dif)
        efd_list_dt.append(efd_dif)

    os.makedirs(SAVE_PATH+os.sep+FILE_NAME,exist_ok=True)
    for i in range(frame):
        efd_list_frame = []
        for j in range(N):
            efd_a = efd_list_a_n[j]
            efd_b = efd_list_b_n[j]
            efd_dt = ( efd_b - efd_a )/frame
            a_dif = dif_array(efd_a[0],efd_b[0],2+j,frame)[i]
            b_dif = dif_array(efd_a[1],efd_b[1],2+j,frame)[i]
            c_dif = dif_array(efd_a[2],efd_b[2],2+j,frame)[i]
            d_dif = dif_array(efd_a[3],efd_b[3],2+j,frame)[i]

            efd_list_frame.append(np.array([a_dif, b_dif, c_dif, d_dif]) )
            # efd_dt =( ( np.log(efd_b)-np.log(efd_a) ) ) /frame
            # efd_frame = ( np.log(efd_a) + (efd_dt*i) )
            # efd_list_frame.append( np.exp( efd_frame ) )

        x_plot, y_plot = reconstContourCoord_efd(efd_list_frame,N,t,T)
        x_plot, y_plot = convertContour2Meander(x_plot,y_plot)
        fig,ax = plt.subplots(figsize=(8,8))
        ax.set_aspect('equal')
        ax.set_xlim(0,5)
        ax.set_ylim(-1,1)
        ax.plot(x_plot,y_plot,color='black')
        # ax.fill(x_plot,y_plot,color='black',alpha=1.0)
        fig.savefig(SAVE_PATH+os.sep+FILE_NAME+os.sep+"frame"+str(i+1)+'.pdf',bbox_inches='tight')
        plt.close()

        
        