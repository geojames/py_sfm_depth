#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'James T. Dietrich'
__contact__ = 'james.t.dietrich@dartmouth.edu'
__copyright__ = '(c) James Dietrich 2016'
__license__ = 'MIT'
__date__ = '13 July 2016'
__version__ = '1.0'
__status__ = "initial release"
__url__ = "https://github.com/geojames/py_sfm_depth"

"""
Name:           py_sfm_refraction_corr.py
Compatibility:  Python 2.7
Description:    This program performs a per-camera refration correction on a 
                Structure-from-Motion point cloud. Additional documnetation,
                sample data, and a tutorial are availible from the GitHub
                address below.

URL:            https://github.com/geojames/py_sfm_depth

Requires:       wx, numpy, pandas, sympy, matplotlib

Dev ToDo:       1) speed up camera geometry calculations

AUTHOR:         James T. Dietrich
ORGANIZATION:   Dartmouth College
Contact:        james.t.dietrich@dartmouth.edu
Copyright:      (c) James Dietrich 2016

Licence:        MIT
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""

#------------------------------------------------------------------------------
# Imports
import os
import sys
import wx
import numpy as np
import pandas as pd
import sympy.geometry as spg
import matplotlib.path as mplPath
from datetime import datetime
# END IMPORTS

def footprints(cam,sensor,base_elev): 
    """
    This function calculates the instantaneous field of view (IFOV) for 
    the camera(s) that are passed.\n
    Vars:\n
    \t cam = pandas dataframe (n x ~6, fields: x,y,z,yaw,pitch,roll)\n
    \t sensor = pandas dataframe (1 x 3, fields: focal, sensor_x, sensor_y):
    \t focal length (mm), sensor x dim (mm), sensor y dim (mm)\n
    \t base_elev = average elevation of your site (meters, or in the same
    \t measure as your coordinates)\n
    Creates approx. coordinates for sensor
    corners (north-oriented and zero pitch) at the camera's x,y,z. Rotates
    the sensor coords in 3D space to the camera's pitch and yaw angles (roll
    angles are ignored for now) and projects corner rays through the camera 
    x,y,z to a approx ground plane. The intersection of the rays with the
    ground are the corners of the photo footprint.\n
    *** Photos that have picth angles that cause the horizon to be visable will
    cause the UL and UR path coordniates to wrong. These cameras are 
    disreguarded and the footprint will be set to NaN in the output.***\n 
    RETURNS: footprints = Pandas dataframe (n x 1) of Matplotlib Path objects()
    """
    # Setup DF to house camera footprint polygons
    footprints = pd.DataFrame(np.zeros((cam.shape[0],1)), columns=['fov'])
    
    # convert sensor dimensions to meters, divide x/y for corner coord calc
    f = sensor.focal[0] * 0.001
    sx = sensor.sensor_x[0] / 2 * 0.001
    sy = sensor.sensor_y[0] / 2 * 0.001

    # calculate the critical pitch (in degrees) where the horizon will be 
    #   visible with the horizon viable, the ray projections go backward 
    #   and produce erroneous IFOV polygons (90 - 0.5*vert_fov)
    crit_pitch = 90 - np.rad2deg(np.arctan(sy / f))
    
    # User Feedback
    print "Proccesing Camera IFOVs (%i total)..." %(cam.shape[0])
    sys.stdout.flush()
     
    # for each camera...
    for idx, row in cam.iterrows():
        
        # check is the camera pitch is over the critical value
        if row.pitch < crit_pitch:
            
            # sensor corners (UR,LR,LL,UL), north-oriented and zero pitch
            corners = np.array([[row.x+sx,row.y-f,row.z+sy],
                               [row.x+sx,row.y-f,row.z-sy],
                               [row.x-sx,row.y-f,row.z-sy],
                               [row.x-sx,row.y-f,row.z+sy]])
            
            # offset corner points by cam x,y,z for rotation
            cam_pt = np.atleast_2d(np.array([row.x, row.y, row.z]))
            corner_p = corners - cam_pt
    
            # get pitch and yaw from the camera, convert to radians
            pitch = np.deg2rad(90.0-row.pitch)
            roll = np.deg2rad(row.roll)
            yaw = np.deg2rad(row.yaw)
            
            # setup picth rotation matrix (r_x) and yaw rotation matrix (r_z)
            r_x = np.matrix([[1.0,0.0,0.0],
                             [0.0,np.cos(pitch),-1*np.sin(pitch)],
                             [0.0,np.sin(pitch),np.cos(pitch)]])
                             
            r_y = np.matrix([[np.cos(roll),0.0,np.sin(roll)],
                             [0.0,1.0,0.0],
                             [-1*np.sin(roll),0.0,np.cos(roll)]])
            
            r_z =  np.matrix([[np.cos(yaw),-1*np.sin(yaw),0],
                              [np.sin(yaw),np.cos(yaw),0],
                              [0,0,1]])
            
            # rotate corner_p by r_x, then r_z, add back cam x,y,z offsets
            # produces corner coords rotated for pitch and yaw
            p_pr = np.matmul(np.matmul(corner_p, r_x),r_y)            
            p_out = np.matmul(p_pr, r_z) + cam_pt
            
            # GEOMETRY
            # Set Sympy 3D point for the camera and a 3D plane for intersection
            cam_sp = spg.Point3D(row.x, row.y, row.z)
            plane = spg.Plane(spg.Point3D(row.x, row.y, base_elev),
                                      normal_vector=(0,0,1))
            
            # blank array for footprint intersection coords
            inter_points = np.zeros((corners.shape[0],2))
            
            # for each sensor corner point
            idx_b = 0
            for pt in np.asarray(p_out):
                
                # create a Sympy 3D point and create a Sympy 3D ray from 
                #   corner point through camera point
                pt_sp = spg.Point3D(pt[0],pt[1],pt[2])
                ray = spg.Ray3D(pt_sp,cam_sp)
                
                # calculate the intersection of the ray with the plane                
                inter_pt = plane.intersection(ray)
                
                # Extract out the X,Y coords fot eh intersection point
                #   ground intersect points will be in this order (LL,UL,UR,LR)
                inter_points[idx_b,0] = inter_pt[0].x.evalf()
                inter_points[idx_b,1] = inter_pt[0].y.evalf()
                
                idx_b += 1
        
        # if crit_pitch is exceeded set inter_points to NaN
        else:
            inter_points = np.full((4,2),np.nan)
        
        # append inter_points to footprints as a matplotlib path object
        footprints.fov[idx] = mplPath.Path(inter_points)
        
        # User feedback
        if (idx+1) % 10 == 0:
            print "%i cameras processed..." %(idx+1)
            sys.stdout.flush()
    
    return footprints
# END - footprints
        
def visibility(cam, footprints, targets):
    """    
    This function tests is the target points (x,y only) are "visable" (i.e.
    within the photo footprints) and calculates the "r" angle for the refraction 
    correction\n
    Vars:\n
    \t cam = Pandas dataframe (n x ~6, fields: x,y,z,yaw,pitch,roll)\n
    \t footprints = Pandas dataframe (n x 1) of Matplotlib Path objects\n
    \t targets = Pandas dataframe (n x ~3, fields: x,y,sfm_z...)\n
    
    RETURNS: r_filt = numpy array (n_points x n_cams) of filtered "r" angles.\n
    Points that are not visable to a camera will have a NaN "r" angle. 
    """
    
    # Setup boolean array for visability
    vis = np.zeros((targets.shape[0],cam.shape[0])) 
    
    # for each path objec in footprints, check is the points in targets are
    #   within the path polygon. path.contains_points returns boolean.
    #   the results are accumulated in the vis array.
    for idx in range(footprints.shape[0]):
        path = footprints.fov[idx]
        vis[:,idx] = path.contains_points(np.array([targets.x.values, targets.y.values]).T)
    
    # calculate the coord. deltas between the cameras and the target
    dx = np.atleast_2d(cam.x.values) - np.atleast_2d(targets.x.values).T
    dy = np.atleast_2d(cam.y.values) - np.atleast_2d(targets.y.values).T
    dz = np.atleast_2d(cam.z.values) - np.atleast_2d(targets.sfm_z).T
    
    # calc xy distance (d)
    d = np.sqrt((dx)**2+(dy)**2)
    
    # calc inclination angle (r) from targets to cams
    r = np.rad2deg(np.arctan(d/dz))
    
    r_filt = r * vis
    r_filt[r_filt == 0] = np.nan
          
    return r_filt

def correction(r,target):
    """Performs the per camera refraction correction on a target point.
    Refer to the documentation for the specifics"""
    
    # convert r array to radians for trig calculations
    ang_r = np.radians(r)

    # calculate the refraction angle i 
    ang_i = np.arcsin(1.0/1.337 * np.sin(ang_r))
    
    # calculate the apparent depth from the water surface elev. and the 
    #   target SfM elevation
    target['h_a'] = target.w_surf - target.sfm_z

    # calculate the distance from the point to the air/water interface
    x_dist = np.array([target.h_a.values]).T * np.tan(ang_r)
    
    # calculate the corrected (actual) depth
    h =  x_dist / np.tan(ang_i)
   
    # subtract the corrected depth from the water surface elevation to get the
    #   corrected elevation
    cor_elev = np.array([target.w_surf]).T - h
    
    # append the mean values for the actual depth and corrected elevation to
    #   the target data frame   
    target['h_avg'] = np.nanmean(h, axis = 1)
    target['corElev_avg'] = np.nanmean(cor_elev, axis = 1)
    
    # some extra statistics for playing around (commented out by default)
#    target['h_std'] = np.nanstd(h, axis = 1)
#    target['h_med'] = np.nanmedian(h, axis = 1)
#    target['h_min'] = np.nanmin(h, axis = 1)
#    target['h_max'] = np.nanmax(h, axis = 1)
#    target['h_25'] = np.nanpercentile(h, 25,axis = 1)
#    target['h_55'] = np.nanpercentile(h, 55,axis = 1)
#    target['h_60'] = np.nanpercentile(h, 60,axis = 1)
#    target['h_75'] = np.nanpercentile(h, 75,axis = 1)
#    target['h_80'] = np.nanpercentile(h, 80,axis = 1)
#    target['h_90'] = np.nanpercentile(h, 90,axis = 1)
#    target['h_95'] = np.nanpercentile(h, 95,axis = 1)
#    target['h_iqr'] = target['h_75'] - target['h_25']
#    target['h_lif'] = target['h_25'] - (1.5 * target['h_iqr'])
#    target['h_uif'] = target['h_75'] + (1.5 * target['h_iqr'])
#    target['h_lof'] = target['h_25'] - (3 * target['h_iqr'])
#    target['h_uof'] = target['h_75'] + (3 * target['h_iqr']) 
#    mild_out = np.zeros_like(h)
#    ext_out = np.zeros_like(h)
#    mild_out[h < np.atleast_2d(target['h_lif']).T] = 1
#    mild_out[h > np.atleast_2d(target['h_uif']).T] = 1   
#    ext_out[h < np.atleast_2d(target['h_lof']).T] = 1
#    ext_out[h > np.atleast_2d(target['h_uof']).T] = 1   
#    target['h_mildout'] = np.nansum(mild_out, axis = 1)
#    target['h_extout'] = np.nansum(ext_out, axis = 1)

    
    # return the target dataframe
    return target
# END def correction

def timer(length,start_t):
    """timer function to calculate the running time"""
    
    num_proc = sum(length)    
    
    # time since processing started
    t_step = datetime.now() - start_t

    if t_step.total_seconds() <= 60:
        print "-> Finished %i points in %0.2f secs" %(num_proc,t_step.total_seconds())
    else:
        ts = t_step.total_seconds() / 60    
        print "-> Finished %i points in %0.2f mins" %(num_proc,ts)
  
#END def timer

#-----------
# MAIN
def main():
    
    # INPUTS - see sample_data folder in GitHub repository for file header formats
    # from wx flie chooser...
    # start WX for file choosing
    app = wx.App()
    
    # target points - as CSV point cloud (x,y,z,w_surf,r,g,b) from CloudCompare
    #   will be read in 7500 point chunks for memory managment purposes
    target_file = wx.FileSelector("Open Point Cloud",default_path=os.getcwd(),
                               wildcard = "Comma-Delimited Files (*.csv)|*.csv")
    targets = pd.read_csv(target_file, chunksize = 7500)
    path = os.path.dirname(target_file)
    
    # camera file - from Photoscan (Name, Position, Orientation...)
    cam_file = wx.FileSelector("Open Camera File",default_path=path,
                               wildcard = "Comma-Delimited Files (*.csv)|*.csv")
    cams = pd.read_csv(cam_file)
    
    
    # camera sensor parameters - user generated
    sensor_file = wx.FileSelector("Open Sensor File",default_path=path,
                               wildcard = "Comma-Delimited Files (*.csv)|*.csv")
    sensor = pd.read_csv(sensor_file)
    
    # OUTPUT - CSV file for saving outputs
    outfile = wx.FileSelector("Output File",default_path=path,
                              wildcard = "Comma-Delimited Files (*.csv)|*.csv")
    out_exists = os.path.exists(outfile)
    
    # check if the output file exists
    while out_exists == True:
        print "The output file already exists, choose another..."
        outfile = wx.FileSelector("Select Another Output File",default_path=path,
                              wildcard = "Comma-Delimited Files (*.csv)|*.csv")
        out_exists = os.path.exists(outfile)
       
    # clean up the WX app
    app.Destroy()
    
    # user feedback
    print "Data Loaded..."
    sys.stdout.flush()
    
    # record the start time of the actual processing
    start_time = datetime.now()
    
    # array for count of total points
    count = []
    
    # Main Processing Loop, for each chunk of points from the reader
    for idx, tar in enumerate(targets):
        
        count.append(tar.shape[0])    
        
        if idx == 0:
            # establish mean elevation for footprint mapping from the mean 
            #   elevation of the target points
            base_elev = np.mean(tar.sfm_z)      
            
            # build camera footprints
            foot_prints = footprints(cams,sensor,base_elev)
            
            cam_end_time = datetime.now()
            mins_c = (cam_end_time - start_time).total_seconds() / 60
            print "Processed %i cameras in %0.2f minutes" %(np.count_nonzero(cams.x),mins_c)
          
        # test the visability of target point based on the camera footprints
        cam_r = visibility(cams,foot_prints,tar)
        
        # perform the refraction correction
        tar_out = correction(cam_r, tar)
        
        # output - for the first chunk write header row, else append subsequent
        #   chunks without headers
        if idx == 0:
            with open(outfile, 'a') as f:
                tar_out.to_csv(f, header=True, index=False)
        else:
            with open(outfile, 'a') as f:
                 tar_out.to_csv(f, header=False, index=False)

            # user feedback, def timer
        timer(count, cam_end_time)

    # User feedback on the total processing time
    tot_count = sum(count)
    tot_time = (datetime.now() - start_time).total_seconds() / 60
    print "%i points processed, Total Running Time = %0.2f minutes" %(tot_count,tot_time)

if __name__ == "__main__":
    main()