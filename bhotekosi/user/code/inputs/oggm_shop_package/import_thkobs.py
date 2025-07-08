#!/usr/bin/env python3

# Custom module 2025 by Gillian Smith

import numpy as np
#import matplotlib.pyplot as plt
import os#, glob, shutil, scipy
#from netCDF4 import Dataset
#import tensorflow as tf
import pandas as pd
import xarray as xr
import geopandas as gpd
#import shapely
#from pyproj import Transformer
#from scipy.interpolate import RectBivariateSpline

'''
New cfg variables:
cfg.inputs.oggm_shop.path_custom_thkobs - path to thkobs file
cfg.inputs.oggm_shop.custom_thkobs_find_method - "outline" or "RGIId"
cfg.inputs.oggm_shop.custom_thkobs_profiles_constrain
cfg.inputs.oggm_shop.custom_thkobs_profiles_test
cfg.inputs.oggm_shop.custom_thkobs_column_name
'''
def import_thkobs(cfg, ds, ds_vars, path_data, RGI_version):
    
    path_RGI = os.path.join(path_data,cfg.inputs.oggm_shop.RGI_ID)
    path_outline = os.path.join(path_RGI,"outlines")
    if not os.path.exists(path_outline):
        os.makedirs(path_outline)
    os.system('tar -xvzf '+ path_RGI + '/outlines.tar.gz -C ' + path_outline);
    rgi_outline = gpd.read_file(path_outline)
    from shapely import force_2d
    rgi_outline.geometry = force_2d(rgi_outline.geometry)

    x = np.squeeze(ds.variables["x"]).astype("float32").values
    #y = np.flip(np.squeeze(nc.variables["y"]).astype("float32"))
    y = np.squeeze(ds.variables["y"]).astype("float32").values

    df = gpd.read_file(cfg.inputs.oggm_shop.path_custom_thkobs)
    df.geometry = force_2d(df.geometry)
    df = df.to_crs(rgi_outline.crs)

    if cfg.inputs.oggm_shop.custom_thkobs_find_method=="outline":
        # spatial join - find data points inside outline
        df = gpd.sjoin(df,rgi_outline)
    elif cfg.inputs.oggm_shop.custom_thkobs_find_method=="RGIId":
        # find points labelled with same RGI ID as this glacier (even if they are outside the outline)
        if RGI_version==6:
            df = df[df["RGIId"]==cfg.inputs.oggm_shop.RGI_ID]
        elif RGI_version==7:
            df = df[df["rgi_id"]==cfg.inputs.oggm_shop.RGI_ID]

    # this assumes every profile is identified by the last character of the profile_id
    df_constrain = df[df["profile_id"].str.strip().str[-1].isin(cfg.inputs.oggm_shop.custom_thkobs_profiles_constrain)]
    df_test = df[df["profile_id"].str.strip().str[-1].isin(cfg.inputs.oggm_shop.custom_thkobs_profiles_test)]
    
    if df_constrain.empty:
        print("No profiles for constraint")
        ds_vars["thkobs"] = xr.full_like(ds_vars["thkinit"],np.nan)
    else:
        ds_vars["thkobs"] = rasterize(df_constrain,x,y,cfg.inputs.oggm_shop.custom_thkobs_column_name)
    
    if df_test.empty:
        print("No profiles for test")
        ds_vars["thkobs_test"] = xr.full_like(ds_vars["thkinit"],np.nan)
    else:
        ds_vars["thkobs_test"] = rasterize(df_test,x,y,cfg.inputs.oggm_shop.custom_thkobs_column_name)
    
    # Exclude thkobs_test cells from thkobs
    MASK = ds_vars["thkobs_test"].isnull() & ~ds_vars["thkobs"].isnull()
    ds_vars["thkobs"] = xr.where(MASK, ds_vars["thkobs"], np.nan)

    # Exclude cells outside outline
    ds_vars["thkobs"] = xr.where(ds_vars["icemaskobs"], ds_vars["thkobs"], np.nan)
    ds_vars["thkobs_test"] = xr.where(ds_vars["icemaskobs"], ds_vars["thkobs_test"], np.nan)

    count_cells_constraining = np.count_nonzero(~np.isnan(ds_vars["thkobs"].values))
    count_cells_test = np.count_nonzero(~np.isnan(ds_vars["thkobs_test"].values))

    print(f"# Grid cells in constraining set = {count_cells_constraining}")
    print(f"# Grid cells in test set = {count_cells_test}")

    return ds_vars

def rasterize(df,x,y,thkobs_column):

    xx = df.geometry.x
    yy = df.geometry.y

    thickness = df[thkobs_column].copy() 
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]

    # Rasterize
    gridded = (
    pd.DataFrame(
        {
            "col": np.floor((xx - np.min(x) + dx/2) / dx).astype(int),
            "row": np.floor((yy - np.min(y) + dy/2) / dy).astype(int),
            "thickness": thickness,
        }
    )
    .groupby(["row", "col"])["thickness"])

    # Thickness - mean over each grid cell
    thickness_gridded = gridded.mean() # mean over each grid cell
    thkobs = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
    thickness_gridded[thickness_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
    thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded
    thkobs_xr = xr.DataArray(thkobs,coords={'y':y,'x':x})#,attrs={'long_name':"Ice Thickness",'units':"m",'standard_name':"thkobs"})

    return thkobs_xr