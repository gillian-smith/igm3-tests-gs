#!/usr/bin/env python3

# Custom module 2025 by Gillian Smith
# Original oggm_shop module copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import sys
sys.path.append('/home/s1639117/Documents/igm_folder/igm3/igm3-tests-gs/bhotekosi/user/code/inputs')

import os  
from igm.inputs.oggm_shop.oggm_util import oggm_util
# from .make_input_file_old import make_input_file
from igm.inputs.oggm_shop.open_gridded_data import open_gridded_data
from igm.inputs.oggm_shop.arrange_data import arrange_data

from oggm_shop_package import import_thkobs, make_input_file # we import our own make_input_file with a minor change to the original

import xarray as xr 

import numpy as np
#import matplotlib.pyplot as plt
#import os, glob, shutil, scipy
#from netCDF4 import Dataset
#import tensorflow as tf
import pandas as pd
import geopandas as gpd
from shapely import force_2d
#from scipy.interpolate import RectBivariateSpline

'''
New cfg variables:
cfg.inputs.oggm_shop.path_custom_thkobs - path to thkobs file
cfg.inputs.oggm_shop.custom_thkobs_find_method - "outline" or "RGIId"
cfg.inputs.oggm_shop.custom_thkobs_profiles_constrain
cfg.inputs.oggm_shop.custom_thkobs_profiles_test
cfg.inputs.oggm_shop.custom_thkobs_column_name
'''

def run(cfg, state):

    if cfg.inputs.oggm_shop.RGI_ID=="":
        RGI_ID = cfg.inputs.oggm_shop.RGI_IDs[0]
    else:
        RGI_ID = cfg.inputs.oggm_shop.RGI_ID

    # Get the RGI version and product from the RGI_ID
    if (RGI_ID.count('-')==4)&(RGI_ID.split('-')[1][1]=='7'):
        RGI_version = 7
        RGI_product = RGI_ID.split('-')[2]
    elif (RGI_ID.count('-')==1)&(RGI_ID.split('-')[0][3]=='6'):
        RGI_version = 6
        RGI_product = None
    else:
        print("RGI version not recognized")

    path_data = os.path.join(state.original_cwd,cfg.core.folder_data)

    if cfg.inputs.oggm_shop.RGI_ID=="":
        path_RGIs = [os.path.join(path_data,path_RGI) for path_RGI in cfg.inputs.oggm_shop.RGI_IDs]
    else:
        path_RGIs = [os.path.join(path_data,cfg.inputs.oggm_shop.RGI_ID)]

    path_file = os.path.join(path_data,cfg.inputs.oggm_shop.filename)

    if not os.path.exists(path_data):
        os.makedirs(path_data)

    # Fetch the data from OGGM if it does not exist
    if not all(os.path.exists(p) for p in path_RGIs):
        oggm_util(cfg, path_RGIs, RGI_version, RGI_product)

    # transform the data into IGM readable data if it does not exist
    if not os.path.exists(path_file):
        # make_input_file(cfg, state, path_RGIs[0], path_file, RGI_version, RGI_product)

        ds = open_gridded_data(cfg, path_RGIs[0], state)

        ds_vars = arrange_data(cfg, state, path_RGIs[0], ds, RGI_version, RGI_product)

        ds_vars = import_thkobs(cfg, ds, ds_vars, path_data, RGI_version)    

        make_input_file(cfg, ds, ds_vars, path_file)

# # Variable metadata
# def build_var_info(cfg):
#     info = {
#         "thk": ["Ice Thickness", "m"],
#         "usurf": ["Surface Topography", "m"],
#         "usurfobs": ["Surface Topography", "m"],
#         "thkobs": ["Ice Thickness", "m"],
#         "thkobs_test": ["Ice Thickness", "m"],
#         "thkinit": ["Ice Thickness", "m"],
#         "uvelsurfobs": ["x surface velocity of ice", "m/y"],
#         "vvelsurfobs": ["y surface velocity of ice", "m/y"],
#         "icemask": ["Ice mask", "no unit"],
#         "icemaskobs": ["Accumulation Mask", "bool"],
#         "dhdt": ["Ice thickness change", "m/y"]
#     }
#     if cfg.inputs.oggm_shop.sub_entity_mask:
#         info["tidewatermask"] = ["Tidewater glacier mask", "no unit"]
#         info["slopes"] = ["Average glacier surface slope", "deg"]
#     return info

# def make_input_file(cfg, ds, ds_vars, path_file):

#     # Build output dataset
#     coords = {
#         "x": ds["x"],
#         "y": ds["y"]
#     }
#     var_info = build_var_info(cfg)

#     pyproj_srs = ds.attrs.get("pyproj_srs", None)

#     ds_out = xr.Dataset(
#         {
#             v: xr.DataArray(data=ds_vars[v], dims=("y", "x"), attrs={
#                 "long_name": var_info[v][0],
#                 "units": var_info[v][1],
#                 "standard_name": v
#             }) for v in ds_vars
#         },
#         coords=coords,
#         attrs={"pyproj_srs": pyproj_srs} if pyproj_srs else {}
#     )

#     # Save to disk
#     ds_out.to_netcdf(path_file, format="NETCDF4")

# def import_thkobs(cfg, ds, ds_vars, path_data, RGI_version):
    
#     path_RGI = os.path.join(path_data,cfg.inputs.oggm_shop.RGI_ID)
#     path_outline = os.path.join(path_RGI,"outlines")
#     if not os.path.exists(path_outline):
#         os.makedirs(path_outline)
#     os.system('tar -xvzf '+ path_RGI + '/outlines.tar.gz -C ' + path_outline);
#     rgi_outline = gpd.read_file(path_outline)
#     rgi_outline.geometry = force_2d(rgi_outline.geometry)

#     x = np.squeeze(ds.variables["x"]).astype("float32").values
#     #y = np.flip(np.squeeze(nc.variables["y"]).astype("float32"))
#     y = np.squeeze(ds.variables["y"]).astype("float32").values

#     df = gpd.read_file(cfg.inputs.oggm_shop.path_custom_thkobs)
#     df.geometry = force_2d(df.geometry)
#     df = df.to_crs(rgi_outline.crs)

#     if cfg.inputs.oggm_shop.custom_thkobs_find_method=="outline":
#         # spatial join - find data points inside outline
#         df = gpd.sjoin(df,rgi_outline)
#     elif cfg.inputs.oggm_shop.custom_thkobs_find_method=="RGIId":
#         # find points labelled with same RGI ID as this glacier (even if they are outside the outline)
#         if RGI_version==6:
#             df = df[df["RGIId"]==cfg.inputs.oggm_shop.RGI_ID]
#         elif RGI_version==7:
#             df = df[df["rgi_id"]==cfg.inputs.oggm_shop.RGI_ID]

#     # this assumes every profile is identified by the last character of the profile_id
#     df_constrain = df[df["profile_id"].str.strip().str[-1].isin(cfg.inputs.oggm_shop.custom_thkobs_profiles_constrain)]
#     df_test = df[df["profile_id"].str.strip().str[-1].isin(cfg.inputs.oggm_shop.custom_thkobs_profiles_test)]
    
#     if df_constrain.empty:
#         print("No profiles for constraint")
#         ds_vars["thkobs"] = xr.full_like(ds_vars["thkinit"],np.nan)
#     else:
#         ds_vars["thkobs"] = rasterize(df_constrain,x,y,cfg.inputs.oggm_shop.custom_thkobs_column_name)
    
#     if df_test.empty:
#         print("No profiles for test")
#         ds_vars["thkobs_test"] = xr.full_like(ds_vars["thkinit"],np.nan)
#     else:
#         ds_vars["thkobs_test"] = rasterize(df_test,x,y,cfg.inputs.oggm_shop.custom_thkobs_column_name)
    
#     # Exclude thkobs_test cells from thkobs
#     MASK = ds_vars["thkobs_test"].isnull() & ~ds_vars["thkobs"].isnull()
#     ds_vars["thkobs"] = xr.where(MASK, ds_vars["thkobs"], np.nan)

#     # Exclude cells outside outline
#     ds_vars["thkobs"] = xr.where(ds_vars["icemaskobs"], ds_vars["thkobs"], np.nan)
#     ds_vars["thkobs_test"] = xr.where(ds_vars["icemaskobs"], ds_vars["thkobs_test"], np.nan)

#     count_cells_constraining = np.count_nonzero(~np.isnan(ds_vars["thkobs"].values))
#     count_cells_test = np.count_nonzero(~np.isnan(ds_vars["thkobs_test"].values))

#     print(f"# Grid cells in constraining set = {count_cells_constraining}")
#     print(f"# Grid cells in test set = {count_cells_test}")

#     return ds_vars

# def rasterize(df,x,y,thkobs_column):

#     xx = df.geometry.x
#     yy = df.geometry.y

#     thickness = df[thkobs_column].copy() 
    
#     dx = x[1]-x[0]
#     dy = y[1]-y[0]

#     # Rasterize
#     gridded = (
#     pd.DataFrame(
#         {
#             "col": np.floor((xx - np.min(x) + dx/2) / dx).astype(int),
#             "row": np.floor((yy - np.min(y) + dy/2) / dy).astype(int),
#             "thickness": thickness,
#         }
#     )
#     .groupby(["row", "col"])["thickness"])

#     # Thickness - mean over each grid cell
#     thickness_gridded = gridded.mean() # mean over each grid cell
#     thkobs = np.full((y.shape[0], x.shape[0]), np.nan) # fill array with nans
#     thickness_gridded[thickness_gridded == 0] = np.nan # put nans where we have zero thickness / no observations
#     thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded
#     thkobs_xr = xr.DataArray(thkobs,coords={'y':y,'x':x})#,attrs={'long_name':"Ice Thickness",'units':"m",'standard_name':"thkobs"})

#     return thkobs_xr