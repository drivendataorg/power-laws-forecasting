import os
import pandas as pd
import sys
import time
from datetime import datetime
import numpy as np


# returns (original_path_to_downloaded_data, path_to_save_produced_data)

def get_paths():
  if os.name == 'nt':                          # Windows
    if (os.path.isfile('C:\\notebook.id')):
      return ('I:\\ml\\energy\\', 'I:\\ml\\energy\\my\\')
    else:
      return ('F:\\ml\\energy\\', 'F:\\ml\\energy\\my\\')
  else:                                        # UNIX
    return ('data/','data/my/')



start = time.time()
lasttime = time.time()

def prtime(*args, **kwargs):
  global lasttime
  print(" ".join(map(str,args)), '|time:', str(datetime.now()),'|',time.time() - start,'secs from start', time.time() - lasttime, 'secs from last', **kwargs)
  lasttime = time.time()
  sys.stdout.flush()



# Below come hyperparameter tuning functions
# Not cleaned up yet and cumbersome, but can be safely skipped when trying to grasp the main logic.



def create_vector():
  l = int(np.random.random()*6+1)
  v = []
  for i in range(l):
    new_val = int(1+2**(np.random.random()*8))
    if new_val not in v:
      v.append(new_val)
  return sorted(v)


def update_vector(v):
  v = v.copy()
  a = np.random.random()
  if a < 0.25:
    print('update_vector : creating new')
    v = create_vector()
  elif a < 0.5:                 # changing one value
    print('update_vector : changing one value')
    idx = np.random.choice(len(v))
    a2 = np.random.random()
    if (a2 < 0.3) and (v[idx] > 1):
      print('update_vector : decreasing')
      cnt = 0
      while cnt < 100:
        cnt+=1
        new_val = max(1,v[idx] - int(2**(np.random.random()*6)))
        if new_val not in v:
          v[idx] =  new_val 
          break
    elif (a2 < 0.6) and (v[idx] < 256):
      print('update_vector : increasing')
      cnt = 0
      while cnt<100:
        cnt += 1
        new_val = min(256,v[idx] + int(2**(np.random.random()*6)))
        if new_val not in v:
          v[idx] = new_val
          break
  elif (a < 0.75) and (len(v) > 1):                 # removing one value
    print('update_vector : removing')
    idx = np.random.choice(len(v))
    if idx < len(v)-1:
      v = v[:idx]+v[idx+1:]
    else:
      v = v[:-1]
  else:                   # adding one value
    print('update_vector : adding one value')
    cnt = 0
    while cnt<100:
      cnt += 1
      new_val = int(1+2**(np.random.random()*8))
      if new_val not in v:
        break
    v.append(new_val)
    v = sorted(v)
  return sorted(v)
   


def update_params(total_best_params_, grd):
  total_best_params = total_best_params_.copy() if total_best_params_ is not None else None
  if total_best_params is not None:
    for p in grd:
      if p not in total_best_params:
        print('extending params with', p)
        idx = np.random.choice(len(grd[p]))
        total_best_params[p] = grd[p][idx]
      
    
  if (total_best_params is None) or (np.random.random() < 0.1):  # selecting random parameters
    print('selecting random parameters')
    params = {}
    cnt = 0
    while cnt < 100:
      cnt += 1
      for p in grd:
        idx = np.random.choice(len(grd[p]))
        params[p] = grd[p][idx]
        if type(params[p]) == list:
          if np.random.random()<0.2:
            print('updating vector', p)
            params[p] = update_vector(params[p])
      if 'conv_strides' not in params:
        break
      if (params['conv_strides'] == 1) or (params['conv_dilation_rate'] == 1):
        break

  else:                                                      # updating a few parameters
    parameters_changed = 0
    params = total_best_params.copy()
    while (parameters_changed == 0) or (np.random.random() < 0.5):   
      p = np.random.choice(list(params.keys()))
      if type(params[p]) == list:
         print('updating vector', p)
         params[p] = update_vector(params[p])
         parameters_changed += 1
      else:
        while ((params[p] == total_best_params[p]) and (len(grd[p]) > 1)):# or (('conv_strides' in params) and (params['conv_strides']>1 and params['conv_dilation_rate']>1)):
          print('adjusting parameter',p) 
          if np.random.random() < 0.5:                         # changing by one
            idx = grd[p].index(params[p])
            if (idx == len(grd[p])-1) or ((idx > 0) and (np.random.random() < 0.5)):         # changing to previous item from list
              print('decreasing parameter')
              params[p] = grd[p][idx-1] 
            else:
              print('increasing parameter')
              if idx < len(grd[p])-1:                          # changing to next item from list
                params[p] = grd[p][idx+1]
          else:
            print('choosing random ')
            idx = np.random.choice(len(grd[p]))
            params[p] = grd[p][idx]
      if len(grd[p]) > 1 or type(params[p]) == list:
        print('changed param',p,'from', total_best_params[p],'to',params[p])
        parameters_changed += 1
      if (p == 'conv_strides') and (params[p] > 1):
        params['conv_dilation_rate'] = 1
      if (p == 'conv_dilation_rate') and (params[p] > 1):
        params['conv_strides'] = 1

  for p in params:
    if type(params[p]) == np.int32:
      params[p] = int(params[p])
  print('trying params :', params)
  return params
