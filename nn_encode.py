import numpy as np

nlayers = 12  # 5 (CSC) + 4 (RPC) + 3 (GEM)

nvariables = (nlayers * 6) + 8

nvariables_input = (nlayers * 7) + 3

nparameters_input = 3

class Encoder(object):

  def __init__(self, x, y, adjust_scale=0, reg_pt_scale=1.0):
    if x is not None and y is not None:
      assert(x.shape[1] == nvariables_input)
      assert(y.shape[1] == nparameters_input)
      assert(x.shape[0] == y.shape[0])

      self.nentries = x.shape[0]
      self.x_orig  = x
      self.y_orig  = y
      self.x_copy  = x.copy()
      self.y_copy  = y.copy()

      # Get views
      self.x_phi   = self.x_copy[:, nlayers*0:nlayers*1]
      self.x_theta = self.x_copy[:, nlayers*1:nlayers*2]
      self.x_bend  = self.x_copy[:, nlayers*2:nlayers*3]
      self.x_time  = self.x_copy[:, nlayers*3:nlayers*4]
      self.x_ring  = self.x_copy[:, nlayers*4:nlayers*5]
      self.x_fr    = self.x_copy[:, nlayers*5:nlayers*6]
      self.x_mask  = self.x_copy[:, nlayers*6:nlayers*7].astype(np.bool)  # this makes a copy
      self.x_road  = self.x_copy[:, nlayers*7:nlayers*8]  # ipt, ieta, iphi
      self.y_pt    = self.y_copy[:, 0]  # q/pT
      self.y_phi   = self.y_copy[:, 1]
      self.y_eta   = self.y_copy[:, 2]

      # Make event weight
      #self.w       = np.ones(self.y_pt.shape, dtype=np.float32)
      self.w       = np.abs(self.y_pt)/0.2 + 1.0

      # Straightness & zone
      self.x_straightness = self.x_road[:, 0][:, np.newaxis]
      self.x_zone         = self.x_road[:, 1][:, np.newaxis]

      # Subtract median phi from hit phis
      #self.x_phi_median    = self.x_road[:, 2] * 32 - 16  # multiply by 'quadstrip' unit (4 * 8)
      self.x_phi_median    = self.x_road[:, 2] * 16 - 8  # multiply by 'doublestrip' unit (2 * 8)
      self.x_phi_median    = self.x_phi_median[:, np.newaxis]
      self.x_phi          -= self.x_phi_median

      # Subtract median theta from hit thetas
      self.x_theta_median  = np.nanmedian(self.x_theta[:,:5], axis=1)  # CSC only
      self.x_theta_median[np.isnan(self.x_theta_median)] = np.nanmedian(self.x_theta[np.isnan(self.x_theta_median)], axis=1)  # use all
      self.x_theta_median  = self.x_theta_median[:, np.newaxis]
      self.x_theta        -= self.x_theta_median

      # Standard scales
      # + Remove outlier hits by checking hit thetas
      if adjust_scale == 0:  # do not adjust
        x_theta_tmp = np.abs(self.x_theta) > 10000.0
      elif adjust_scale == 1:  # use mean and std
        self.x_mean  = np.nanmean(self.x_copy, axis=0)
        self.x_std   = np.nanstd(self.x_copy, axis=0)
        self.x_std   = self._handle_zero_in_scale(self.x_std)
        self.x_copy -= self.x_mean
        self.x_copy /= self.x_std
        x_theta_tmp = np.abs(self.x_theta) > 1.0
      elif adjust_scale == 2:  # adjust by hand
        theta_cuts    = np.array((6., 6., 6., 6., 6., 12., 12., 12., 12., 9., 9., 9.), dtype=np.float32)
        x_theta_tmp   = np.abs(self.x_theta) > theta_cuts
        self.x_phi   *= 0.000991  # GE1/1 dphi linear correlation with q/pT
        self.x_theta *= (1/12.)   # 12 integer theta units
        self.x_bend  *= 0.188082  # ME1/2 bend linear correlation with q/pT
        x_ring_tmp    = self.x_ring.astype(np.int32)
        x_ring_tmp    = (x_ring_tmp == 1) | (x_ring_tmp == 4)
        self.x_ring[x_ring_tmp] = 0  # ring 1,4 -> 0
        self.x_ring[~x_ring_tmp] = 1 # ring 2,3 -> 1
        x_fr_tmp      = self.x_fr.astype(np.int32)
        x_fr_tmp      = (x_fr_tmp == 0)
        self.x_fr[x_fr_tmp] = 0  # rear chamber
        self.x_fr[~x_fr_tmp] = 1  # front chamber
      elif adjust_scale == 3:  # adjust by hand #2
        #theta_cuts    = np.array((6., 6., 6., 6., 6., 12., 12., 12., 12., 9., 9., 9.), dtype=np.float32)
        theta_cuts    = np.array((6., 6., 6., 6., 6., 10., 10., 10., 10., 8., 8., 8.), dtype=np.float32)
        x_theta_tmp   = np.abs(self.x_theta) > theta_cuts
        self.x_bend[:, 5:9] = 0  # do not use RPC bend
        x_ring_tmp    = self.x_ring.astype(np.int32)
        x_ring_tmp    = (x_ring_tmp == 1) | (x_ring_tmp == 4)
        self.x_ring[x_ring_tmp] = 0  # ring 1,4 -> 0
        self.x_ring[~x_ring_tmp] = 1 # ring 2,3 -> 1
        x_fr_tmp      = self.x_fr.astype(np.int32)
        x_fr_tmp      = (x_fr_tmp == 0)
        self.x_fr[x_fr_tmp] = 0  # rear chamber
        self.x_fr[~x_fr_tmp] = 1  # front chamber
        s = [ 0.005938,  0.012215, -0.015295, -0.01128 , -0.008312,  0.013397,
             -0.026915, -0.009992, -0.00771 ,  0.00414 , -0.01836 ,  0.005182,
              0.601718,  0.581062,  1.470157,  1.488481,  1.062894,  0.219505,
              0.290072,  0.345564,  0.388044,  0.509891,  0.60273 ,  0.718745,
              0.823569,  0.48355 ,  1.662107, -1.182058, -1.213553,  0.426063,
              0.424225, -0.41433 , -0.403946,  0.721469,  1.468051, -0.111478,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,
              1.      ,  1.      ,  1.      ]
        self.x_copy *= s

      # Remove outlier hits by checking hit thetas
      self.x_phi  [x_theta_tmp] = np.nan
      self.x_theta[x_theta_tmp] = np.nan
      self.x_bend [x_theta_tmp] = np.nan
      self.x_time [x_theta_tmp] = np.nan
      self.x_ring [x_theta_tmp] = np.nan
      self.x_fr   [x_theta_tmp] = np.nan
      self.x_mask [x_theta_tmp] = 1.0

      # Add variables: straightness, zone, theta_median and mode variables
      self.x_straightness = (self.x_straightness - 6.) / 6.  # scaled to [-1,1]
      self.x_zone         = (self.x_zone - 0.) / 5.  # scaled to [0,1]
      self.x_theta_median = (self.x_theta_median - 3.) / 83.  # scaled to [0,1]
      hits_to_station = np.array((5,1,2,3,4,1,2,3,4,5,2,5), dtype=np.int32)  # '5' denotes ME1/1
      assert(len(hits_to_station) == nlayers)
      self.x_mode_vars = np.zeros((self.nentries, 5), dtype=np.bool)
      self.x_mode_vars[:,0] = np.any(self.x_mask[:,hits_to_station == 5] == 0, axis=1)
      self.x_mode_vars[:,1] = np.any(self.x_mask[:,hits_to_station == 1] == 0, axis=1)
      self.x_mode_vars[:,2] = np.any(self.x_mask[:,hits_to_station == 2] == 0, axis=1)
      self.x_mode_vars[:,3] = np.any(self.x_mask[:,hits_to_station == 3] == 0, axis=1)
      self.x_mode_vars[:,4] = np.any(self.x_mask[:,hits_to_station == 4] == 0, axis=1)

      # Remove NaN
      #np.nan_to_num(self.x_copy, copy=False)
      self.x_copy[np.isnan(self.x_copy)] = 0.0

      # Scale q/pT for training
      self.y_pt *= reg_pt_scale
      return

  # Copied from scikit-learn
  def _handle_zero_in_scale(self, scale):
    scale[scale == 0.0] = 1.0
    return scale

  def get_x(self):
    #x_new = self.x_phi
    x_new = np.hstack((self.x_phi, self.x_theta, self.x_bend, self.x_time, self.x_ring, self.x_fr, self.x_straightness, self.x_zone, self.x_theta_median, self.x_mode_vars))
    return x_new

  def get_x_mask(self):
    x_mask = self.x_mask.copy()
    return x_mask

  def get_y(self):
    y_new = self.y_pt.copy()
    return y_new

  def get_w(self):
    w_new = self.w.copy()
    return w_new

  def save_encoder(self, filepath):
    np.savez_compressed(filepath, x_mean=self.x_mean, x_std=self.x_std)

  def load_endcoder(self, filepath):
    loaded = np.load(filepath)
    self.x_mean = loaded['x_mean']
    self.x_std = loaded['x_std']
