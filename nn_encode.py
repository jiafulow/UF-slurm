import numpy as np

nlayers = 12  # 5 (CSC) + 4 (RPC) + 3 (GEM)

nvariables = (nlayers * 6) + 3 - 32

nvariables_input = (nlayers * 7) + 3

nparameters_input = 3


# ______________________________________________________________________________
class Encoder(object):

  def __init__(self, x, y, adjust_scale=0, reg_pt_scale=1.0,
               drop_ge11=False, drop_ge21=False, drop_me0=False, drop_irpc=False):
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
      # Each layer has 6 sets of features (phi, theta, bend, time, ring, fr) and 1 set of mask
      # Additionally, each road has 3 more features.
      # Some inputs are not actually used.
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

      # Drop detectors
      x_dropit = self.x_mask
      if drop_ge11:
        x_dropit[:, 9] = 1  # 9: GE1/1
      if drop_ge21:
        x_dropit[:, 10] = 1 # 10: GE2/1
      if drop_me0:
        x_dropit[:, 11] = 1 # 11: ME0
      if drop_irpc:
        x_ring_tmp = self.x_ring.astype(np.int32)
        x_ring_tmp = (x_ring_tmp == 2) | (x_ring_tmp == 3)
        x_dropit[~x_ring_tmp[:,7], 7] = 1  # 7: RE3, neither ring2 nor ring3
        x_dropit[~x_ring_tmp[:,8], 8] = 1  # 8: RE4, neither ring2 nor ring3

      self.x_phi  [x_dropit] = np.nan
      self.x_theta[x_dropit] = np.nan
      self.x_bend [x_dropit] = np.nan
      self.x_time [x_dropit] = np.nan
      self.x_ring [x_dropit] = np.nan
      self.x_fr   [x_dropit] = np.nan
      self.x_mask [x_dropit] = 1

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
      #self.x_theta_median[np.isnan(self.x_theta_median)] = np.nanmedian(self.x_theta[np.isnan(self.x_theta_median)], axis=1)  # use all types
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
        x_theta_tmp   = np.where(np.isnan(self.x_theta), 99., self.x_theta)  # take care of nan
        x_theta_tmp   = np.abs(x_theta_tmp) > theta_cuts
        self.x_phi   *= 0.000991  # GE1/1 dphi linear correlation with q/pT
        self.x_theta *= (1/12.)   # 12 integer theta units
        self.x_bend  *= 0.188082  # ME1/2 bend linear correlation with q/pT
        x_ring_tmp    = self.x_ring.astype(np.int32)
        x_ring_tmp    = (x_ring_tmp == 2) | (x_ring_tmp == 3)
        self.x_ring[x_ring_tmp] = 1  # ring 2,3 -> 1
        self.x_ring[~x_ring_tmp] = 0 # ring 1,4 -> 0
        x_fr_tmp      = self.x_fr.astype(np.int32)
        x_fr_tmp      = (x_fr_tmp == 1)
        self.x_fr[x_fr_tmp] = 1   # front chamber -> 1
        self.x_fr[~x_fr_tmp] = 0  # rear chamber  -> 0
      elif adjust_scale == 3:  # adjust by hand #2
        #theta_cuts    = np.array((6., 6., 6., 6., 6., 12., 12., 12., 12., 9., 9., 9.), dtype=np.float32)
        #theta_cuts    = np.array((6., 6., 6., 6., 6., 10., 10., 10., 10., 8., 8., 8.), dtype=np.float32)
        #x_theta_tmp   = np.where(np.isnan(self.x_theta), 99., self.x_theta)  # take care of nan
        #x_theta_tmp   = np.abs(x_theta_tmp) > theta_cuts
        if True:  # modify ring and F/R definitions
          x_ring_tmp    = self.x_ring.astype(np.int32)
          x_ring_tmp    = (x_ring_tmp == 2) | (x_ring_tmp == 3) | (x_ring_tmp == 4)
          self.x_ring[x_ring_tmp]  = 1 # ring 2,3,4 -> 1; also differentiate ring 4 (ME1/1a) from ring 1 (ME1/1b)
          self.x_ring[~x_ring_tmp] = 0 # ring 1 -> 0
          x_fr_tmp      = self.x_fr.astype(np.int32)
          x_fr_tmp      = (x_fr_tmp == 1)
          self.x_fr[x_fr_tmp]  = 1  # front chamber -> 1
          self.x_fr[~x_fr_tmp] = 0  # rear chamber  -> 0
        if True:  # zero out some variables
          self.x_bend[:, 5:11] = 0  # no bend for RPC, GEM
          self.x_time[:, :]    = 0  # no time for everyone
          self.x_ring[:, 5:12] = 0  # no ring for RPC, GEM, ME0
          self.x_fr[:, 5:12]   = 0  # no F/R for RPC, GEM, ME0
        s = [ 0.005083,  0.019142, -0.015694, -0.010744, -0.007120,  0.021043,
             -0.042957, -0.009228, -0.006421,  0.003815, -0.015731,  0.003477,
              0.596512,  0.592788,  1.459119,  1.507292,  1.062175,  0.228293,
              0.289786,  0.355389,  0.385781,  0.454640,  0.603444,  0.710116,
             -0.059926, -0.065415, -0.149701,  0.086035,  0.108797,  1.000000,
              1.000000,  1.000000,  1.000000, -0.515074, -0.599277, -0.073872,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,
              1.000000,  1.000000,  1.000000]
        self.x_copy *= s

      # Remove outlier hits by checking hit thetas
      #self.x_phi  [x_theta_tmp] = np.nan
      #self.x_theta[x_theta_tmp] = np.nan
      #self.x_bend [x_theta_tmp] = np.nan
      #self.x_time [x_theta_tmp] = np.nan
      #self.x_ring [x_theta_tmp] = np.nan
      #self.x_fr   [x_theta_tmp] = np.nan
      #self.x_mask [x_theta_tmp] = 1

      # Add variables: straightness, zone, theta_median and mode variables
      self.x_straightness = (self.x_straightness - 6.) / 6.   # scaled to [-1,1]
      self.x_zone         = (self.x_zone - 0.) / 6.           # scaled to [0,1]
      self.x_theta_median = (self.x_theta_median - 3.) / 83.  # scaled to [0,1]
      hits_to_station = np.array((5,1,2,3,4,1,2,3,4,5,2,5), dtype=np.int32)  # '5' denotes ME1/1
      assert(len(hits_to_station) == nlayers)
      self.x_mode_vars = np.zeros((self.nentries, 5), dtype=np.bool)
      self.x_mode_vars[:,0] = np.any(self.x_mask[:,hits_to_station == 5] == 0, axis=1)
      self.x_mode_vars[:,1] = np.any(self.x_mask[:,hits_to_station == 1] == 0, axis=1)
      self.x_mode_vars[:,2] = np.any(self.x_mask[:,hits_to_station == 2] == 0, axis=1)
      self.x_mode_vars[:,3] = np.any(self.x_mask[:,hits_to_station == 3] == 0, axis=1)
      self.x_mode_vars[:,4] = np.any(self.x_mask[:,hits_to_station == 4] == 0, axis=1)

      # Add dedicated GEM-CSC bend
      # Need to account for ME1/1 f or r
      self.x_gem_csc_bend = (self.x_orig[:,9] - self.x_orig[:,0])         # 9: GE1/1, 0: ME1/1
      self.x_gem_csc_bend[(self.x_mask[:,9] | self.x_mask[:,0])] = np.nan # 9: GE1/1, 0: ME1/1
      self.x_gem_csc_bend = np.hstack((self.x_gem_csc_bend[:,np.newaxis], self.x_gem_csc_bend[:,np.newaxis]))
      self.x_gem_csc_bend[(self.x_fr[:,0]!=0),0] = np.nan  # for ME1/1r bend, set ME1/1f to nan
      self.x_gem_csc_bend[(self.x_fr[:,0]!=1),1] = np.nan  # for ME1/1f bend, set ME1/1r to nan
      if adjust_scale == 3:
        self.x_gem_csc_bend *= [0.012216, 0.027306]

      # Remove NaN
      self._handle_nan_in_x(self.x_copy)
      self._handle_nan_in_x(self.x_gem_csc_bend)

      # Scale q/pT for training
      self.y_pt *= reg_pt_scale
      return

  # Copied from scikit-learn
  def _handle_zero_in_scale(self, scale):
    scale[scale == 0.0] = 1.0
    return scale

  def _handle_nan_in_x(self, x):
    x[np.isnan(x)] = 0.0
    return x

  def get_x(self, drop_columns_of_zeroes=True):
    x_new = np.hstack((self.x_phi, self.x_theta, self.x_bend,
                       self.x_time, self.x_ring, self.x_fr,
                       self.x_straightness, self.x_zone, self.x_theta_median))
    # Drop input nodes
    if drop_columns_of_zeroes:
      drop_phi   = [nlayers*0 + x for x in xrange(0,0)]  # keep everyone
      drop_theta = [nlayers*1 + x for x in xrange(0,0)]  # keep everyone
      drop_bend  = [nlayers*2 + x for x in xrange(5,11)] # no bend for RPC, GEM
      drop_time  = [nlayers*3 + x for x in xrange(0,12)] # no time for everyone
      drop_ring  = [nlayers*4 + x for x in xrange(5,12)] # no ring for RPC, GEM, ME0
      drop_fr    = [nlayers*5 + x for x in xrange(5,12)] # no F/R for RPC, GEM, ME0

      x_dropit = np.zeros(x_new.shape[1], dtype=np.bool)
      for i in drop_phi + drop_theta + drop_bend + drop_time + drop_ring + drop_fr:
        x_dropit[i] = True

      x_new = x_new[:, ~x_dropit]
    return x_new

  def get_x_mask(self):
    x_mask = self.x_mask.copy()
    return x_mask

  def get_y(self):
    y_new = self.y_pt.copy()
    return y_new

  def get_y_corrected_for_eta(self):
    y_new = self.y_pt * (np.sinh(1.8587) / np.sinh(np.abs(self.y_eta)))
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
