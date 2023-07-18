#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ff.ff_class module
# ******************************************************************************

import numpy as np

__version__ = '0.2.8'


class GAFF_Angle():
    """
        GAFF_Angle() class
    """
    def __init__(self, ff_type=None, k=None, theta0=None):
        self.type = ff_type
        self.k = k
        self.theta0 = theta0
        self.theta0_rad = theta0*(np.pi/180)
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': float(self.k),
            'theta0': float(self.theta0),
        }
        return dic
    
    
class GAFF_Dihedral():
    """
        GAFF_Dihedral() class
    """
    def __init__(self, ff_type=None, k=[], d0=[], m=None, n=[]):
        self.type = ff_type
        self.k = np.array(k)
        self.d0 = np.array(d0)
        self.d0_rad = np.array(d0)*(np.pi/180)
        self.m = m
        self.n = np.array(n)
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': [float(x) for x in self.k],
            'd0': [float(x) for x in self.d0],
            'm': int(self.m),
            'n': [int(x) for x in self.n],
        }
        return dic


class GAFF_Improper():
    """
        GAFF_Improper() class
    """
    def __init__(self, ff_type=None, k=None, d0=-1, n=None):
        self.type = ff_type
        self.k = k
        self.d0 = d0
        self.n = n
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': float(self.k),
            'd0': float(self.d0),
            'n': int(self.n),
        }
        return dic

