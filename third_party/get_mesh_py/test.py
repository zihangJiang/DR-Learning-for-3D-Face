import sys
sys.path.append('build')
from get_mesh import *
import numpy as np
import time
#a= time.time()
feature = np.fromfile('/raid/jzh/RimdFeature1002/Feature1/face_38_1002.dat')
a=time.time()
m = get_mesh('/raid/jzh/FeatureDistangle/data/distangle/Mean_Face.obj', feature)
print('Time cost {}'.format(time.time()-a))
