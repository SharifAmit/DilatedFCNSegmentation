import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))


proto = 'path-to-VGG_ILSVRC_16_layers_deploy.prototxt' 
weights = 'path-to-VGG_ILSVRC_16_layers.caffemodel'
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('Dilated_FCN-2s_VGG16/solver.prototxt')
# surgeries
custom_net = caffe.Net(proto, weights, caffe.TEST) 
surgery.transplant(solver.net, custom_net)
del custom_net


interp_layers = [k for k in solver.net.params.keys() if 'up' in k] 
surgery.interp(solver.net, interp_layers)  

# scoring
val = np.loadtxt('../seg12val.txt', dtype=str)

for _ in range(1):
    solver.step(1) 
    score.seg_tests(solver, False, val, layer='score')
