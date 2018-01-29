import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))



weights = 'Dilated_FCN-2s_VGG16/snapshot/vgg16surgery.caffemodel'
# init
#restoring ='Dilated_FCN-2s_VGG16/snapshot/voctraining_iter_100000.solverstate' #uncomment this out for resuming training  
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('voc-dil-rnn/solver.prototxt')
solver.net.copy_from(weights)
# solver.restore(restoring) #uncomment this out while restoring for resuming training  

interp_layers = [k for k in solver.net.params.keys() if 'up' in k] 
surgery.interp(solver.net, interp_layers)  

# scoring
val = np.loadtxt('seg12val.txt', dtype=str)

for _ in range(20):
    solver.step(5000) 
    score.seg_tests(solver, False, val, layer='score')
