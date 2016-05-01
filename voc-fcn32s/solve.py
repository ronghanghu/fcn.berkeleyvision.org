import caffe
import sys
import surgery, score

import numpy as np
import os

#import setproctitle
#setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '/home/ronghang/workspace/caffe/models/vgg_16_layers/VGG_ILSVRC_16_layers.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('./voc-fcn32s/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

## scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)

solver.step(200000)
#for _ in range(25):
#    solver.step(4000)
#    score.seg_tests(solver, False, val, layer='score')
