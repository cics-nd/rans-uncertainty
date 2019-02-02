"""
Converts svaed PyTorch NN to ONNX to Caffe2 protobuf files for loading in c++
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""

import sys, getopt
import numpy as np
import torch as th
import torch.onnx
import torch.nn.functional as F

import onnx
import caffe2.python.onnx.backend as backend

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from caffe2.python.predictor import mobile_exporter #Import the caffe2 mobile exporter
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils

dtype = th.FloatTensor
class TurbNN(th.nn.Module):
    """
    Note: This must be fully consistent with the architecture
    """
    def __init__(self, D_in, H, D_out):
        """
        Architecture of the turbulence deep neural net
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
        """
        super(TurbNN, self).__init__()
        self.linear1 = th.nn.Linear(D_in, H)
        self.f1 = th.nn.LeakyReLU()
        self.linear2 = th.nn.Linear(H, H)
        self.f2 = th.nn.LeakyReLU()
        self.linear3 = th.nn.Linear(H, H)
        self.f3 = th.nn.LeakyReLU()
        self.linear4 = th.nn.Linear(H, H)
        self.f4 = th.nn.LeakyReLU()
        self.linear5 = th.nn.Linear(H, int(H/5))
        self.f5 = th.nn.LeakyReLU()
        self.linear6 = th.nn.Linear(int(H/5), int(H/10))
        self.f6 = th.nn.LeakyReLU()
        self.linear7 = th.nn.Linear(int(H/10), D_out)
        self.log_beta = Parameter(th.Tensor([1.]).type(dtype))

    def forward(self, x):
        """
        Forward pass of the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x D_out] matrix of neural network outputs
        """
        lin1 = self.f1(self.linear1(x))
        lin2 = self.f2(self.linear2(lin1))
        lin3 = self.f3(self.linear3(lin2))
        lin4 = self.f4(self.linear4(lin3))
        lin5 = self.f5(self.linear5(lin4))
        lin6 = self.f6(self.linear6(lin5))
        out = self.linear7(lin6)

        return out

    def loadNeuralNet(self, filename):
        '''
        Load the current neural network state
        Args:
            filename (string): name of the file to save the neural network in
        '''
        self.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))

    def onnxExport(self, filename):
        '''
        Export the neural network to an proto file using the ONNX format
        Args:
            filename (string): name of the file to save the neural network in
        '''

        # Model input tensor (values do not matter, just defines the shape of inputs we should expect)
        batchsize = 1
        x = Variable(th.randn(batchsize, 5), requires_grad=True)
	    # Provide labels for input and output layers (needed in Caffe2)
        # These labels for the input and output are essential as they are the way we can access 
        # the vectors with the C++ API. One can reference reynoldsNet.C for details on C++
        input_names = [ "input" ] + [ "learned_%d" % i for i in range(10) ]
        output_names = [ "output" ]
	    # Now export to onnx format
        torch.onnx.export(self, x, filename+'.onnx', verbose=True, export_params=True, input_names=input_names, output_names=output_names)

if __name__ == '__main__':

    # Default parameters
    netNum = 0
    netDir = 'torchNets'

    # Can put in -n as a command line parameter to control which network is converted
    arg = sys.argv[1:]
    try:
        opts, args = getopt.getopt(arg,"n:",["net="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-n','--net'):
            netNum = int(arg)
        if opt in ('-d','--dir'):
            netDir = arg
    
    # First load up the neural network
    # Must be consistent with the saved neural network
    turb_nn = TurbNN(D_in=5, H=200, D_out=10)

    print('Reading PyTorch net work : {}/foamNet-{:d}.nn'.format(netDir, netNum))
    turb_nn.loadNeuralNet('{}/foamNet-{:d}.nn'.format(netDir, netNum))
    print('Converting to foamNet.onnx')
    turb_nn.onnxExport('foamNet')

    # Load the ONNX ModelProto object. model is a standard Python protobuf object
    print('Loading onnx network')
    model = onnx.load("foamNet.onnx")
    onnx.checker.check_model(model)

    print('Converting to Caffe 2 ini_net and predict_net')
    # prepare the caffe2 backend for executing the model this converts the ONNX model into a
    # Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
    # availiable soon.
    prepared_backend = backend.prepare(model)

    # Input vector (batch size of 1)
    x = Variable(th.ones(1, 5), requires_grad=True)
    w = {model.graph.input[0].name: x.data.numpy()}

    # Run the Caffe2 net:
    c2_out = prepared_backend.run(w)[0]
    
    # extract the workspace and the model proto from the internal representation
    c2_workspace = prepared_backend.workspace
    c2_model = prepared_backend.predict_net

    # call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

    # Save the init_net and predict_net to a file that we will later use for running them on mobile
    with open('init_net.pb', "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open('predict_net.pb', "wb") as fopen:
        fopen.write(predict_net.SerializeToString())
    print('Done converting {}/foamNet-{:d}.nn'.format(netDir, netNum))


    # Let's run the mobile nets that we generated above so that caffe2 workspace is properly initialized
    workspace.RunNetOnce(init_net)
    workspace.RunNetOnce(predict_net)

    workspace.FeedBlob("input", x.data.numpy().astype(np.float32))
    workspace.RunNetOnce(predict_net)
    caffe2_out = workspace.FetchBlob("output")
    
    print("Python PyTorch Network")
    print(turb_nn.forward(x))
    print("Python Caffe2 Network")
    print(c2_out)
    print("Caffe2 Protobuf Network")
    print(caffe2_out)
