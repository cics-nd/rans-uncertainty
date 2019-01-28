"""
Executes training/testing/prediction of the neural network using SVGD. 
Structured after the SVGD work of Zhu and Zabaras.
https://github.com/cics-nd/cnn-surrogate
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: 
doi: 
github: https://github.com/cics-nd/rans-uncertainty
===
"""
from utils.log import Log
from utils.dataManager import DataManager
from utils.foamNetDataSet import FoamNetDataset, PredictDataset
from fluid.invariant import Invariant
from nn.turbnn import TurbNN

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import copy, os
import numpy as np
import torch as th
import nn.nnUtils as nnUtils
import torch.nn.functional as F

# Default tensor datatype
if th.cuda.is_available():
    dtype = th.cuda.DoubleTensor
else:
    dtype = th.DoubleTensor

class FoamSVGD():
    """
    Additional Useful References:
        - Zhu, Yinhao, and Nicholas Zabaras. "Bayesian deep convolutional 
        encoder–decoder networks for surrogate modeling and uncertainty 
        quantification." Journal of Computational Physics 366 (2018): 415-447.

        - Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.
    
    """
    def __init__(self, n_samples):
        self.lg = Log()
        self.lg.info('Constructing neural network...')

        self.n_samples = n_samples # Number of SVGD particles
        self.turb_nn = TurbNN(D_in=5, H=200, D_out=10).double() #Construct neural network        

        # Create n_samples SVGD particles
        # This is done by deep copying the invariant nn
        instances = []
        for i in range(n_samples):
            new_instance = copy.deepcopy(self.turb_nn)
            new_instance.reset_parameters() # Reset parameters to spread particles out
            instances.append(new_instance)

        self.models = th.nn.ModuleList(instances)
        del instances

        # Student's t-distribution: w ~ St(w | mu=0, lambda=shape/rate, nu=2*shape)
        # See PRML by Bishop Page 103
        self.prior_w_shape = 1.0
        self.prior_w_rate = 0.025
        # noise variance: beta ~ Gamma(beta | shape, rate)
        self.prior_beta_shape = 100
        self.prior_beta_rate = 2e-4

        # Now set up parameters for the noise hyper-prior
        beta_size = (self.n_samples, 1)
        log_beta = np.log(np.random.gamma(self.prior_beta_shape,
                        1. / self.prior_beta_rate, size=beta_size))
        # Log of the additive output-wise noise (beta)
        for i in range(n_samples):
            self.models[i].log_beta = Parameter(th.Tensor(log_beta[i]).type(dtype))

        # Network weights learning weight
        lr = 1e-5
        # Output-wise noise learning weight
        lr_noise=0.001

        # Construct individual optimizers and learning rate schedulers
        self.schedulers = []
        self.optimizers = []
        for i in range(n_samples):
            # Pre-pend output-wise noise to model parameter list
            parameters = [{'params': [self.models[i].log_beta], 'lr': lr_noise},
                  {'params': [p for n, p in self.models[i].named_parameters() if n!='log_beta']}]
            # ADAM optimizer (minor weight decay)
            optim = th.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
            # Decay learning weight on plateau, can adjust these parameters depending on data
            scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.75, patience=3,
                verbose=True, threshold=0.05, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-07)

            self.optimizers.append(optim)
            self.schedulers.append(scheduler)
        

    def forward(self, input, t_data):
        """
        Computes all the `n_samples` NN output
        Args: 
            input (Tensor): [nx5] tensor of input invariants
            t_data (Tensor): [nx10x3x3] tensor of linear independent tensore basis functions
        Return: out (Tensor): [nx3x3] tensor of predicted scaled anisotropic terms
        """
        out_size = (3, 3)
        output = Variable(th.Tensor(
            self.n_samples, input.size(0), *out_size).type(dtype))
        for i in range(self.n_samples):
            output[i] = self.models[i].forward(input)
            g_pred = self.models[i].forward(input)
            g_pred0 = th.unsqueeze(th.unsqueeze(g_pred, 2), 3)
            output[i] = th.sum(g_pred0*t_data,1)
        return output

    def compute_loss(self, output, target, index=None):
        """
        Computes the joint log probability ignoring constant terms
        See Eq. 23 in paper
        Args:
            output: B x oC x oH x oW
            target: B x oC x oH x oW
            index (None or int): model index, 0, 1, ..., n_samples.
        Returns:
            If index = None, return a list of joint probabilities, i.e. log
            unnormalized posterior. If index is assigned an int, return the
            joint probability for this model instance.
        """
        if index not in range(self.n_samples):
            ValueError("model index should be in [0, ..., {}], but got {}"
                       .format(self.n_samples, index))
        else:
            # Log Gaussian likelihood 
            # See Eq. 18-19 in paper
            log_likelihood = len(self.trainingLoader.dataset) / output.size(0) \
                                * (-0.5 * self.models[index].log_beta.exp()
                                * (target - output).pow(2).sum()
                                + 0.5 * target.numel()
                                * self.models[index].log_beta)

            # Log Gaussian weight prior
            # See Eq. 17 in paper
            prior_ws = Variable(th.Tensor([0]).type(dtype))
            for param in self.models[index].parameters():
                prior_ws += th.log1p(0.5 / self.prior_w_rate * param.pow(2)).sum()
            prior_ws *= -(self.prior_w_shape + 0.5)

            # Log Gamma Output-wise noise prior
            # See Eq. 20 in paper
            prior_log_beta = (self.prior_beta_shape * self.models[index].log_beta \
                        - self.models[index].log_beta.exp() * self.prior_beta_rate).sum()

            return log_likelihood + prior_ws + prior_log_beta, \
                   log_likelihood.data.item()

    def _squared_dist(self, X):
        """
        Computes the square distance for a set of vectors ||x1 - x2||.
        For two vectors ||q - p||^2 = ||p||^2 + ||q||^2 - 2p.q
        Args:
            X (Tensor): [sxp] Tensor we wish to compute the squared distance.
                              In SVGD, s is the number of particle (invar neural networks)
                              and p is the number of parameters (weights) assigned to that network
        Returns:
            (Tensor): [pxp] metrics of computed distances between each vector
        """
        # Compute dot product of each vector componation [PxP]
        XXT = th.mm(X, X.t())
        # Get ||p||^2 and ||q||^2 terms on diagonal [P vector]
        XTX = XXT.diag()
        # Return squared distance, note that PyTorch will broadcast the vectors
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)

    def _Kxx_dxKxx(self, X):
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in Eq. (8)
        of SVGD paper
        Args:
            X (Tensor): [sxp] Tensor we wish to compute the covariance matrix for.
                              In SVGD, s is the number of particle (invar neural networks)
                              and p is the number of parameters (weights) assigned to that network
        """
        squared_dist = self._squared_dist(X)

        triu_indices = squared_dist.triu(1).nonzero().transpose(0, 1)
        off_diag = squared_dist[triu_indices[0], triu_indices[1]]
        l_square = 0.5 * off_diag.median() / self.n_samples
        Kxx = th.exp(-0.5 / l_square * squared_dist)
        # Matrix form for the second term of optimal functional gradient in eqn (8) of SVGD paper
        # This line needs S x P memory
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx

    def train(self, n_epoch=250, gpu=True):
        """
        Training the neural network(s) using SVGD
        Args:
            n_epoch (int): Number of epochs to train for
            gpu (boolean): Whether or not to use a GPU (default is true)
        """
        if(not self.trainingLoader):
            self.lg.error('Training data loader not created! Stopping training')
            return
        
        if(gpu):
            self.lg.info('GPU network training enabled')
            #Transfer network and training data onto GPU
            for i in range(self.n_samples):
                    self.models[i].cuda()
        else:
            self.lg.info('CPU network training enabled')

        # Unique indexs of the symmetric deviatoric tensor
        indx = Variable(th.LongTensor([0,1,2,4,5,8]), requires_grad=False)
        if (gpu):
            indx = indx.cuda()

        self.lg.warning('Starting NN training with a experiment size of '+str(self.n_data))
        # store the joint probabilities
        for epoch in range(n_epoch):

            if (epoch + 1) % 1 == 0:
                self.lg.info('Running test samples...')
                self.test(epoch, gpu=gpu)

            training_loss = 0.
            training_MNLL = 0.
            training_MSE = 0.
            # Mini-batch the training set
            for batch_idx, (x_data, t_data, k_data, y_data) in enumerate(self.trainingLoader):
                x_data = Variable(x_data)
                t_data = Variable(t_data, requires_grad=False)
                k_data = Variable(k_data.unsqueeze(0).unsqueeze(0).expand(3,3,k_data.size()[0]).permute(2, 0, 1).type(dtype), requires_grad=False)
                y_data = Variable(y_data, requires_grad=False)
                if (gpu):
                    x_data = x_data.cuda()
                    t_data = t_data.cuda()
                    k_data = k_data.cuda()
                    y_data = y_data.cuda()

                # all gradients of log joint probability:
                # n_samples x num_parameters
                grad_log_joints = []
                # all model parameters (particles): n_samples x num_parameters
                theta = []
                b_pred_tensor = Variable(th.Tensor(
                    self.n_samples, x_data.size(0), 9).type(dtype), requires_grad=False)

                # Now iterate through each model
                for i in range(self.n_samples):
                    self.models[i].zero_grad()
                    # Predict mixing coefficients -> g_pred [Nx10]
                    g_pred = self.models[i].forward(x_data)
                    g_pred0 = th.unsqueeze(th.unsqueeze(g_pred, 2), 3)

		            # Multiply by respective tensor basis functions [Nx10x1x1]*[Nx10x3x3]
                    b_pred = th.sum(g_pred0[:,:,:,:]*t_data[:,:,:,:],1).view(-1,9)
                    b_pred_tensor[i] = b_pred

                    # For the loss since matrix is symmetric only select unique indices
                    # This is just a trick to speed things up and equally weight all parameters
                    b_pred = th.index_select(b_pred, 1, indx)
                    y_target = th.index_select(y_data, 1, indx)
                    loss, log_likelihood = self.compute_loss(b_pred, y_target, i)
                    # backward to compute gradients of log joint probabilities
                    loss.backward()
                    # monitoring purpose
                    training_loss += loss.data.item()
                    training_MNLL += log_likelihood

                    # Extract parameters and their gradients out from models
                    vec_param, vec_grad_log_joint = nnUtils.parameters_to_vector(
                        self.models[i].parameters(), both=True)

                    grad_log_joints.append(vec_grad_log_joint.unsqueeze(0))
                    theta.append(vec_param.unsqueeze(0))

                # calculating the kernel matrix and its gradients
                theta = th.cat(theta)
                Kxx, dxKxx = self._Kxx_dxKxx(theta)
                grad_log_joints = th.cat(grad_log_joints)
                grad_logp = th.mm(Kxx, grad_log_joints)
                # Negate the gradients here
                grad_theta = - (grad_logp + dxKxx) / self.n_samples

                # update param gradients
                for i in range(self.n_samples):
                    nnUtils.vector_to_parameters(grad_theta[i], self.models[i].parameters(), grad=True)
                    self.optimizers[i].step()
                del grad_theta

                # Scaled MSE
                training_MSE += ((y_data - b_pred_tensor.mean(0)) ** 2).sum(1).sum(0).data.item()
                # Mini-batch progress log
                if ((batch_idx+1) % 500 == 0):
                    self.lg.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNoise: {:.3f}'.format(
                    epoch, batch_idx * len(x_data), len(self.trainingLoader.dataset),
                    100. * batch_idx * len(x_data) / len(self.trainingLoader.dataset),
                    loss.data.item(), self.models[0].log_beta.data.item()))

            # Log training loss, MNLL, and mean mse
            ndata = len(self.trainingLoader.dataset)
            if (epoch + 1) % 1 == 0:
                self.lg.log("===> Epoch: {}, Current loss: {:.6f} Log Beta: {:.6f} Scaled-MSE: {:.6f}".format(
                    epoch + 1, training_loss, self.models[0].log_beta.data.item(), training_MSE/ndata))
                    
                self.lg.logLoss(epoch, training_loss/(ndata*self.n_samples), \
                    training_MNLL/(ndata*self.n_samples), training_MSE/ndata)
            
            # Update learning rate if needed
            for i in range(self.n_samples):
                self.schedulers[i].step(abs(training_loss))
    
    def test(self, epoch, gpu=True):
        """
        Tests the neural network(s) on validation/testing datasets
        Args:
            n_epoch (int): current epoch (just used for logging purposes)
            gpu (boolean): Whether or not to use a GPU (default is true)
        """
        try:
            self.testingLoaders
        except:
            self.lg.error('Testing data loader not created! Stopping testing')
            return
                  
        # Unique indexs of the symmetric deviatoric tensor
        indx = Variable(th.LongTensor([0,1,2,4,5,8]), requires_grad=False)
        if (gpu):
            indx = indx.cuda()

        # Switch models to eval state (shuts off dropout)
        for i in range(self.n_samples):
            self.models[i].eval()

        # Mini-batch the training set
        flow_mspe = np.zeros(len(self.testingLoaders))
        flow_mnll = np.zeros(len(self.testingLoaders))
        for n, testingLoader in enumerate(self.testingLoaders):
            net_mse = 0
            testing_MNLL = 0
            for batch_idx, (x_data, t_data, k_data, y_data) in enumerate(testingLoader):
                # Make mini-batch data variables
                x_data = Variable(x_data)
                t_data = Variable(t_data, requires_grad=False)
                k_data = Variable(k_data.unsqueeze(0).unsqueeze(0).expand(3,3,k_data.size()[0]).permute(2, 0, 1).type(dtype), requires_grad=False)
                y_data = Variable(y_data, requires_grad=False)
                if (gpu):
                    x_data = x_data.cuda()
                    t_data = t_data.cuda()
                    k_data = k_data.cuda()
                    y_data = y_data.cuda()

                b_pred = Variable(th.Tensor(
                    self.n_samples, x_data.size(0), 9).type(dtype))

                # Now iterate through each SVGD particle (invariant nn)
                for i in range(self.n_samples):
                    self.models[i].zero_grad()
                    g_pred = self.models[i].forward(x_data)
                    g_pred0 = th.unsqueeze(th.unsqueeze(g_pred, 2), 3)
                    b_pred[i] = th.sum(g_pred0[:,:,:,:]*t_data[:,:,:,:],1).view(-1,9)
                    # For the loss since matrix is symmetric only select unique indices
                    loss, log_likelihood = self.compute_loss(th.index_select(b_pred[i],1,  indx), th.index_select(y_data, 1, indx), i)
                    testing_MNLL += log_likelihood

                # MSE of unscaled anisotropic tensor a = k*b
                # This is used as a metric since a is used in the RANS equations
                # Thus measuring the MSE of b would be bias
                mse = (th.index_select(k_data.view(-1,9)*(y_data - b_pred.mean(0)), 1, indx) ** 2).sum()
                net_mse += mse.data.item()

            # Total amount of testing data
            ndata = len(self.testingLoaders[n].dataset)
            flow_mspe[n] = net_mse/(ndata)
            flow_mnll[n] = testing_MNLL/(ndata*self.n_samples)

        # Log testing results
        self.lg.log("===> Current Total Validation Loss: {}, Validation MSE: {}" \
                .format(flow_mnll.sum(0)/ndata, flow_mspe.sum(0)))
        self.lg.logTest(epoch, flow_mnll, flow_mspe)

    def predict(self, ransS, ransR, n_mb=200, gpu=True):
        """
        Method for obtaining mean and variance predictions for a given flow [depreciated]
        (Un-used during training/ testing but left in for reference)
        Args:
            ransS (Tensor): [nCellx3x3] Baseline RANS rate-of-strain tensor
            ransS (Tensor): [nCellx3x3] Baseline RANS rotation tensor
        """
        # Get invariants and tensor functions for the provided field
        fl = Invariant()
        x_data = fl.getInvariants(ransS, ransR)
        t_data  = fl.getTensorFunctions(ransS, ransR)
        # Create data loader
        predictionLoader = th.utils.data.DataLoader(PredictDataset(x_data, t_data), batch_size=n_mb, shuffle=False)

        # Switch models to eval state (shuts off dropout)
        for i in range(self.n_samples):
            self.models[i].eval()
        
        if(gpu):
            self.lg.info('GPU network training enabled')
            #Transfer network and training data onto GPU
            for i in range(self.n_samples):
                    self.models[i].cuda()
        else:
            self.lg.info('CPU network training enabled')

        # Note we store the mean and variance fields on the CPU since the fluid field may be very larges
        # Mini-batches are then brought onto the GPU from forward passes.
        out_size = (3,3)
        b_mean = th.Tensor(x_data.size(0), *out_size).type(th.DoubleTensor)
        b_var = th.Tensor(x_data.size(0), *out_size).type(th.DoubleTensor)
        b_index = 0
        # Mini-batch the pred
        for batch_idx, (x_data, t_data) in enumerate(predictionLoader):
            x_data = Variable(x_data, requires_grad=False)
            t_data = Variable(t_data, requires_grad=False)
            if (gpu):
                x_data = x_data.cuda()
                t_data = t_data.cuda()

            b_pred = Variable(th.Tensor(
                self.n_samples, x_data.size(0), *out_size).type(dtype))

            # Now iterate through each model
            for i in range(self.n_samples):
                self.models[i].zero_grad()
                g_pred = self.models[i].forward(x_data)
                g_pred0 = th.unsqueeze(th.unsqueeze(g_pred, 2), 3)
                b_pred[i] = th.sum(g_pred0[:,:,:,:]*t_data[:,:,:,:],1)
                
            # Compute expectation (Eq. 30 in paper)
            b_mean[b_index:b_index+x_data.size(0)] = b_pred.mean(0).data.cpu()
            # Compute variance of each output  (Eq. 31 in paper)
            b_eyyt = (b_pred.data ** 2).mean(0)
            b_eyeyt = b_pred.mean(0).data ** 2
            beta_inv = th.cat([(-self.models[i].log_beta.data).exp().unsqueeze(0) for i in range(self.n_samples)])
            y_noise_var = beta_inv.mean(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            b_var[b_index:b_index+x_data.size(0)] = b_eyyt.cpu() - b_eyeyt.cpu() + y_noise_var.cpu()
            # Step index
            b_index += x_data.size(0)

            if ((batch_idx+1) % 50 == 0):
                self.lg.log('Mini-batching predictive field. [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(x_data), len(predictionLoader.dataset),
                    100. * batch_idx * len(x_data) / len(predictionLoader.dataset)))

        return b_mean, b_var

    def getTrainingPoints(self, dataManager, n_data = 5000, n_mb = 250, n_valid=0):
        """ 
        Used for creating training and also validation datasets
        Args: dataManager: dataManager object for loading openFoam data
              n_data: total number of training/ validation data to use
              n_mb: size of minibatch
              n_valid: number of points to use as validation for current dataset
        """
        self.lg.info('Creating training data-set')
        self.n_data = n_data
        self.n_mb = n_mb

        # Get the set of training points from the data manager
        x0, t0, k0, y0 = dataManager.getDataPoints(self, n_data)

        # Randomly permute the read data 
        perm0 = th.randperm(n_data)
        x0 = x0[perm0]
        t0 = t0[perm0]
        k0 = k0[perm0]
        y0 = y0[perm0]

        # Set training and test data
        x_train = x0[n_valid:]
        t_train = t0[n_valid:]
        k_train = k0[n_valid:]
        y_train = y0[n_valid:]
    
        # Create data sets
        self.trainingDataSet = FoamNetDataset(x_train, t_train, k_train, y_train)
        # Now create loaders (set mini-batch size and also turn on shuffle)
        self.trainingLoader = th.utils.data.DataLoader(self.trainingDataSet, batch_size=n_mb, shuffle=True)

        # If we wish to have validation dataset create testing loaders
        # This ensures validation data is never used during training
        if(n_valid > 0):
            x_test = x0[0:n_valid]
            t_test = t0[0:n_valid]
            k_test = k0[0:n_valid]
            y_test = y0[0:n_valid]

            self.testingDataSet = FoamNetDataset(x_test, t_test, k_test, y_test)
            # Check to see if testing loaders created
            try:
                self.testingLoaders
            except AttributeError:
                self.lg.info('Creating testing loader list...')
                self.testingLoaders = []
            self.testingLoaders.append(th.utils.data.DataLoader(self.testingDataSet, batch_size=n_mb, shuffle=True))

    def getTestingPoints(self, dataManager, n_data=500, n_mb=250):
        """ 
        Creates independent testing/validation datasets
        Args: dataManager: dataManager object for loading openFoam data
              n_data: total number of training/ validation data to use
              n_mb: size of minibatch
              n_valid: number of points to use as validation
        """
        self.lg.info('Creating testing data-set')
        # Check to see if testing loaders created
        try:
            self.testingLoaders
        except AttributeError:
            self.lg.info('Creating testing loader list...')
            self.testingLoaders = []

        # Get the set of testing points from the data manager
        # Note turn mask -> True so these points are not used during training
        x_test, t_test, k_test, y_test = dataManager.getDataPoints(self, n_data, mask=True)

        # Create data sets
        self.testingDataSet = FoamNetDataset(x_test, t_test, k_test, y_test)
        # Now create test loaders
        self.testingLoaders.append(th.utils.data.DataLoader(self.testingDataSet, batch_size=n_mb, shuffle=True))

    def saveNeuralNet(self, filename):
        """
        Save the current neural network state
        Args:
            filename (string): name of the file to save the neural network in
        """
        self.lg.log("Saving neural networks to: ./torchNets/{}".format(filename))
        if not os.path.exists("torchNets"):
            os.makedirs("torchNets")
        # Iterate through each model
        for i in range(self.n_samples):
            th.save(self.models[i].state_dict(), "./torchNets/{}-{}.nn".format(filename,i))
        
        
    def loadNeuralNet(self, filename):
        """
        Load the current neural network state
        Args:
            filename (string): name of the file to save the neural network in
        """
        for i in range(self.n_samples):
            self.models[i].load_state_dict(th.load("{}-{}.nn".format(filename,i), map_location=lambda storage, loc: storage))

    def getTurbNet(self):
        """
        Accessor to get the neural net object
        Returns:
            TurbNN (th.nn.Module): PyTorch neural network object
        """
        return self.turb_nn

    def btrace(self, a, gpu=True):
        """
        Return the batch trace of tensor a
        """
        if(gpu):
            eye = th.eye(3).unsqueeze(0).repeat(a.size()[0],1,1).cuda()
        else:
            eye = th.eye(3).unsqueeze(0).repeat(a.size()[0],1,1)
        return th.sum(th.sum(th.bmm(a, eye), 2), 1)
    
    def btraceVariable(self, a, gpu=True):
        """
        Return the batch trace of tensor a
        """
        if(gpu):
            eye = Variable(th.eye(3).unsqueeze(0).repeat(a.size()[0],1,1), requires_grad=False).cuda()
        else:
            eye = Variable(th.eye(3).unsqueeze(0).repeat(a.size()[0],1,1), requires_grad=False)
        return th.sum(th.sum(th.bmm(a, eye), 2), 1)
