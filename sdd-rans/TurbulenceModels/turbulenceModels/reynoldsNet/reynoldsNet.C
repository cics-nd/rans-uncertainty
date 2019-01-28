/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.
\*---------------------------------------------------------------------------*/
#include "reynoldsNet.H"

// Global variables
std::string init_net_dir = "test";
std::string pred_net_dir = "";
// Define NN model
caffe2::NetDef initNet, predictNet;
// Define workspace
caffe2::Workspace workSpace;

caffe2::reynoldsNet::reynoldsNet() {
  init_net_dir = "init_net.pb";
  pred_net_dir = "predict_net.pb";
  caffe2::GlobalInit();
}

caffe2::reynoldsNet::reynoldsNet(std::string init_dir, std::string pred_dir) {
  init_net_dir = init_dir;
  pred_net_dir = pred_dir;
  caffe2::GlobalInit();
}

void caffe2::reynoldsNet::readNetFromFile() const {
  //Check to see if the protobuf files exist...
  if (!std::ifstream(init_net_dir).good()) {
    std::cerr << "error: init net model file missing: "
              << init_net_dir
              << std::endl;
    return;
  }
  if (!std::ifstream(pred_net_dir).good()) {
    std::cerr << "error: predict net model file missing: "
              << pred_net_dir
              << std::endl;
    return;
  }

  //Open protobuffer files and read NN
  std::cout << "Reading in protobuf files" << std::endl;
  CAFFE_ENFORCE(ReadProtoFromFile(init_net_dir, &initNet));
  CAFFE_ENFORCE(ReadProtoFromFile(pred_net_dir, &predictNet));
  std::cout << "Reading complete..." << std::endl;

  std::cout << "Setting NN device to CPU" << std::endl;
  predictNet.mutable_device_option()->set_device_type(DeviceTypeProto::PROTO_CPU);
  initNet.mutable_device_option()->set_device_type(DeviceTypeProto::PROTO_CPU);
  for(int i = 0; i < predictNet.op_size(); ++i){
      predictNet.mutable_op(i)->mutable_device_option()->set_device_type(DeviceTypeProto::PROTO_CPU);
  }
  for(int i = 0; i < initNet.op_size(); ++i){
      initNet.mutable_op(i)->mutable_device_option()->set_device_type(DeviceTypeProto::PROTO_CPU);
  }

  std::cout << "Setting NN device to CPU" << std::endl;
  CAFFE_ENFORCE(workSpace.RunNetOnce(initNet));
  CAFFE_ENFORCE(workSpace.CreateNet(predictNet));
}


std::vector<float> caffe2::reynoldsNet::forward(std::vector<float> inputdata) const {

  // Now create input tensor (only supports 1 size for now)
  CPUContext ctx;
  auto inputTensor = Tensor({1, 5}, inputdata, &ctx);

  // Get data input blob out of network object
  // NOTE!!! The names of the input and output blog are defined
  // when converting from PyTorch to Caffe2. MUST be consistent
  // or bad things will happen. See pytorchToCaffe2.py onnxExport function
  auto data = workSpace.GetBlob("input")->GetMutable<TensorCPU>();
  // Copy Tensor data onto blob in NN
  data->CopyFrom(inputTensor);

  // Forward
  workSpace.RunNet(predictNet.name());

  // Get blob of NN output
  auto outBlob = workSpace.GetBlob("output")->Get<TensorCPU>();

  std::vector<float> out_vect(outBlob.data<float>(), outBlob.data<float>() + outBlob.size());

  return out_vect;
}