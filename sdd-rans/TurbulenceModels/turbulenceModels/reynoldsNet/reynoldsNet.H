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
Class
    caffe2

Group
    n/a

Description
    Loads neural net for deviatoric Reynolds stress calculation

SourceFiles
    reynoldsNet.C
\*---------------------------------------------------------------------------*/
#ifndef reynoldsNet_H__
#define reynoldsNet_H__

#pragma push_macro("TypeName")
#undef TypeName

#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/init.h"

#pragma pop_macro("TypeName")

namespace caffe2{

class reynoldsNet {

  protected:
    // Protected data

  public:

    //- Constructors 
    reynoldsNet();

    reynoldsNet(std::string init_dir, std::string pred_dir);

    //- Destructor
    virtual ~reynoldsNet(){}

    //- Member Functions
    void readNetFromFile() const;
    
    std::vector<float> forward(std::vector<float> inputdata) const;
};
}  // namespace caffe2 

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
