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

#include "linearViscousStress.H"
#include "fvc.H"
#include "fvcSmooth.H"
#include "fvm.H"
#include "wallDist.H"
#include <typeinfo>
#include <math.h>
#include <cmath>
#include <algorithm>

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Foam::linearViscousStress<BasicTurbulenceModel>::linearViscousStress
(
    const word& modelName,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    BasicTurbulenceModel
    (
        modelName,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    ident_
    (
        IOobject
        (
            "ident_",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        tensor(1, 0, 0, 0, 1, 0, 0, 0, 1)
    ),
    b0
    (
        IOobject
        (
            "b0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("b0", dimless, tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    a_dd
    (
        IOobject
        (
            "a_dd",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_dd", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    a_0
    (
        IOobject
        (
            "a_0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_0", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    a_star
    (
        IOobject
        (
            "a_star",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_star", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),

    gamma_mix
    (
        IOobject
        (
            "gamma_mix",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("gamma_mix", dimless, 0.0)
    )
{
    //Read in neural network from file
    rn.readNetFromFile();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool Foam::linearViscousStress<BasicTurbulenceModel>::read()
{
    return BasicTurbulenceModel::read();
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::devRhoReff() const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("devRhoReff", this->U_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            (-(this->alpha_*this->rho_*this->nuEff()))
           *dev(twoSymm(fvc::grad(this->U_)))
        )
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U
) const
{
    return
    (
      - fvc::div((this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(U))))
      - fvm::laplacian(this->alpha_*this->rho_*this->nuEff(), U)
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
    return
    (
      - fvc::div((this->alpha_*rho*this->nuEff())*dev2(T(fvc::grad(U))))
      - fvm::laplacian(this->alpha_*rho*this->nuEff(), U)
    );
}

template<class BasicTurbulenceModel> 
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U,
    volTensorField& S,
    volTensorField& R
) const
{
    //size of input and output vectors from NN
    size_t sizei = 5;
    size_t sizet = 10;

    label timeIndex = this->mesh_.time().timeIndex();
    label startTime = this->runTime_.startTimeIndex();
    // Only do a forward pass of the network on the FIRST timestep
    if(timeIndex - 1 == startTime){
        Info << "[WARNING] Entered Data-driven turbulence predictor..." << endl;
        Info << "Hopefully theres an init_net.pb and predict_net.pb for me to run." << endl;
        Info << "Calculating Flow Field Invariants" << endl;

        //First get the invariant inputs to the NN
        volTensorField s2 (S&S);
        volTensorField r2 (R&R);
        volTensorField s3 (s2&S);
        volTensorField r2s (r2&S);
        volTensorField r2s2 (r2&s2);

        //Get the invariant inputs to the NN
        std::vector<volScalarField*> invar(sizei);
        std::vector<volScalarField*> invar0(sizei);
        invar[0] = new volScalarField(tr(s2));
        invar[1] = new volScalarField(tr(r2));
        invar[2] = new volScalarField(tr(s3));
        invar[3] = new volScalarField(tr(r2s));
        invar[4] = new volScalarField(tr(r2s2));

        invar0[0] = new volScalarField(*invar[0]);
        invar0[1] = new volScalarField(*invar[1]);
        invar0[2] = new volScalarField(*invar[2]);
        invar0[3] = new volScalarField(*invar[3]);
        invar0[4] = new volScalarField(*invar[4]);

        // Normalize the invariants by the sigmoid
        forAll(this->mesh_.C(), cell){            
            (*invar0[0])[cell] = Foam::sign((*invar[0])[cell])*(1 - std::exp(-abs((float)(*invar[0])[cell])))/ \
            (1 + std::exp(-abs((float)(*invar[0])[cell])));
            (*invar0[1])[cell] = Foam::sign((*invar[1])[cell])*(1 - std::exp(-abs((float)(*invar[1])[cell])))/ \
            (1 + std::exp(-abs((float)(*invar[1])[cell])));
            (*invar0[2])[cell] = Foam::sign((*invar[2])[cell])*(1 - std::exp(-abs((float)(*invar[2])[cell])))/ \
            (1 + std::exp(-abs((float)(*invar[2])[cell])));
            (*invar0[3])[cell] = Foam::sign((*invar[3])[cell])*(1 - std::exp(-abs((float)(*invar[3])[cell])))/ \
            (1 + std::exp(-abs((float)(*invar[3])[cell])));
            (*invar0[4])[cell] = Foam::sign((*invar[4])[cell])*(1 - std::exp(-abs((float)(*invar[4])[cell])))/ \
            (1 + std::exp(-abs((float)(*invar[4])[cell])));
        }
        
        //calculate tensor functions
        std::vector<volTensorField*> tensorf(sizet);
        tensorf[0] = new volTensorField(S);
        tensorf[1] = new volTensorField((S&R) - (R&S));
        tensorf[2] = new volTensorField(s2 - 1./3*ident_*(*invar[0]));
        tensorf[3] = new volTensorField(r2 - 1./3*ident_*(*invar[1]));
        tensorf[4] = new volTensorField((R&s2) - (s2&R));
        tensorf[5] = new volTensorField((r2&S) + (S&r2) - 2./3*ident_*tr(S&r2));
        tensorf[6] = new volTensorField(((R&S)&r2) - ((r2&S)&R));
        tensorf[7] = new volTensorField(((S&R)&s2) - ((s2&R)&S));
        tensorf[8] = new volTensorField((r2&s2) + (s2&r2) - 2./3*ident_*tr(s2&r2));
        tensorf[9] = new volTensorField(((R&s2)&r2) - ((r2&s2)&R));

        //Normalize the tensors by L2 norm
        forAll(this->mesh_.C(), cell){
            for ( int i = 0; i < 10; i++){
                float l2norm = 0;
                Foam::Tensor<double> t_array = (*tensorf[i])[cell];
                
                l2norm += pow(t_array.xx(), 2);
                l2norm += pow(t_array.xy(), 2);
                l2norm += pow(t_array.xz(), 2);
                l2norm += pow(t_array.yx(), 2);
                l2norm += pow(t_array.yy(), 2);
                l2norm += pow(t_array.yz(), 2);
                l2norm += pow(t_array.zx(), 2);
                l2norm += pow(t_array.zy(), 2);
                l2norm += pow(t_array.zz(), 2);
                l2norm = sqrt(l2norm);

                t_array.replace(0, t_array.xx()/l2norm);
                t_array.replace(1, t_array.xy()/l2norm);
                t_array.replace(2, t_array.xz()/l2norm);
                t_array.replace(3, t_array.yx()/l2norm);
                t_array.replace(4, t_array.yy()/l2norm);
                t_array.replace(5, t_array.yz()/l2norm);
                t_array.replace(6, t_array.zx()/l2norm);
                t_array.replace(7, t_array.zy()/l2norm);
                t_array.replace(8, t_array.zz()/l2norm);

                (*tensorf[i])[cell] = t_array;
            }
        }

        // Now get the data-driven prediction
        Info << "Executing forward pass of the neural network" << endl;
        // Iterate over cells
        forAll(this->mesh_.C(), cell){
            // Define the test inputs
            std::vector<float> inputdata({ (float)(*invar0[0])[cell], (float)(*invar0[1])[cell], (float)(*invar0[2])[cell], \
            (float)(*invar0[3])[cell], (float)(*invar0[4])[cell] });

            // Forward pass of the neural network
            this->outdata = rn.forward(inputdata);

            for(int i = 0; i < outdata.size(); i++){
		
                this->b0[cell] = this->b0[cell] + outdata[i]*(*tensorf[i])[cell];
            }
        }

        // Now calculate the Reynolds stress field
        this->a_dd = this->alpha_*this->rho_*this->k()*(this->b0);

        // Mixing field
	    // Defined from https://doi.org/10.2514/1.2094
        float lh = 1.0; // Characteristic length
        volScalarField d = wallDist(this->mesh_).y()/dimensionedScalar("lh", dimensionSet(0, 1, 0, 0, 0, 0, 0), lh);
        volScalarField lambdb0 = sqrt(this->k()/this->epsilon());
        lambdb0.dimensions().reset(dimless);

        dimensionedScalar alpha1 = dimensionedScalar("alpha1", dimless, 1.0);
        dimensionedScalar alpha2 = dimensionedScalar("alpha2", dimless, 0.05);
        dimensionedScalar small0 = dimensionedScalar("small", dimless, 1e-10);

        this->gamma_mix = min(alpha1, pow(tanh((d/(alpha2*lambdb0 + small0))), 2));

        // Assuming the flow is incompressible, otherwise add dev()
        // K-e turbulence model
        // a = -nu_t*(du/dv + dv/du)
        this->a_0 = -1*this->alpha_*this->rho_*(this->nut()*(ident_ & twoSymm(fvc::grad(U))));    
        Info << "Job Done" << '\n' << endl;
    }
    // Not mix orginal and data-driven prediction
    this->a_star = gamma_mix*a_dd + (1-gamma_mix)*a_0;

    // Still returns fvm due to the viscous component
    // No TKE term because this is absorbed into a modified pressure
    return 
    (
        - fvm::laplacian(this->alpha_*this->rho_*this->nu(), U)
        - fvc::div((this->alpha_*this->rho_*this->nu())*dev2(T(fvc::grad(U))) - this->a_star)
    );
}

template<class BasicTurbulenceModel>
void Foam::linearViscousStress<BasicTurbulenceModel>::correct()
{
    BasicTurbulenceModel::correct();
}


// ************************************************************************* //