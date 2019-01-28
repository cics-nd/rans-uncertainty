//First get the gradient of the velocity field
//fvc = finite volume calculus
//fvm = finite volume method (matrix)
/*volTensorField gradU = fvc::grad(U)

//Get the specific gradients of each
volScalarField	XX=gradU.component(0);
volScalarField	YY=gradU.component(4);
volScalarField	ZZ=gradU.component(8);

volScalarField	XY=gradU.component(1);
volScalarField	YZ=gradU.component(5);
volScalarField	ZX=gradU.component(6);

volScalarField	YX=gradU.component(3);
volScalarField	ZY=gradU.component(7);
volScalarField	XZ=gradU.component(2);*/

//First get the gradient of the velocity field
tmp<volTensorField> tgradU(fvc::grad(U));
UGrad = tgradU();

//Now to get the rotational field
//skew return full Tensor Feild
volTensorField G(skew(UGrad));
volScalarField ke = (turbulence->k())/(turbulence->epsilon());
R = ke*G;

//Now to get the symmetric field
//skew return symTensor Feild
tmp<volSymmTensorField> sym0(symm(UGrad));
volSymmTensorField sym = ke*sym0();
//Convert symmetric matrix to full matrix
S.replace(0, sym.component(symmTensor::XX));
S.replace(1, sym.component(symmTensor::XY));
S.replace(2, sym.component(symmTensor::XZ));
S.replace(3, sym.component(symmTensor::XY));
S.replace(4, sym.component(symmTensor::YY));
S.replace(5, sym.component(symmTensor::YZ));
S.replace(6, sym.component(symmTensor::XZ));
S.replace(7, sym.component(symmTensor::YZ));
S.replace(8, sym.component(symmTensor::ZZ));









