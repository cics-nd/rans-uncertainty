// First get the gradient of the velocity field
tmp<volTensorField> tgradU(fvc::grad(U));
UGrad = tgradU();
// The gradient of the pressure and TKE are not used
// But are calculated for potential use as a flow feature
// Gradient of pressure
pGrad = fvc::grad(p);
// TKE gradient
kGrad = fvc::grad(turbulence->k());

// Now to get the rotational field
// skew return full Tensor Feild
// Remember that both skew and symm already account for the 1/2 in Pope's paper
volTensorField G(skew(UGrad));
volScalarField ke = (turbulence->k())/(turbulence->epsilon());
R = ke*G; // Scale

// Now to get the symmetric field
// symm return symTensor Feild
tmp<volSymmTensorField> sym0(symm(UGrad));
volSymmTensorField sym = ke*sym0(); // Scale

// Convert symmetric matrix to full matrix
S.replace(0, sym.component(symmTensor::XX));
S.replace(1, sym.component(symmTensor::XY));
S.replace(2, sym.component(symmTensor::XZ));
S.replace(3, sym.component(symmTensor::XY));
S.replace(4, sym.component(symmTensor::YY));
S.replace(5, sym.component(symmTensor::YZ));
S.replace(6, sym.component(symmTensor::XZ));
S.replace(7, sym.component(symmTensor::YZ));
S.replace(8, sym.component(symmTensor::ZZ));
