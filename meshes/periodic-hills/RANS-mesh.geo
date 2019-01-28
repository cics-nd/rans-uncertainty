// Flow over Periodic Hills
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: 
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: http://www.doi.org/10.1016/S0142-727X(02)00222-9
// url: https://turbmodels.larc.nasa.gov/Other_LES_Data/2dhill_periodic.html

// Input Variables
//Flow geometry
h0 = 1;
val_width = 9*h0; //Width between peaks
chan_h = 3.035*h0; // Channel height
lz = 4.5*h0; // Span length

// Mesh Resolution
n_hill = 25;
n_val = 50; //# on bump spline
n_vert = 50; //Vertical
n_z = 50;

//Mesh Progression
p_vert = 1.05; //Vertical progression

SetFactory("OpenCASCADE");

// Hill Based off the Normalized Geometry of Almeida et al. 1993
// https://turbmodels.larc.nasa.gov/Other_LES_Data/2dhill_periodic.html
// Inlet side
// 0 to 9
For i In {0 : n_hill}
  x = 9*i/(n_hill);
  pList[i] = newp;
  h = 2.800000000000*10^(1) + 6.775070969851*10^(-3)*x^2 - 2.124527775800*10^(-3)*x^3;
  If(h > 28)
    h = 28;
  EndIf
  Point(pList[i]) = {h0*x/28, h0*h/28, 0.0, 1.0};
EndFor
// 9 to 14
For i In {n_hill+1 : 2*n_hill}
  x = 9 + 5*(i-n_hill)/(n_hill);
  pList[i] = newp;
  h = 2.507355893131*10^(1) + 9.754803562315*10^(-1)*x - 1.016116352781*10^(-1)*x^2  + 1.889794677828*10^(-3)*x^3;
  Point(pList[i]) = {h0*x/28, h0*h/28, 0.0, 1.0};
EndFor
// 14 to 20
For i In {2*n_hill+1 : 3*n_hill}
  x = 14 + 6*(i-2*n_hill)/(n_hill);
  pList[i] = newp;
  h = 2.579601052357*10^(1) + 8.206693007457*10^(-1)*x - 9.055370274339*10^(-2)*x^2  + 1.626510569859*10^(-3)*x^3;
  Point(pList[i]) = {h0*x/28, h0*h/28, 0.0, 1.0};
EndFor
// 20 to 30
For i In {3*n_hill+1 : 4*n_hill}
  x = 20 + 10*(i-3*n_hill)/(n_hill);
  pList[i] = newp;
  h = 4.046435022819*10^(1) - 1.379581654948*10^(0)*x + 1.945884504128*10^(-2)*x^2 - 2.070318932190*10^(-4)*x^3;
  Point(pList[i]) = {h0*x/28, h0*h/28, 0.0, 1.0};
EndFor
// 30 to 40
For i In {4*n_hill+1 : 5*n_hill}
  x = 30 + 10*(i-4*n_hill)/(n_hill);
  pList[i] = newp;
  h = 1.792461334664*10^(1) + 8.743920332081*10^(-1)*x - 5.567361123058*10^(-2)*x^2 + 6.277731764683*10^(-4)*x^3;
  Point(pList[i]) = {h0*x/28, h0*h/28, 0.0, 1.0};
EndFor
// 40 to 54
For i In {5*n_hill+1 : 6*n_hill}
  x = 40 + 14*(i-5*n_hill)/(n_hill);
  pList[i] = newp;
  h = 5.639011190988*10^(1) - 2.010520359035*10^(0)*x + 1.644919857549*10^(-2)*x^2 + 2.674976141766*10^(-5)*x^3;
  If(h < 0)
    h = 0;
  EndIf
  Point(pList[i]) = {h0*x/28, h0*h/28, 0.0, 1.0};
EndFor
// Spline the profile
Spline(1) = pList[]; Transfinite Line {1} = n_hill Using Progression 1;

// Outlet side
For i In {0 : 6*n_hill}
  p = Point{pList[i]};
  pList2[i] = newp;
  Point(pList2[i]) = {val_width - p[0], p[1], p[2], 1};
EndFor
// Spline the profile
Spline(2) = pList2[]; Transfinite Line {2} = n_hill Using Progression 1;

//Form rest of channel
p0 = pList2[6*n_hill]+1;
pv1 = Point{pList[6*n_hill]};
pv2 = Point{pList2[6*n_hill]};
Point(p0) = {0, chan_h, 0 ,1};
Point(p0 + 1) = {pv1[0], chan_h, 0 ,1};
Point(p0 + 2) = {pv2[0], chan_h, 0 ,1};
Point(p0 + 3) = {9, chan_h, 0 ,1};

// Line for valley
Line(3) = {pList[6*n_hill], pList2[6*n_hill]}; Transfinite Line {3} = n_val Using Progression 1;
//Top of channel
Line(4) = {p0, p0+1}; Transfinite Line {4} = n_hill Using Progression 1;
Line(5) = {p0+1, p0+2}; Transfinite Line {5} = n_val Using Progression 1;
Line(6) = {p0+2, p0+3}; Transfinite Line {6} = n_hill Using Progression 1;
// Vertical Lines
Line(7) = {1, p0}; Transfinite Line {7} = n_vert Using Progression p_vert;
Line(8) = {pList[6*n_hill], p0+1}; Transfinite Line {8} = n_vert Using Progression p_vert;
Line(9) = {pList2[6*n_hill], p0+2}; Transfinite Line {9} = n_vert Using Progression p_vert;
Line(10) = {pList2[0], p0+3}; Transfinite Line {10} = n_vert Using Progression p_vert;//+

//Set up planes
Line Loop(1) = {1, 8, -4, -7};
Plane Surface(1) = {1};
Line Loop(2) = {3, 9, -5, -8};
Plane Surface(2) = {2};
Line Loop(3) = {2, 9, 6, -10};
Plane Surface(3) = {3};
//+
Transfinite Surface {1, 2, 3};
Recombine Surface {1, 2, 3};

// Extrude out
Extrude {0, 0, lz} {
  Surface{1};
  Surface{2}; 
  Surface{3}; 
  Layers{n_z}; 
  Recombine;
}

// Define physical surfaces for openFOAM
Physical Surface("TopWall") = {6, 11, 14};
Physical Surface("BottomWall") = {4, 9, 13};
Physical Surface("SideWall1") = {8, 12, 16};
Physical Surface("SideWall2") = {1, 2, 3};
Physical Surface("Inlet") = {7};
Physical Surface("Outlet") = {15};

// Now physical volume
Physical Volume(7) = {1, 2, 3};
