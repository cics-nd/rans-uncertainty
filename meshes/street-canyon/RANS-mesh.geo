// Street Canyon Mesh
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: https://doi.org/10.1016/j.jcp.2019.01.021
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: http://www.doi.org/10.1016/j.jweia.2010.12.005

// Input Variables
h = 1; // Characteristic length
dimx = 18*h;
dimy = 6*h;
dimz = 12*h;
gs = 20;

// Box point (-x, -y, -x)
boxSize = {h,h,2*h};
boxPos1 = {4*h,0,(dimz-boxSize[0])/2};
boxPos2 = {6*h,0,(dimz-boxSize[0])/2};

SetFactory("OpenCASCADE");
// Set Front Lower Box

Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 0, boxPos1[2], 1.0};
Point(3) = {0, 0, boxPos1[2]+boxSize[2], 1.0};
Point(4) = {0, 0, dimz, 1.0};
//+
Point(5) = {boxPos1[0], 0, 0, 1.0};
Point(6) = {boxPos1[0], 0, boxPos1[2], 1.0};
Point(7) = {boxPos1[0], 0, boxPos1[2]+boxSize[2], 1.0};
Point(8) = {boxPos1[0], 0, dimz, 1.0};
//+
Point(9) = {boxPos1[0]+boxSize[0], 0, 0, 1.0};
Point(10) = {boxPos1[0]+boxSize[0], 0, boxPos1[2], 1.0};
Point(11) = {boxPos1[0]+boxSize[0], 0, boxPos1[2]+boxSize[2], 1.0};
Point(12) = {boxPos1[0]+boxSize[0], 0, dimz, 1.0};
//+
Point(13) = {boxPos2[0], 0, 0, 1.0};
Point(14) = {boxPos2[0], 0, boxPos2[2], 1.0};
Point(15) = {boxPos2[0], 0, boxPos2[2]+boxSize[2], 1.0};
Point(16) = {boxPos2[0], 0, dimz, 1.0};
//+
Point(17) = {boxPos2[0]+boxSize[0], 0, 0, 1.0};
Point(18) = {boxPos2[0]+boxSize[0], 0, boxPos2[2], 1.0};
Point(19) = {boxPos2[0]+boxSize[0], 0, boxPos2[2]+boxSize[2], 1.0};
Point(20) = {boxPos2[0]+boxSize[0], 0, dimz, 1.0};
//+
Point(21) = {dimx, 0, 0, 1.0};
Point(22) = {dimx, 0, boxPos2[2], 1.0};
Point(23) = {dimx, 0, boxPos2[2]+boxSize[2], 1.0};
Point(24) = {dimx, 0, dimz, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {1, 5};
//+
Line(5) = {2, 6};
//+
Line(6) = {3, 7};
//+
Line(7) = {4, 8};
//+
Line(8) = {5, 6};
//+
Line(9) = {6, 7};
//+
Line(10) = {7, 8};
//+
Line(11) = {5, 9};
//+
Line(12) = {6, 10};
//+
Line(13) = {7, 11};
//+
Line(14) = {8, 12};
//+
Line(15) = {9, 10};
//+
Line(16) = {10, 11};
//+
Line(17) = {11, 12};
//+
Line(18) = {9, 13};
//+
Line(19) = {10, 14};
//+
Line(20) = {11, 15};
//+
Line(21) = {12, 16};
//+
Line(22) = {13, 14};
//+
Line(23) = {14, 15};
//+
Line(24) = {15, 16};
//+
Line(25) = {13, 17};
//+
Line(26) = {14, 18};
//+
Line(27) = {15, 19};
//+
Line(28) = {16, 20};
//+
Line(29) = {17, 18};
//+
Line(30) = {18, 19};
//+
Line(31) = {19, 20};
//+
Line(32) = {17, 21};
//+
Line(33) = {18, 22};
//+
Line(34) = {19, 23};
//+
Line(35) = {20, 24};
//+
Line(36) = {21, 22};
//+
Line(37) = {22, 23};
//+
Line(38) = {23, 24};
//+
Line Loop(1) = {1, 5, -8, -4};
//+
Plane Surface(1) = {1};
//+
Line Loop(2) = {2, 6, -9, -5};
//+
Plane Surface(2) = {2};
//+
Line Loop(3) = {3, 7, -10, -6};
//+
Plane Surface(3) = {3};
//+
Line Loop(4) = {11, 15, -12, -8};
//+
Plane Surface(4) = {4};
//+
Line Loop(6) = {13, 17, -14, -10};
//+
Plane Surface(6) = {6};
//+
Line Loop(7) = {18, 22, -19, -15};
//+
Plane Surface(7) = {7};
//+
Line Loop(8) = {19, 23, -20, -16};
//+
Plane Surface(8) = {8};
//+
Line Loop(9) = {20, 24, -21, -17};
//+
Plane Surface(9) = {9};
//+
Line Loop(10) = {25, 29, -26, -22};
//+
Plane Surface(10) = {10};
//+
Line Loop(12) = {27, 31, -28, -24};
//+
Plane Surface(12) = {12};
//+
Line Loop(13) = {32, 36, -33, -29};
//+
Plane Surface(13) = {13};
//+
Line Loop(14) = {33, 37, -34, -30};
//+
Plane Surface(14) = {14};
//+
Line Loop(15) = {34, 38, -35, -31};
//+
Plane Surface(15) = {15};
//+
Extrude {0, boxSize[1], 0} {
  Surface{1}; 
  Surface{2}; 
  Surface{3}; 
  Surface{4}; 
  Surface{6}; 
  Surface{7}; 
  Surface{8}; 
  Surface{9}; 
  Surface{10}; 
  Surface{12}; 
  Surface{13}; 
  Surface{14}; 
  Surface{15};

  Layers{gs}; 
  Recombine;
}
//+
Line Loop(67) = {63, -75, -61, 51};
Plane Surface(67) = {67};
//+
Line Loop(68) = {83, 97, -85, -73};
Plane Surface(68) = {68};


//Extrude Progression (Manual)
n = gs; // number of intervals
r = 1/0.9365549739; // progression

a = (r - 1) / (r^n - 1);
one[0] = 1;
layer[0] = a;
For i In {1:n-1}
  one[i] = 1;
  layer[i] = layer[i-1] + a * r^i;
EndFor

Extrude {0, dimy-boxSize[1], 0} {
  Surface{20}; 
  Surface{32}; 
  Surface{40}; 
  Surface{51}; 
  Surface{59}; 
  Surface{63}; 
  Surface{68}; 
  Surface{44}; 
  Surface{67}; 
  Surface{24}; 
  Surface{28}; 
  Surface{36}; 
  Surface{47}; 
  Surface{55}; 
  Surface{66};
  Layers{one[], layer[]};
  Recombine; 
}

//Front X+ Progression
Transfinite Line {4, 5, 6, 7} = gs Using Progression 0.9057255027;
Transfinite Line {55, 50, 43, 46} = gs Using Progression 0.9057255027;
Transfinite Line {149, 145, 105, 108} = gs Using Progression 0.9057255027;

// Back X- Progression
Transfinite Line {-35, -34, -33, -32} = 2*gs Using Progression 0.9345152035;
// Transfinite Line {100, 96, 93, 90} = 2*gs Using Progression 0.9345152035;
// Transfinite Line {-162, -132, -128, -125} = 2*gs Using Progression 0.9345152035;

// Z Progression
Transfinite Line {-3, -10, -17, -24, -31, -38, 36, 29, 22, 15, 8, 1} = gs Using Progression 0.9057255027;
// Transfinite Line {99, 87, 77, 65, 56, 53, 41, 45, 60, 70, 82, 92} = gs Using Progression 0.9057255027;
// Transfinite Line {-161, -158, -155, -152, -150, -147, 103, 107, 112, 117, 122, 127} = gs Using Progression 0.9057255027;

// Box Uniform Meshes
Transfinite Line {37, 25, 18, 11, 12, 19, 26, 30, 27, 23, 20, 16, 13, 9, 2, 14, 21, 28} = gs Using Progression 1;
Transfinite Line {66, 78, 88, 95, 80, 68, 58, 48, 51, 61, 63, 23, 71, 74, 85, 83, 97, 73, 75} = gs Using Progression 1;
Transfinite Line {130, 120, 115, 110, 118, 113, 123, 133, 135, 139, 142, 141, 138, 136, 144, 153, 156, 159} = gs Using Progression 1;

// Force structured grid for all
Transfinite Surface "*";
Recombine Surface "*";

// Physical Surfaces for OpenFOAM
Physical Surface("FixedWalls") = {13, 14, 15, 12, 9, 6, 3, 2, 1, 4, 7, 10, 52, 62, 68, 50, 41, 8, 33, 67, 31, 23, 43};
Physical Surface("SideWall1") = {26, 107, 111, 114, 117, 120, 65, 54, 46, 35};
Physical Surface("SideWall2") = {19, 72, 29, 37, 48, 56, 86, 82, 78, 74};
Physical Surface("Outlet") = {60, 64, 119, 90, 87, 57};
Physical Surface("Inlet") = {16, 69, 103, 106, 25, 21};
Physical Surface("UpperWall") = {121, 93, 89, 109, 105, 73, 112, 115, 118, 96, 99, 102, 77, 81, 85};

Physical Volume("Volume") = {11, 18, 19, 28, 13, 12, 17, 27, 20, 21, 26, 22, 8, 10, 15, 16, 4, 7, 6, 9, 1, 5, 14, 3, 2, 23, 24, 25};
