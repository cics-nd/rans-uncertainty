// Flow Around Wall Mounted Cube Mesh
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: https://doi.org/10.1016/j.jcp.2019.01.021
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: http://www.doi.org/10.1016/j.ijheatfluidflow.2006.02.026

// Input Variables
dimx = 14;
dimy = 3;
dimz = 7;
gs = 50;

boxPos = {3,0,3};
boxSize = {1,1,1};

SetFactory("OpenCASCADE");
// Set Front Lower Box
Point(1) = {0,0,0};
Point(2) = {0,0,boxPos[2]};
Point(3) = {boxPos[0],0,boxPos[2]};
Point(4) = {boxPos[0],0,0};

Line(5) = {1,2};
Line(6) = {2,3};
Line(7) = {3,4};
Line(8) = {4,1};

Line Loop(9) = {5,6,7,8};
Plane Surface(101) = 9;

// Set Front Middle Lower Box
Point(5) = {0, 0, boxPos[2]+boxSize[2], 1.0};
Point(6) = {boxPos[0], 0, boxPos[2]+boxSize[2], 1.0};

Line(9) = {2, 5};
Line(10) = {5, 6};
Line(11) = {6, 3};

Line Loop(11) = {9, 10, 11, -6};
Plane Surface(102) = {11};

// Set Front Z+ Lower Box
Point(7) = {0, 0, dimz, 1.0};
Point(8) = {boxPos[0], 0, dimz, 1.0};

Line(12) = {5, 7};
Line(13) = {7, 8};
Line(14) = {8, 6};

Line Loop(12) = {12, 13, 14, -10};
Plane Surface(103) = {12};

// Middle Lower Z- Box
Point(9) = {boxPos[0]+boxSize[0], 0, boxPos[2], 1.0};
Point(10) = {boxPos[0]+boxSize[0], 0, 0, 1.0};

Line(15) = {3, 9};
Line(16) = {9, 10};
Line(17) = {10, 4};

Line Loop(13) = {7, -17, -16, -15};
Plane Surface(104) = {13};

// Middle Lower Z+ Box
Point(11) = {boxPos[0]+boxSize[0], 0, boxPos[2]+boxSize[2], 1.0};
Point(12) = {boxPos[0]+boxSize[0], 0, dimz, 1.0};

Line(18) = {6, 11};
Line(19) = {11, 12};
Line(20) = {12, 8};

Line Loop(14) = {14, 18, 19, 20};
Plane Surface(105) = {14};

// Back Lower Z- Box
Point(13) = {dimx, 0, boxPos[2], 1.0};
Point(14) = {dimx, 0, 0, 1.0};

Line(21) = {9, 13};
Line(22) = {13, 14};
Line(23) = {14, 10};

Line Loop(15) = {16, -23, -22, -21};
Plane Surface(106) = {15};

// Back Lower Middle Box
Point(15) = {dimx, 0, boxPos[2]+boxSize[2], 1.0};

Line(24) = {9, 11};
Line(25) = {11, 15};
Line(26) = {15, 13};

Line Loop(16) = {24, 25, 26, -21};
Plane Surface(107) = {16};

// Back Lower Z+Box
Point(16) = {dimx, 0, dimz, 1.0};

Line(27) = {12, 16};
Line(28) = {16, 15};

Line Loop(17) = {19, 27, 28, -25};
Plane Surface(108) = {17};

//+
Extrude {0, 1, 0} {
  Surface{101}; 
  Surface{102}; 
  Surface{103}; 
  Surface{105}; 
  Surface{104}; 
  Surface{106}; 
  Surface{107}; 
  Surface{108}; 
  Layers{gs}; 
  Recombine;
}
//+
Line Loop(50) = {41, 56, 62, -48};
Plane Surface(141) = {50};

//Extrude Progression (Manual)
n = gs; // number of intervals
r = 1/0.9747679224; // progression

a = (r - 1) / (r^n - 1);
one[0] = 1;
layer[0] = a;
For i In {1:n-1}
  one[i] = 1;
  layer[i] = layer[i-1] + a * r^i;
EndFor

Extrude {0, dimy-boxSize[1], 0} {
  Surface{113}; 
  Surface{117}; 
  Surface{121}; 
  Surface{125}; 
  Surface{141}; 
  Surface{129}; 
  Surface{133}; 
  Surface{137}; 
  Surface{140};
  Layers{one[], layer[]};
  Recombine;
}

//Front X+ Progression
Transfinite Line {6, -8, 10, 13} = gs Using Progression 0.9620641198;
Transfinite Line {45, 40, 33, 36} = gs Using Progression 0.9620641198;
Transfinite Line {85, 80, 73, -76} = gs Using Progression 0.9620641198;

// Back X- Progression
Transfinite Line {23, -21, -25, -27} = 2*gs Using Progression 0.973562156;
Transfinite Line {58, 61, 64, 67} = 2*gs Using Progression 0.973562156; 
Transfinite Line {103, -99, -94, -91} = 2*gs Using Progression 0.973562156;

// Z Progression
Transfinite Line {5, -7, -16, -22, 28, -19, 14, -12} = gs Using Progression 0.9620641198; // Bottom Y
Transfinite Line {68, 60, 55, 50, 31, 43, 46, 35} = gs Using Progression 0.9620641198; // Middle Y
Transfinite Line {93, -89, 86, -83, -71, 75, -101, -104} = gs Using Progression 0.9620641198; // Top Y


//Box Uniform Meshes
// Transfinite Line {17, 15, 11, 24, 18, 9, 38, 78, 20, 51, 108, 48, 41, 56, 62, 24, 37, 30, 32, 54, 39, 47, 105, 106, 96, 107, 81, 26, 65, 98, 52, 53} = gs Using Progression 1;
Transfinite Line {20, 51, 108, 9, 38, 78, 17, 53, 105, 26, 65, 98, 24, 62, 96, 11, 41, 81, 18, 48, 107, 15, 56, 106, 49, 44, 37, 30, 34, 52, 59, 63, 54, 47, 39, 32} = gs Using Progression 1;

// Force structured grid for all
Transfinite Surface "*";
Recombine Surface "*";


//Now Physical Faces for OpenFOAM
//+
Physical Surface("FixedWalls") = {154, 158, 174, 171, 168, 164, 161, 150, 146, 108, 105, 103, 102, 107, 106, 104, 101};
//+
Physical Surface("Inlet") = {142, 147, 151, 118, 114, 109};
//+
Physical Surface("SideWall_2") = {165, 162, 145, 112, 126, 130};
//+
Physical Surface("SideWall_1") = {152, 157, 172, 138, 124, 119};
//+
Physical Surface("Box") = {128, 134, 141, 122, 116};
//+
Physical Surface("Outlet") = {173, 139, 170, 136, 131, 166};
//+
Show "*";
//+
Physical Volume(8) = {17, 16, 15, 6, 7, 8, 4, 12, 13, 14, 5, 9, 10, 11, 3, 2, 1};
