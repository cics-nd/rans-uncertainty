// Jet in Crossflow Mesh
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: 
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: https://doi.org/10.1063/1.4915065

// Input Variables
//Diameter of the jet
d = 1;
// Size of the box
dimx = 33*d;
dimy = 33*d;
dimz = 40*d;
pipeLen = 10*d;
// Location of jet (z,y)
pipeLoc = {5*d, 20*d};

// Mesh Resolution
n_arch = 30; //# on jet arch
n_wall = 150; //# on cube walls
n_outerwall = 60; //# on outer wall
n_face = 100; //# on cube walls
n_pipe = 100; // # of pipe layers


SetFactory("OpenCASCADE");
// Box Points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {dimx, 0, 0, 1.0};
Point(3) = {dimx, dimy, 0, 1.0};
Point(4) = {0, dimy, 0, 1.0};

Point(5) = {0, 0, dimz, 1.0};
Point(6) = {dimx, 0, dimz, 1.0};
Point(7) = {dimx, dimy, dimz, 1.0};
Point(8) = {0, dimy, dimz, 1.0};

//Points for connections to cylinder
Point(9) = {0, 0, pipeLoc[1], 1.0};
Point(10) = {0, pipeLoc[0], dimz, 1.0};
Point(11) = {0, dimy, pipeLoc[1], 1.0};
Point(12) = {0, pipeLoc[0], 0, 1.0};

// Box Lines (Z=0 plane)
Line(1) = {1, 2}; Transfinite Line {1} = n_wall Using Progression 1/0.9;
Line(2) = {2, 3}; Transfinite Line {2} = n_outerwall Using Progression 1;
Line(3) = {3, 4}; Transfinite Line {3} = n_wall Using Progression 0.9;
Line(4) = {4, 12}; Transfinite Line {4} = n_arch Using Progression 1;
Line(5) = {12, 1}; Transfinite Line {5} = n_arch Using Progression 1;
//Diag
Line(41) = {12, 3}; Transfinite Line {41} = n_wall Using Progression 1;

// Box Lines (Z=dimz plane)
Line(6) = {5, 6}; Transfinite Line {6} = n_wall Using Progression 1/0.9;
Line(7) = {6, 7}; Transfinite Line {7} = n_outerwall Using Progression 1;
Line(8) = {7, 8}; Transfinite Line {8} = n_wall Using Progression 0.9;
Line(9) = {8, 10}; Transfinite Line {9} = n_arch Using Progression 1;
Line(10) = {10, 5}; Transfinite Line {10} = n_arch Using Progression 1;
//Diag
Line(42) = {10, 7}; Transfinite Line {42} = n_wall Using Progression 1;

// Box Lines (Connector lines)
Line(11) = {1, 9}; Transfinite Line {11} = n_arch Using Progression 1;
Line(12) = {9, 5}; Transfinite Line {12} = n_arch Using Progression 1;
Line(13) = {2, 6}; Transfinite Line {13} = n_outerwall Using Progression 1;
Line(14) = {3, 7}; Transfinite Line {14} = n_outerwall Using Progression 1;
Line(15) = {4, 11}; Transfinite Line {15} = n_arch Using Progression 1;
Line(16) = {11, 8}; Transfinite Line {16} = n_arch Using Progression 1;
//Diags
Line(43) = {11, 3}; Transfinite Line {43} = n_wall Using Progression 1/0.9;
Line(44) = {11, 7}; Transfinite Line {44} = n_wall Using Progression 1/0.9;
Line(45) = {9, 2}; Transfinite Line {45} = n_wall Using Progression 1/0.9;
Line(46) = {9, 6}; Transfinite Line {46} = n_wall Using Progression 1/0.9;

// Jet Outlet
// Center
Point(17) = {0, pipeLoc[0], pipeLoc[1], 1.0};
//Outer Points
For i In {0:7}
    pList[i] = newp;
    Point(pList[i]) = {0, pipeLoc[0]+d*Cos(i*Pi/4), pipeLoc[1]+d*Sin(i*Pi/4), 1.0};
EndFor
// Archs
For i In {0:6}
    Circle(17+i) = {18+i, 17, 19+i}; Transfinite Line {17+i} = n_arch Using Progression 1;
EndFor
Circle(24) = {25, 17, 18}; Transfinite Line {24} = n_arch Using Progression 1;
// Lines from outer to inner of circle
For i In {0:7}
    Line(25+i) = {17, 18+i}; Transfinite Line {25+i} = n_arch Using Progression 0.9;
EndFor

// Connection lines from circle to outer box
Line(33) = {18, 11}; 
Line(34) = {19, 8};
Line(35) = {20, 10};
Line(36) = {21, 5};
Line(37) = {22, 9};
Line(38) = {23, 1};
Line(39) = {24, 12};
Line(40) = {25, 4};
For i In {33:40}
    Transfinite Line {i} = n_wall Using Progression 1.05;
EndFor

// Create Surfaces
// Pipe Surfaces
Line Loop(14) = {29, -20, -28};
Plane Surface(14) = {14};
Line Loop(15) = {28, -19, -27};
Plane Surface(15) = {15};
Line Loop(16) = {27, -18, -26};
Plane Surface(16) = {16};
Line Loop(17) = {26, -17, -25};
Plane Surface(17) = {17};
Line Loop(18) = {25, -24, -32};
Plane Surface(18) = {18};
Line Loop(19) = {32, -23, -31};
Plane Surface(19) = {19};
Line Loop(20) = {31, -22, -30};
Plane Surface(20) = {20};
Line Loop(21) = {30, -21, -29};
Plane Surface(21) = {21};
For i In {14:21}
    Transfinite Surface {i};
EndFor

// Face around pipe
Line Loop(22) = {34, 9, -35, -18};
Plane Surface(22) = {22};
Line Loop(23) = {33, 16, -34, -17};
Plane Surface(23) = {23};
Line Loop(24) = {40, 15, -33, -24};
Plane Surface(24) = {24};
Line Loop(25) = {39, -4, -40, -23};
Plane Surface(25) = {25};
Line Loop(26) = {38, -5, -39, -22};
Plane Surface(26) = {26};
Line Loop(27) = {37, -11, -38, -21};
Plane Surface(27) = {27};
Line Loop(28) = {36, -12, -37, -20};
Plane Surface(28) = {28};
Line Loop(29) = {35, 10, -36, -19};
Plane Surface(29) = {29};
For i In {22:29}
    //Transfinite Surface {i};
EndFor

// Remaining Cub Faces (Keep nor structurec mesh)
Line Loop(30) = {42, 8, 9};
Plane Surface(30) = {30};
Line Loop(31) = {10, 6, 7, -42};
Plane Surface(31) = {31};
Line Loop(32) = {3, 4, 41};
Plane Surface(32) = {32};
Line Loop(33) = {41, -2, -1, -5};
Plane Surface(33) = {33};
Line Loop(34) = {14, -7, -13, 2};
Plane Surface(34) = {34};
Line Loop(35) = {46, -6, -12};
Plane Surface(35) = {35};
Line Loop(36) = {45, 13, -46};
Plane Surface(36) = {36};
Line Loop(37) = {1, -45, -11};
Plane Surface(37) = {37};
Line Loop(38) = {3, 15, 43};
Plane Surface(38) = {38};
Line Loop(39) = {43, 14, -44};
Plane Surface(39) = {39};
Line Loop(40) = {44, 8, -16};
Plane Surface(40) = {40};

// Cube Volume
Surface Loop(1) = {33, 32, 38, 24, 25, 26, 27, 28, 29, 22, 23, 40, 39, 34, 31, 35, 36, 37, 30, 17, 16, 15, 14, 21, 20, 19, 18};
Volume(1) = {1};

// Jet Extrude
Extrude {-10*d, 0, 0} {
  Surface{20}; 
  Surface{19}; 
  Surface{18}; 
  Surface{17}; 
  Surface{16}; 
  Surface{15}; 
  Surface{14}; 
  Surface{21};
  Layers{n_pipe}; 
  Recombine;
}


Physical Surface("Jet_Inlet") = {56, 53, 50, 47, 44, 64, 62, 59};
Physical Surface("Jet_Walls") = {55, 52, 49, 42, 63, 61, 58, 46};
Physical Surface("Inlet") = {37, 36, 35};
Physical Surface("Outlet") = {40, 39, 38};
Physical Surface("Fixed_Walls") = {25, 32, 24, 23, 22, 30, 29, 28, 27, 26, 31, 34, 33};
//+
Physical Volume("Volume") = {1, 2, 9, 8, 7, 6, 5, 4, 3};
