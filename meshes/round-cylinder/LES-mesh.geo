// Flow Around Cylinder
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: https://doi.org/10.1016/j.jcp.2019.01.021
// github: https://github.com/cics-nd/rans-uncertainty

// Input Variables
// Diameter of the cylinders
d = 1;
r = d/2;
// Size of the box
dimx = 30*d;
dimy = 16*d;
dimz = 3*d;
// Location of cylinders (x,y)
cydLoc = {8*d, dimy/2};

// Mesh Resolution
n_arch = 30; //# on 1/4 cylinder surface
n_norm = 30; //# on mesh diagonals normal to cylinder
n_span = 40; //# of points on Y parallel lines
n_xInlet = 50; //# of points on X parallel inlet lines
n_xOutlet = 100; //# of points on X parallel inlet lines
n_z = 25;
//
p_norm = 1.05;
p_span = 1.05;
p_xInlet = 1.05;
p_xOutlet = 1.03;

SetFactory("OpenCASCADE");
//Draw out cylinder 1
// Center
cen1 = newp;
Point(cen1) = {cydLoc[0], cydLoc[1], 0, 1.0};
//Outer Points
For i In {0:3}
    cyldPList1[i] = newp;
    Point(cyldPList1[i]) = {cydLoc[0]+r*Cos(i*Pi/2 + Pi/4), cydLoc[1]+r*Sin(i*Pi/2 + Pi/4), 0, 1.0};
EndFor
// Archs
For i In {0:2}
    circp = newl;
    Circle(circp) = {cyldPList1[i], cen1, cyldPList1[i+1]}; Transfinite Line {circp} = n_arch Using Progression 1;
EndFor
circp = newl;
Circle(circp) = {cyldPList1[3], cen1, cyldPList1[0]}; Transfinite Line {circp} = n_arch Using Progression 1;

// Now construct box around the two circles
c1 = Point{cen1};

Point(newp) = {cydLoc[0]-d, cydLoc[1]+d, 0, 1.0};
Point(newp) = {cydLoc[0]+d, cydLoc[1]+d, 0, 1.0};
Point(newp) = {cydLoc[0]+d, cydLoc[1]-d, 0, 1.0};
Point(newp) = {cydLoc[0]-d, cydLoc[1]-d, 0, 1.0};
//+
Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 9};
Line(8) = {9, 6};

// Diagonals
Line(9) = {6, 3};
Line(10) = {7, 2};
Line(11) = {8, 5};
Line(12) = {9, 4};

// Outer Box
Point(newp) = {0, 0, 0, 1.0};
Point(newp) = {0, cydLoc[1]-d, 0, 1.0};
Point(newp) = {0, cydLoc[1]+d, 0, 1.0};
Point(newp) = {0, dimy, 0, 1.0};
Point(newp) = {cydLoc[0]-d, dimy, 0, 1.0};
Point(newp) = {cydLoc[0]+d, dimy, 0, 1.0};
Point(newp) = {dimx, dimy, 0, 1.0};
Point(newp) = {dimx, cydLoc[1]+d, 0, 1.0};
Point(newp) = {dimx, cydLoc[1]-d, 0, 1.0};
Point(newp) = {dimx, 0, 0, 1.0};
Point(newp) = {cydLoc[0]-d, 0, 0, 1.0};
Point(newp) = {cydLoc[0]+d, 0, 0, 1.0};
//+
Line(13) = {10, 11};
Line(14) = {11, 12};
Line(15) = {12, 13};
Line(16) = {13, 14};
Line(17) = {14, 15};
Line(18) = {15, 16};
Line(19) = {16, 17};
Line(20) = {17, 18};
Line(21) = {18, 19};
Line(22) = {19, 21};
Line(23) = {21, 20};
Line(24) = {20, 10};

// Inter connection lines
Line(25) = {11, 9};
Line(26) = {12, 6};
Line(27) = {14, 6};
Line(28) = {15, 7};
Line(29) = {17, 7};
Line(30) = {18, 8};
Line(31) = {21, 8};
Line(32) = {20, 9};

// Mesh around circle
Transfinite Line {1, 5, 4, 6, 7, 3, 8, 2} = n_arch Using Progression 1;
Transfinite Line {12, 12, 9, 10, 11} = n_norm Using Progression p_norm;

Line Loop(1) = {1, -9, 5, 10};
Plane Surface(1) = {1};
Line Loop(2) = {4, -10, 6, 11};
Plane Surface(2) = {2};
Line Loop(3) = {3, -11, 7, 12};
Plane Surface(3) = {3};
Line Loop(4) = {12, -2, -9, -8};
Plane Surface(4) = {4};

Transfinite Surface {3};
Transfinite Surface {2};
Transfinite Surface {1};
Transfinite Surface {4};

// Mesh outer region
Transfinite Line {20, 17, 14, 23} = n_arch Using Progression 1;
Transfinite Line {-22, -30, -29, 18} = n_xOutlet Using Progression p_xOutlet;
Transfinite Line {-16, -26, -25, 24} = n_xInlet Using Progression p_xInlet;
Transfinite Line {-32, -31, -13, 21, -19, -28, -27, 15} = n_span Using Progression p_span;

//+
Line Loop(5) = {29, -28, 18, 19};
Plane Surface(5) = {5};
Line Loop(6) = {29, 6, -30, -20};
Plane Surface(6) = {6};
Line Loop(7) = {30, -31, -22, -21};
Plane Surface(7) = {7};
Line Loop(8) = {31, 7, -32, -23};
Plane Surface(8) = {8};
Line Loop(9) = {32, -25, -13, -24};
Plane Surface(9) = {9};
Line Loop(10) = {25, 8, -26, -14};
Plane Surface(10) = {10};
Line Loop(11) = {26, -27, -16, -15};
Plane Surface(11) = {11};
Line Loop(12) = {5, -28, -17, 27};
Plane Surface(12) = {12};
//+
Transfinite Surface {11};
Transfinite Surface {12};
Transfinite Surface {5};
Transfinite Surface {6};
Transfinite Surface {7};
Transfinite Surface {8};
Transfinite Surface {9};
Transfinite Surface {10};
Recombine Surface "*";

//+
Extrude {0, 0, 3*d} {
  Surface{11}; 
  Surface{12}; 
  Surface{5}; 
  Surface{6}; 
  Surface{7}; 
  Surface{8}; 
  Surface{9}; 
  Surface{10}; 
  Surface{1}; 
  Surface{2}; 
  Surface{3}; 
  Surface{4};
  Layers{n_z}; 
  Recombine;
}

//+
Physical Surface("Inlet") = {16, 43, 39};
//+
Physical Surface("Cylinder") = {55, 45, 49, 52};
//+
Physical Surface("Outlet") = {24, 28, 32};
//+
Physical Surface("FixedWalls") = {15, 20, 23, 40, 36, 31};
//+
Physical Surface("SideWall1") = {17, 21, 25, 29, 48, 51, 56, 54, 44, 41, 37, 33};
//+
Physical Surface("SideWall2") = {11, 12, 5, 6, 7, 8, 10, 1, 2, 3, 4, 9};
//+
Physical Volume("Volume") = {3, 4, 5, 6, 7, 8, 1, 2, 9, 10, 11, 12};
