// Tandem Cylinders Mesh
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: 
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: http://www.doi.org/10.1016/j.jweia.2015.03.017

// Input Variables
// Diameter of the cylinders
d = 2;
r = d/2;
// Size of the box
dimx = 30*d;
dimy = 20*d;
dimz = 3*d;
// Location of cylinders (x,y)
cydLoc = {10*d, dimy/2};
cydLoc2 = {10*d + 3.7*d, dimy/2};

// Mesh Resolution
n_arch = 120; //# on 1/4 cylinder surface
n_norm = 120; //# on mesh diagonals normal to cylinder
n_span = 60; //# of points on Y parallel lines
n_xInlet = 50; //# of points on X parallel inlet lines
n_xOutlet = 180; //# of points on X parallel inlet lines
n_z = 120;
//
p_norm = 1.025;
p_span = 1.04;
p_xInlet = 1.05;
p_xOutlet = 1.01;

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


//Draw out cylinder 2
// Center
cen2 = newp;
Point(cen2) = {cydLoc2[0], cydLoc2[1], 0, 1.0};
//Outer Points
For i In {0:3}
    cyldPList2[i] = newp;
    Point(cyldPList2[i]) = {cydLoc2[0]+r*Cos(i*Pi/2 + Pi/4), cydLoc2[1]+r*Sin(i*Pi/2 + Pi/4), 0, 1.0};
EndFor
// Archs
For i In {0:2}
    circp = newl;
    Circle(circp) = {cyldPList2[i], cen2, cyldPList2[i+1]}; Transfinite Line {circp} = n_arch Using Progression 1;
EndFor
circp = newl;
Circle(circp) = {cyldPList2[3], cen2, cyldPList2[0]}; Transfinite Line {circp} = n_arch Using Progression 1;


// Now construct box around the two circles
c1 = Point{cen1};
c2 = Point{cen2};

dx0 = Abs(cydLoc[0] - cydLoc2[0])/2;
p0 = newp;
Point(p0) = {c1[0]-dx0, c1[1]+dx0, 0, 1.0};
Point(p0+1) = {c1[0]+dx0, c1[1]+dx0, 0, 1.0};
Point(p0+2) = {c2[0]+dx0, c2[1]+dx0, 0, 1.0};
Point(p0+3) = {c2[0]+dx0, c2[1]-dx0, 0, 1.0};
Point(p0+4) = {c1[0]+dx0, c1[1]-dx0, 0, 1.0};
Point(p0+5) = {c1[0]-dx0, c1[1]-dx0, 0, 1.0};
//+
Line(9) = {11, 12}; Transfinite Line {9} = n_arch Using Progression 1;
Line(10) = {12, 13}; Transfinite Line {10} = n_arch Using Progression 1;
Line(11) = {13, 14}; Transfinite Line {11} = n_arch Using Progression 1;
Line(12) = {14, 15}; Transfinite Line {12} = n_arch Using Progression 1;
Line(13) = {15, 16}; Transfinite Line {13} = n_arch Using Progression 1;
Line(14) = {16, 11}; Transfinite Line {14} = n_arch Using Progression 1;
Line(15) = {12, 15}; Transfinite Line {15} = n_arch Using Progression 1;

//Diagonals
Line(16) = {2, 12}; Transfinite Line {16} = n_norm Using Progression p_norm;
Line(17) = {3, 11}; Transfinite Line {17} = n_norm Using Progression p_norm;
Line(18) = {4, 16}; Transfinite Line {18} = n_norm Using Progression p_norm;
Line(19) = {5, 15}; Transfinite Line {19} = n_norm Using Progression p_norm;
Line(20) = {8, 12}; Transfinite Line {20} = n_norm Using Progression p_norm;
Line(21) = {9, 15}; Transfinite Line {21} = n_norm Using Progression p_norm;
Line(22) = {10, 14}; Transfinite Line {22} = n_norm Using Progression p_norm;
Line(23) = {7, 13}; Transfinite Line {23} = n_norm Using Progression p_norm;

//Outer box
p0 = newp;
Point(p0) = {0, 0, 0, 1.0};
Point(p0+1) = {0, c1[1]-dx0, 0, 1.0};
Point(p0+2) = {0, c1[1]+dx0, 0, 1.0};
Point(p0+3) = {0, dimy, 0, 1.0};
Point(p0+4) = {c1[0]-dx0, dimy, 0, 1.0};
Point(p0+5) = {c1[0]+dx0, dimy, 0, 1.0};
Point(p0+6) = {c2[0]+dx0, dimy, 0, 1.0};
Point(p0+7) = {dimx, dimy, 0, 1.0};
Point(p0+8) = {dimx, c1[1]+dx0, 0, 1.0};
Point(p0+9) = {dimx, c1[1]-dx0, 0, 1.0};
Point(p0+10) = {dimx, 0, 0, 1.0};
Point(p0+11) = {c2[0]+dx0, 0, 0, 1.0};
Point(p0+12) = {c1[0]+dx0, 0, 0, 1.0};
Point(p0+13) = {c1[0]-dx0, 0, 0, 1.0};

// Lines of outer parameter
Line(24) = {17, 18};
Line(25) = {18, 19};
Line(26) = {19, 20};
Line(27) = {20, 21};
Line(28) = {21, 22};
Line(29) = {22, 23};
Line(30) = {23, 24};
Line(31) = {24, 25};
Line(32) = {25, 26};
Line(33) = {26, 27};
Line(34) = {27, 28};
Line(35) = {28, 29};
Line(36) = {29, 30};
Line(37) = {30, 17};

// Wall/Inlet normal Lines
Line(38) = {16, 18};
Line(39) = {11, 19};
Line(40) = {11, 21};
Line(41) = {12, 22};
Line(42) = {13, 23};
Line(43) = {13, 25};
Line(44) = {14, 26};
Line(45) = {14, 28};
Line(46) = {15, 29};
Line(47) = {16, 30};

// Transfinite the outer lines
// Y parallel lines
Transfinite Line {-24, 47, 46, 45, 33, -31, 42, 41, 40, 26} = n_span Using Progression p_span;
// Front X parallel lines
Transfinite Line {-27, 39, 38, 37} = n_xInlet Using Progression p_xInlet;
// Back X parallel lines
Transfinite Line {30, 43, 44, -34} = n_xOutlet Using Progression p_xOutlet;
// Outer "circle box" lines
Transfinite Line {32, 29, 28, 25, 36, 35} = n_arch Using Progression 1;

//Surface outer region
Line Loop(1) = {39, 26, 27, -40};
Plane Surface(1) = {1};
Line Loop(2) = {9, 41, -28, -40};
Plane Surface(2) = {2};
Line Loop(3) = {10, 42, -29, -41};
Plane Surface(3) = {3};
Line Loop(4) = {43, -31, -30, -42};
Plane Surface(4) = {4};
Line Loop(5) = {11, 44, -32, -43};
Plane Surface(5) = {5};
Line Loop(6) = {44, 33, 34, -45};
Plane Surface(6) = {6};
Line Loop(7) = {12, 46, -35, -45};
Plane Surface(7) = {7};
Line Loop(8) = {13, 47, -36, -46};
Plane Surface(8) = {8};
Line Loop(9) = {38, -24, -37, -47};
Plane Surface(9) = {9};
Line Loop(10) = {14, 39, -25, -38};
Plane Surface(10) = {10};

Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};
Transfinite Surface {6};
Transfinite Surface {7};
Transfinite Surface {8};
Transfinite Surface {9};
Transfinite Surface {10};

//Surface planes around cylinders
Line Loop(11) = {22, 12, -21, 7};
Plane Surface(11) = {11};
Line Loop(12) = {21, -15, -20, 6};
Plane Surface(12) = {12};
Line Loop(13) = {20, 10, -23, 5};
Plane Surface(13) = {13};
Line Loop(14) = {23, 11, -22, 8};
Plane Surface(14) = {14};
Line Loop(15) = {13, -18, 3, 19};
Plane Surface(15) = {15};
Line Loop(16) = {18, 14, -17, 2};
Plane Surface(16) = {16};
Line Loop(17) = {17, 9, -16, 1};
Plane Surface(17) = {17};
Line Loop(18) = {16, 15, -19, 4};
Plane Surface(18) = {18};

Transfinite Surface {16};
Transfinite Surface {17};
Transfinite Surface {18};
Transfinite Surface {15};
Transfinite Surface {12};
Transfinite Surface {13};
Transfinite Surface {14};
Transfinite Surface {11};
Recombine Surface "*";

// And extrudes
Extrude {0, 0, dimz} { 
  Surface{1}; 
  Surface{2}; 
  Surface{3}; 
  Surface{4}; 
  Surface{5}; 
  Surface{6}; 
  Surface{7}; 
  Surface{8}; 
  Surface{9}; 
  Surface{10};
  Surface{16}; 
  Surface{17}; 
  Surface{18}; 
  Surface{15}; 
  Surface{12}; 
  Surface{13}; 
  Surface{14}; 
  Surface{11};
  Layers{n_z}; 
  Recombine;
}

// Physical definitions for OpenFOAM
Physical Surface("FixedWalls") = {21, 26, 30, 34, 54, 50, 46, 41};
Physical Surface("Inlet") = {53, 57, 20};
Physical Surface("Outlet") = {40, 38, 33};
Physical Surface("SideWall2") = {43, 47, 51, 55, 58, 71, 62, 65, 69, 75, 78, 81, 83, 39, 35, 31, 27, 23};
Physical Surface("SideWall1") = {9, 8, 7, 6, 10, 16, 17, 18, 15, 12, 13, 14, 11, 5, 1, 2, 3, 4};
Physical Surface("Cylinder") = {80, 82, 74, 77, 64, 61, 70, 68};

Physical Volume("Volume") = {6, 5, 4, 3, 2, 1, 12, 11, 14, 13, 16, 15, 18, 17, 10, 7, 8, 9};
