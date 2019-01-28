// Square Duct Mesh
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: 
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: http://www.doi.org/10.1017/S0022112009992242

// Input Variables
// Half duct width
h = 1;
l = 4*Pi*h;

// Mesh Resolution
n_wall = 30; //# on cube walls
n_duct = 20; //# Inside of duct
n_length = Floor(n_wall*Pi*0.25); //Streamwise points
n_length = 100;

// Mesh refinement ratios
p_duct = 0.95;

SetFactory("OpenCASCADE");
// Box Points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 0, h, 1.0};
Point(3) = {0, h, h, 1.0};
Point(4) = {0, h, 0, 1.0};
Point(5) = {0, h, -h, 1.0};
Point(6) = {0, 0, -h, 1.0};
Point(7) = {0, -h, -h, 1.0};
Point(8) = {0, -h, 0, 1.0};
Point(9) = {0, -h, h, 1.0};
// Consrtuct outer duct boundary
For i In {2:8}
    j = newl;
    Line(j) = {i, i+1};  Transfinite Line{j} = n_wall Using Progression 1;
EndFor
Line(8) = {9, 2};  Transfinite Line{8} = n_wall Using Progression 1;

// Connector lines
// For i In {2:9}
//     j = newl;
//     Line(j) = {1, i};  Transfinite Line{j} = n_duct Using Progression p_duct;
// EndFor
Line(9) = {1, 2};  Transfinite Line{9} = n_duct Using Progression p_duct;
Line(10) = {1, 4};  Transfinite Line{10} = n_duct Using Progression p_duct;
Line(11) = {1, 6};  Transfinite Line{11} = n_duct Using Progression p_duct;
Line(12) = {1, 8};  Transfinite Line{12} = n_duct Using Progression p_duct;

// Duct planes
Line Loop(1) = {12, -6, -5, -11};
Plane Surface(1) = {1};
Line Loop(2) = {9, -8, -7, -12};
Plane Surface(2) = {2};
Line Loop(3) = {10, -2, -1, -9};
Plane Surface(3) = {3};
Line Loop(4) = {11, -4, -3, -10};
Plane Surface(4) = {4};

// Extrude the duct
Extrude {l, 0, 0} {
  Surface{1}; 
  Surface{2}; 
  Surface{3}; 
  Surface{4};
  Layers{n_length}; 
  Recombine; 
}
//+
Physical Surface("FixedWalls") = {16, 6, 12, 11, 7, 18, 15, 19};
//+
Physical Surface("Inlet") = {1, 4, 3, 2};
//+
Physical Surface("Outlet") = {20, 9, 17, 13};
//+
Physical Volume("Volume") = {4, 1, 3, 2};
