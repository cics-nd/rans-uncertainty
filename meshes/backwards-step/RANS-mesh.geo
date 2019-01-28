// Backwards Step Mesh
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: 
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: http://www.doi.org/10.1002/fld.1650170605

inlet_l = 2; // Inlet length
h = 1; // Inlet height
S = 1; // Step size
l_z = 1;
outlet_l = 25*h; //Outlet length

//Number of mesh Points
n_inlet = 20;
n_h = 20;
n_s = 40;
n_outlet = 200;
n_z = 3;

// Progression
p_inlet = 0.916882637;
p_outlet = 0.99;
p_h = 0.9365549739;

// Points
Point(1) = {-inlet_l, 0, 0, 1.0};
Point(2) = {0, 0, 0, 1.0};
Point(3) = {0, -S, 0, 1.0};
Point(4) = {outlet_l, -S, 0, 1.0};
Point(5) = {outlet_l, 0, 0, 1.0};
Point(6) = {outlet_l, h, 0, 1.0};
Point(7) = {0, h, 0, 1.0};
Point(8) = {-inlet_l, h, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {7, 2};
Line(10) = {5, 2};

Line Loop(1) = {8, 1, -9, 7};
Plane Surface(1) = {1};
Line Loop(2) = {9, -10, 5, 6};
Plane Surface(2) = {2};
Line Loop(3) = {10, 2, 3, 4};
Plane Surface(3) = {3};


Transfinite Line {-7, 1} = n_inlet Using Progression p_inlet;
Transfinite Line {6, 10, -3} = n_outlet Using Progression p_outlet;
Transfinite Line {8, 9, -5} = n_h Using Progression p_h;
Transfinite Line {2, 4} = n_s Using Progression 1;
//+
Transfinite Surface {3};
Transfinite Surface {2};
Transfinite Surface {1};
Recombine Surface "*";
//+
Extrude {0, 0, l_z} {
  Surface{1}; 
  Surface{2}; 
  Surface{3};
  Layers{n_z}; 
  Recombine; 
}
//+
Physical Surface("FixedWalls") = {23, 67, 71, 53, 31};
//+
Physical Surface("Inlet") = {19};
//+
Physical Surface("Outlet") = {49, 75};
//+
Physical Surface("SideWall_1") = {32, 54, 76};
//+
Physical Surface("SideWall_2") = {2, 3, 1};
//+
Physical Volume("Volume") = {1, 2, 3};
