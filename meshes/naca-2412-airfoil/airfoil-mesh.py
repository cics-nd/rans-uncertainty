'''
// NACA 2412 Airfoil
// Created using Gmsh (http://gmsh.info/)
// Mesh Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: https://doi.org/10.1016/j.jcp.2019.01.021
// github: https://github.com/cics-nd/rans-uncertainty
'''

import pygmsh #See: https://pypi.org/project/pygmsh/
import numpy as np
import meshio

def createMesh():
    # Airfoil coordinates
    airfoil_coordinates = np.genfromtxt('airfoil-profile.data', delimiter=',', skip_header=1, skip_footer=0)
    airfoil_coordinates = np.c_[airfoil_coordinates, np.zeros(airfoil_coordinates.shape[0])]
    print(airfoil_coordinates)
    # Scale airfoil to input coord
    coord = 1.0
    airfoil_coordinates *= coord

    # Instantiate geometry object
    geom = pygmsh.built_in.Geometry()

    # Create polygon for airfoil
    char_length = 5.0e-2 # Resolution
    airfoil = geom.add_polygon(airfoil_coordinates, char_length, make_surface=False)

    # Create surface for numerical domain with an airfoil-shaped hole
    left_dist = 1.0
    right_dist = 3.0
    top_dist = 1.0
    bottom_dist = 1.0
    xmin = airfoil_coordinates[:, 0].min() - left_dist * coord
    xmax = airfoil_coordinates[:, 0].max() + right_dist * coord
    ymin = airfoil_coordinates[:, 1].min() - bottom_dist * coord
    ymax = airfoil_coordinates[:, 1].max() + top_dist * coord
    domainCoordinates = np.array(
        [[xmin, ymin, 0.0], [xmax, ymin, 0.0], [xmax, ymax, 0.0], [xmin, ymax, 0.0]]
    )
    polygon = geom.add_polygon(domainCoordinates, char_length, holes=[airfoil])
    
    geom.add_raw_code("Recombine Surface {{{}}};".format(polygon.surface.id))

    axis = [0, 0, 1]
    geom.extrude(
        polygon,
        translation_axis=axis,
        num_layers=3.0/char_length,
        recombine=True
    )

    ref = 10.525891646546
    points, cells, _, _, _ = pygmsh.generate_mesh(geom, remove_faces=True, gmsh_path='../gmsh', geo_filename='LES-mesh.geo')
    return points, cells


if __name__ == "__main__":

    createMesh()
    # meshio.write_points_cells("airfoil.vtu", *test())
