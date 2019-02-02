"""
// Converter Script [Mesh Points from Langley Research Center -> GMSH Format]
// Distributed by: Notre Dame CICS (MIT Liscense)
// - Associated publication:
// doi: https://doi.org/10.1016/j.jcp.2019.01.021
// github: https://github.com/cics-nd/rans-uncertainty
// - Original Flow Reference:
// doi: http://www.doi.org/10.1016/j.ijheatfluidflow.2015.07.006
// url: https://turbmodels.larc.nasa.gov/Other_LES_Data/conv-div-channel20580les.html
"""

import numpy as np

if __name__ == '__main__':

    # Read in surface points from nasa database
    print('Reading data file.')
    surfacePoints = np.genfromtxt('surfacePoints.dat', delimiter=' ', skip_header=3,
                     skip_footer=0, names=["X", "Y"])

    # Seperate out the points on the bump
    bumpPoints = []
    for p0 in surfacePoints:
        if( not p0['Y'] <= 10**(-8)):
            bumpPoints.append([p0['X'],p0['Y']])
    bumpPoints = np.array(bumpPoints)

    print('Writing gmsh points to file.')
    # Write to file to copy into gmsh script
    fh = open('surfacePointsGmsh.dat', 'w')
    for p0 in bumpPoints:
        fh.write('Point(newp) = {%.9f*s, %.9f*s, 0, 1.0};\n' % (p0[0], p0[1]))
    fh.close()

    print('Done')
