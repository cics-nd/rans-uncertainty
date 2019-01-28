# GMSH to OpenFOAM converter
# Distributed by: Notre Dame CICS (MIT Liscense)
# - Associated publication:
# doi: 
# github: https://github.com/cics-nd/rans-uncertainty

export MESH_NAME="LES-mesh" # Name of .geo file
export FOAM_CASE="../LES-OpenFoam" # Directory of OpenFoam case

# Requires the GMSH executable to be in the same directory!
# For getting GMSH see (http://gmsh.info/)
# May have to "chmod +x gmsh" to run it through bash
./gmsh ${MESH_NAME}.geo -3 -o ${MESH_NAME}.msh -algo frontal -algo front3d
# gmshToFoam is a utility that is a part of OpenFOAM
gmshToFoam ${MESH_NAME}.msh -case ${FOAM_CASE}

unset MESH_NAME
unset FOAM_CASE

# WARNING, YOU WILL HAVE TO EDIT BC'S IN constant/polyMesh/boundary AND
# YOUR INITIAL FLOW FILES.
