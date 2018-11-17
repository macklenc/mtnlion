=================
Using COMSOL Data
=================

To convert mesh from COMSOL to FEniCS is a multistep process. The method that has worked in the past is to use FEConv to convert from .mphtxt that comsol exports to gmsh .msh format. Then open the .msh in gmsh and expand Mesh to select the dimensionality of the mesh. Then save over the .msh file. Finally, use dolfin-convert to convert from .msh to .xml.
