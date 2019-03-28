from mshr import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Create 2D mesh
r1 = Rectangle(Point(0, 0), Point(1, 1))
r2 = Rectangle(Point(2, 0), Point(3, 1))
domain = r1 + r2
domain.set_subdomain(1, r1)
domain.set_subdomain(3, r2)
mesh = generate_mesh(domain, 18)

x = mesh.coordinates()[:, 0]
y = mesh.coordinates()[:, 1]

a = 0
b = 1
s = 1.5


def denser(x, y):
    return np.array([x, a + (b - a) * ((y - a) / (b - a)) ** s]).T


xy_bar = denser(x, y)
mesh.coordinates()[:] = xy_bar

meshFile = File("cs_mesh.xml")
meshFile << mesh

plot(mesh)
plt.show()
