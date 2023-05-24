# attempt 1 at program for simulating DM trajectory through target box

import numpy as np
import uproot_methods as root

# -------------------------DETECTOR GEOMETRY--------------------------------
# unsure of actual dimensions of DUNE detector
# temporarily using scale of 60m x 10m x 10m centered around the origin
z_max = 5
z_min = -5
y_max = 5
y_min = -5
x_max = 30
x_min = -30

# r_max
# must be large enough s.t. no matter disk orientation target is covered
r_max = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2)

# ------------------------DM DISK ORIENTATION------------------------
# sample c uniformly from -1 to 1
c = 2.0 * np.random.random() - 1.0
# convert c to theta(azimuthal angle)
theta = np.arccos(c)

# sample phi(polar angle) uniformly from 0 to 2pi
phi = 2.0 * np.pi * np.random.random()

# sample alpha(angle about disk) uniformly from 0 to 2pi
alpha = 2.0 * np.pi * np.random.random()

# sample probability of radius uniformly from 0 to 1
prob_r = np.random.random()
# calculate radius from probability
r = r_max * np.sqrt(prob_r)

# ----------------------TRAJECTORY CALCULATION--------------------------
# initial disk orientation, fixed to z axis
disk = root.TVector3(np.cos(alpha), np.sin(alpha), 0)

# disk oriented by theta around y-axis then by phi around z-axis
disk = disk.rotatey(theta)
disk = disk.rotatez(phi)

# resize disk to radius
disk = r * disk

# trajectory: <x, y, z> = lambda * [orientation] + [disk]
orient = [np.sqrt(1 - c ** 2) * np.cos(phi), np.sqrt(1 - c ** 2) * np.sin(phi), c]

# ---------------------ENTRY AND EXIT POINTS---------------------------
points = []
lam = np.zeros(6)

# face 1 -> xy plane at z_max
lam[0] = (z_max - disk.z)/orient[2]
f1 = root.TVector3(lam[0] * orient[0] + disk.x, lam[0] * orient[1] + disk.y, z_max)
if (f1[0] >= x_min & f1[0] <= x_max) & (f1[1] >= y_min & f1[1] <= y_max):
    points.append(f1)

# face 2 -> xy plane at z_min
lam[1] = (z_min - disk.z)/orient[2]
f2 = root.TVector3(lam[1] * orient[0] + disk.x, lam[1] * orient[1] + disk.y, z_min)
if (f2[0] >= x_min & f2[0] <= x_max) & (f2[1] >= y_min & f2[1] <= y_max):
    points.append(f2)

# face 3 -> zy plane at x_max
lam[2] = (x_max - disk.x)/orient[0]
f3 = root.TVector3(x_max, lam[2] * orient[1] + disk.y, lam[2] * orient[2] + disk.z)
if (f3[1] >= y_min & f3[1] <= y_max) & (f3[2] >= z_min & f3[2] <= z_max):
    points.append(f3)

# face 4 -> zy plane at x_min
lam[3] = (x_min - disk.x)/orient[0]
f4 = root.TVector3(x_min, lam[3] * orient[1] + disk.y, lam[3] * orient[2] + disk.z)
if (f4[1] >= y_min & f4[1] <= y_max) & (f4[2] >= z_min & f4[2] <= z_max):
    points.append(f4)

# face 5 -> xz plane at y_max
lam[4] = (y_max - disk.y)/orient[1]
f5 = root.TVector3(lam * orient[0] + disk.x, y_max, lam[4] * orient[2] + disk.z)
if(f5[0] >= x_min & f5[0] <= x_max) & (f5[2] >= z_min & f5[2] <= z_max):
    points.append(f5)

# face 6 -> xz plane at y_min
lam[5] = (y_min - disk.y)/orient[1]
f6 = root.TVector3(lam * orient[0] + disk.x, y_min, lam[4] * orient[2] + disk.z)
if(f6[0] >= x_min & f6[0] <= x_max) & (f6[2] >= z_min & f6[2] <= z_max):
    points.append(f6)

if len(points) != 2:
    print('Trajectory does not have valid entry and exit points')
