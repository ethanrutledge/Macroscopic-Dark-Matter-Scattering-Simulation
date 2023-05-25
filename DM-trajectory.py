# simulating DM trajectory through target box

import numpy as np
import uproot_methods as urm
import matplotlib.pyplot as plt

# -------------------------DETECTOR GEOMETRY--------------------------------
# unsure of actual dimensions of DUNE detector
# temporarily using scale of 66m x 19m x 18m centered around the origin
z_max = 19/2
z_min = -19/2
y_max = 9
y_min = -9
x_max = 33
x_min = -33

# r_max
# must be large enough s.t. no matter disk orientation target is covered
r_max = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2)

# store found entry and exit points
entries = []
exits = []

# iterate for 1000 samples of trajectories
for j in range(1000):
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
    disk = urm.TVector3(np.cos(alpha), np.sin(alpha), 0)
    # disk oriented by theta around y-axis then by phi around z-axis
    disk = disk.rotatey(theta)
    disk = disk.rotatez(phi)
    # resize disk to radius
    disk = r * disk

    # trajectory: <x, y, z> = lambda * [orientation] + [disk]
    orient = [np.sqrt(1 - c ** 2) * np.cos(phi), np.sqrt(1 - c ** 2) * np.sin(phi), c]

    # ---------------------ENTRY AND EXIT POINTS---------------------------
    # to store solved trajectories at each plane
    faces = []
    lamdas = np.zeros(6)
    # to store trajectories that are either entry or exit points
    valid_points = []
    valid_lamdas = []

    # for each of the 6 faces: solve lamda using fixed axis point then subsequently solve trajectory equation
    # face 0 -> xy plane at z_max
    lamdas[0] = (z_max - disk.z) / orient[2]
    faces.append(urm.TVector3(lamdas[0] * orient[0] + disk.x, lamdas[0] * orient[1] + disk.y, z_max))

    # face 1 -> xy plane at z_min
    lamdas[1] = (z_min - disk.z) / orient[2]
    faces.append(urm.TVector3(lamdas[1] * orient[0] + disk.x, lamdas[1] * orient[1] + disk.y, z_min))

    # face 2 -> zy plane at x_max
    lamdas[2] = (x_max - disk.x) / orient[0]
    faces.append(urm.TVector3(x_max, lamdas[2] * orient[1] + disk.y, lamdas[2] * orient[2] + disk.z))

    # face 3 -> zy plane at x_min
    lamdas[3] = (x_min - disk.x) / orient[0]
    faces.append(urm.TVector3(x_min, lamdas[3] * orient[1] + disk.y, lamdas[3] * orient[2] + disk.z))

    # face 4 -> xz plane at y_max
    lamdas[4] = (y_max - disk.y) / orient[1]
    faces.append(urm.TVector3(lamdas[4] * orient[0] + disk.x, y_max, lamdas[4] * orient[2] + disk.z))

    # face 5 -> xz plane at y_min
    lamdas[5] = (y_min - disk.y) / orient[1]
    faces.append(urm.TVector3(lamdas[5] * orient[0] + disk.x, y_min, lamdas[4] * orient[2] + disk.z))

    # check which (if any) points are valid points on the faces of the detector
    for i in range(6):
        if (faces[i].x <= x_max) & (faces[i].x >= x_min) & (faces[i].y <= y_max) & (faces[i].y >= y_min) & (faces[i].z <= z_max) & (faces[i].z >= z_min):
            valid_points.append(faces[i])
            valid_lamdas.append(lamdas[i])

    # print out valid points and lamdas if found
    if len(valid_points) == 2:
        # check which is entry and exit
        if valid_lamdas[0] < valid_lamdas[1]:
            entries.append(valid_points[0])
            exits.append(valid_points[1])
        else:
            entries.append(valid_points[1])
            exits.append(valid_points[0])

# -----------------------DISPLAY PLOT OF ENTRIES AND EXITS--------------------------
# create 3D figure for entries
entries_fig = plt.figure()
entries_ax = entries_fig.add_subplot(projection='3d')
plt.title('Entries')
entries_ax.set_xlabel('X Axis')
entries_ax.set_ylabel('Y Axis')
entries_ax.set_zlabel('Z Axis')
# plot entry points
for j in range(len(entries)):
    entries_ax.scatter(entries[j].x, entries[j].y, entries[j].z)

# Rotate the axes and update
for angle in range(0, 360*4 + 1):
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180
    # Cycle through a full rotation of elevation, then azimuth, roll, and all
    elev = 15
    azim = roll = 0
    if angle <= 360*2:
        azim = angle_norm
    # Update the axis view
    entries_ax.view_init(elev, azim, roll)
    plt.draw()
    plt.pause(.00001)



