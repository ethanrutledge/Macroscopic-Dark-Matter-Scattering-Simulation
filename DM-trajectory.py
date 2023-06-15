# simulating DM trajectory through DUNE detector

import numpy as np
import uproot_methods as urm
import matplotlib.pyplot as plt

# -------------------------DETECTOR GEOMETRY--------------------------------
# based on dimensions of DUNE detector -- sourced from TDR doc
# using scale of 12 x 14 x 58.2m centered around the origin
x_scale = 12
y_scale = 14
z_scale = 58.2

x_max = x_scale / 2
x_min = -x_scale / 2
y_max = y_scale / 2
y_min = -y_scale / 2
z_max = z_scale / 2
z_min = -z_scale / 2

# r_max
# must be large enough s.t. no matter disk orientation target is covered
r_max = np.sqrt((x_scale ** 2) + (y_scale ** 2) + (z_scale ** 2))/2

# ---------------------SAMPLE TRAJECTORIES-----------------------------------
# store found entry and exit points
entries = []
exits = []

# store c and phis to check distribution
cs = []
phis = []

# iterate for some number of samples of trajectories
for j in range(100000):
    # ------------------------DM DISK ORIENTATION------------------------
    # sample c uniformly from -1 to 1
    c = 2.0 * np.random.random() - 1.0
    cs.append(c)
    # convert c to theta(azimuthal angle)
    theta = np.arccos(c)

    # sample phi(polar angle) uniformly from 0 to 2pi
    phi = 2.0 * np.pi * np.random.random()
    phis.append(phi)

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
    lambdas = np.zeros(6)
    # to store trajectories that are either entry or exit points
    valid_points = []
    valid_lambdas = []

    # for each of the 6 faces: solve lamda using fixed axis point then subsequently solve trajectory equation
    # face 0 -> xy plane at z_max
    lambdas[0] = (z_max - disk.z) / orient[2]
    faces.append(urm.TVector3(lambdas[0] * orient[0] + disk.x, lambdas[0] * orient[1] + disk.y, z_max))

    # face 1 -> xy plane at z_min
    lambdas[1] = (z_min - disk.z) / orient[2]
    faces.append(urm.TVector3(lambdas[1] * orient[0] + disk.x, lambdas[1] * orient[1] + disk.y, z_min))

    # face 2 -> zy plane at x_max
    lambdas[2] = (x_max - disk.x) / orient[0]
    faces.append(urm.TVector3(x_max, lambdas[2] * orient[1] + disk.y, lambdas[2] * orient[2] + disk.z))

    # face 3 -> zy plane at x_min
    lambdas[3] = (x_min - disk.x) / orient[0]
    faces.append(urm.TVector3(x_min, lambdas[3] * orient[1] + disk.y, lambdas[3] * orient[2] + disk.z))

    # face 4 -> xz plane at y_max
    lambdas[4] = (y_max - disk.y) / orient[1]
    faces.append(urm.TVector3(lambdas[4] * orient[0] + disk.x, y_max, lambdas[4] * orient[2] + disk.z))

    # face 5 -> xz plane at y_min
    lambdas[5] = (y_min - disk.y) / orient[1]
    faces.append(urm.TVector3(lambdas[5] * orient[0] + disk.x, y_min, lambdas[5] * orient[2] + disk.z))

    # check which (if any) points are valid points on the faces of the detector
    for i in range(6):
        if (faces[i].x <= x_max) & (faces[i].x >= x_min) & (faces[i].y <= y_max) & (faces[i].y >= y_min) & (
                faces[i].z <= z_max) & (faces[i].z >= z_min):
            valid_points.append(faces[i])
            valid_lambdas.append(lambdas[i])

    # add valid points to entries and exit lists if found
    if len(valid_points) == 2:
        # check which is entry and exit
        if valid_lambdas[0] < valid_lambdas[1]:
            entries.append(valid_points[0])
            exits.append(valid_points[1])
        else:
            entries.append(valid_points[1])
            exits.append(valid_points[0])

    # ----------------------SAMPLE SPEED------------------------------------
    v_bar = 10 ** -3
    v_x = np.random.normal(0, v_bar/np.sqrt(3))
    v_y = np.random.normal(0, v_bar/np.sqrt(3))
    v_z = np.random.normal(0, v_bar/np.sqrt(3))

    v = np.sqrt((v_x ** 2) + (v_y ** 2) + (v_z ** 2))

    # f_v = 3 * np.sqrt(6 / np.pi) * (1 / v_bar ** 3) * (v ** 2) * np.log((-3 * v ** 2) / (2 * v_bar ** 2))


# -----------------------PLOT FUNCTIONS-----------------------------------
def single_face_hex_dist_plot(points, face, point_type):
    """
    :param points: either entries or exits list
    :param face: string of which face dist plot of
    :param point_type: string of either entry or exit
    :return: creates a 2d hexagonal distribution plot of entry or exit points on specific face
    """
    if 'z' in face:
        x_dist = []
        y_dist = []
        x_lim = x_min, x_max
        y_lim = y_min, y_max
        name = "distribution of " + point_type + " points on " + face + " face"
        fig, ax = plt.subplots()
        if face == 'z_max':
            for i in range(len(points)):
                if points[i].z == z_max:
                    x_dist.append(points[i].x)
                    y_dist.append(points[i].y)
        elif face == 'z_min':
            for i in range(len(points)):
                if points[i].z == z_min:
                    x_dist.append(points[i].x)
                    y_dist.append(points[i].y)
        hb = ax.hexbin(x_dist, y_dist, gridsize=10, cmap='inferno')
        ax.set(xlim=x_lim, ylim=y_lim)
    elif 'x' in face:
        z_dist = []
        y_dist = []
        z_lim = z_min, z_max
        y_lim = y_min, y_max
        name = "distribution of " + point_type + " points on " + face + " face"
        fig, ax = plt.subplots()
        if face == 'x_min':
            for i in range(len(points)):
                if points[i].x == x_min:
                    z_dist.append(points[i].z)
                    y_dist.append(points[i].y)
        elif face == 'x_max':
            for i in range(len(points)):
                if points[i].x == x_max:
                    z_dist.append(points[i].z)
                    y_dist.append(points[i].y)
        hb = ax.hexbin(z_dist, y_dist, gridsize=10, cmap='inferno')
        ax.set(xlim=z_lim, ylim=y_lim)
    elif 'y' in face:
        x_dist = []
        z_dist = []
        x_lim = x_min, x_max
        z_lim = z_min, z_max
        name = "distribution of " + point_type + " points on " + face + " face"
        fig, ax = plt.subplots()
        if face == 'y_max':
            for i in range(len(points)):
                if points[i].z == z_max:
                    x_dist.append(points[i].x)
                    z_dist.append(points[i].y)
        elif face == 'y_min':
            for i in range(len(points)):
                if points[i].z == z_min:
                    x_dist.append(points[i].x)
                    z_dist.append(points[i].y)
        hb = ax.hexbin(x_dist, z_dist, gridsize=10, cmap='inferno')
        ax.set(xlim=x_lim, ylim=z_lim)
    ax.set_title(name)
    cb = fig.colorbar(hb, ax=ax, label='counts')
    plt.show()


def rotating_plot(entries, exits):
    """
    STILL NEED RESOLVED AXIS LENGTHS AND LEGEND
    :param entries: entry points list
    :param exits: exit points list
    :return: creates 3D rotating plot of entry(red) and exit(blue) points of detector
    """
    # create 3D figure for entries and exits
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title('Entry and Exit Points')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    # plt.axis([x_min, x_max, y_min, y_max, z_min, z_max]) --------------------- still need resolved
    # plot entry and exit points
    for j in range(len(entries)):
        ax.scatter(entries[j].x, entries[j].y, entries[j].z, c="red", marker=".", label='entry')
        ax.scatter(exits[j].x, exits[j].y, exits[j].z, c="blue", marker=".", label='exit')
    # Rotate the axes and update
    for angle in range(0, 360 * 4 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180
        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = 15
        azim = roll = 0
        if angle <= 360 * 2:
            azim = angle_norm
        # Update the axis view
        ax.view_init(elev, azim, roll)
        plt.draw()
        plt.pause(.00001)


def all_face_hex_dist_plot(points, point_type):
    """
    NEED TO NORMALIZE COLOR SCALE
    :param points: either entry or exit points list
    :param point_type: string of either entry or exit
    :return: creates figure of hex dist plot for all 6 faces of detector
    """
    x_lim = x_min, x_max
    y_lim = y_min, y_max
    z_lim = z_min, z_max

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    x0 = []
    y0 = []
    x1 = []
    y1 = []
    z2 = []
    y2 = []
    z3 = []
    y3 = []
    x4 = []
    z4 = []
    x5 = []
    z5 = []

    for i in range(len(points)):
        if points[i].z == z_max:
            x0.append(points[i].x)
            y0.append(points[i].y)
        elif points[i].z == z_min:
            x1.append(points[i].x)
            y1.append(points[i].y)
        elif points[i].x == x_max:
            z2.append(points[i].z)
            y2.append(points[i].y)
        elif points[i].x == x_min:
            z3.append(points[i].z)
            y3.append(points[i].y)
        elif points[i].y == y_max:
            x4.append(points[i].x)
            z4.append(points[i].z)
        elif points[i].y == y_min:
            x5.append(points[i].x)
            z5.append(points[i].z)

    hb = axs[0, 0].hexbin(x0, y0, gridsize=10, cmap='inferno')
    axs[0, 0].set(xlim=x_lim, ylim=y_lim)
    axs[0, 0].set_title('z_max')
    cb = fig.colorbar(hb, ax=axs[0, 0], label='counts')

    hb = axs[1, 0].hexbin(x1, y1, gridsize=10, cmap='inferno')
    axs[1, 0].set(xlim=x_lim, ylim=y_lim)
    axs[1, 0].set_title('z_min')
    cb = fig.colorbar(hb, ax=axs[1, 0], label='counts')

    hb = axs[0, 1].hexbin(z2, y2, gridsize=10, cmap='inferno')
    axs[0, 1].set(xlim=z_lim, ylim=y_lim)
    axs[0, 1].set_title('x_max')
    cb = fig.colorbar(hb, ax=axs[0, 1], label='counts')

    hb = axs[1, 1].hexbin(z3, y3, gridsize=10, cmap='inferno')
    axs[1, 1].set(xlim=z_lim, ylim=y_lim)
    axs[1, 1].set_title('x_min')
    cb = fig.colorbar(hb, ax=axs[1, 1], label='counts')

    hb = axs[0, 2].hexbin(x4, z4, gridsize=10, cmap='inferno')
    axs[0, 2].set(xlim=x_lim, ylim=z_lim)
    axs[0, 2].set_title('y_max')
    cb = fig.colorbar(hb, ax=axs[0, 2], label='counts')

    hb = axs[1, 2].hexbin(x5, z5, gridsize=10, cmap='inferno')
    axs[1, 2].set(xlim=x_lim, ylim=z_lim)
    axs[1, 2].set_title('y_min')
    cb = fig.colorbar(hb, ax=axs[1, 2], label='counts')

    plt.savefig('allPlotsHexDist' + point_type)


def one_dim_hist_all_faces(points, point_type):
    """
    :param points: either entry or exit list
    :param point_type: string of either "entry" or "exit"
    :return: 12 histograms, 2 per face of point distribution & prints total points on each face
    """
    x_lim = x_min, x_max
    y_lim = y_min, y_max
    z_lim = z_min, z_max

    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(40, 10))

    x0 = []
    y0 = []
    x1 = []
    y1 = []
    z2 = []
    y2 = []
    z3 = []
    y3 = []
    x4 = []
    z4 = []
    x5 = []
    z5 = []

    for i in range(len(points)):
        if points[i].z == z_max:
            x0.append(points[i].x)
            y0.append(points[i].y)
        elif points[i].z == z_min:
            x1.append(points[i].x)
            y1.append(points[i].y)
        elif points[i].x == x_max:
            z2.append(points[i].z)
            y2.append(points[i].y)
        elif points[i].x == x_min:
            z3.append(points[i].z)
            y3.append(points[i].y)
        elif points[i].y == y_max:
            x4.append(points[i].x)
            z4.append(points[i].z)
        elif points[i].y == y_min:
            x5.append(points[i].x)
            z5.append(points[i].z)

    axs[0, 0].hist(x0, range=x_lim)
    axs[0, 0].set_title('face 0 @z_max x dim')
    axs[1, 0].hist(y0, range=y_lim)
    axs[1, 0].set_title('face 0 @z_max y dim')

    axs[0, 1].hist(x1, range=x_lim)
    axs[0, 1].set_title('face 1 @z_min x dim')
    axs[1, 1].hist(y1, range=y_lim)
    axs[1, 1].set_title('face 1 @z_min y dim')

    axs[0, 2].hist(z2, range=z_lim)
    axs[0, 2].set_title('face 2 @x_max z dim')
    axs[1, 2].hist(y2, range=y_lim)
    axs[1, 2].set_title('face 2 @x_max y dim')

    axs[0, 3].hist(z3, range=z_lim)
    axs[0, 3].set_title('face 3 @x_min z dim')
    axs[1, 3].hist(y3, range=y_lim)
    axs[1, 3].set_title('face 3 @x_min y dim')

    axs[0, 4].hist(x4, range=x_lim)
    axs[0, 4].set_title('face 4 @y_max x dim')
    axs[1, 4].hist(z4, range=z_lim)
    axs[1, 4].set_title('face 4 @y_max z dim')

    axs[0, 5].hist(x5, range=x_lim)
    axs[0, 5].set_title('face 5 @y_min x dim')
    axs[1, 5].hist(z5, range=z_lim)
    axs[1, 5].set_title('face 5 @y_min z dim')

    plt.savefig('1DhistAllFaces' + point_type)

    print(point_type)
    print('face 0 total points: ', len(x0) + len(y0))
    print('face 1 total points: ', len(x1) + len(y1))
    print('face 2 total points: ', len(z2) + len(y2))
    print('face 3 total points: ', len(z3) + len(y3))
    print('face 4 total points: ', len(x4) + len(z4))
    print('face 5 total points: ', len(x5) + len(z5))


def costheta_phi_hist(cs, phis):
    """
    :param cs: set of all sampled c values c = cosTheta
    :param phis: set of all sampled phi values
    :return: histograms of both cosTheta and Phi
    """

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].hist(cs)
    axs[0].set_title('cosTheta')
    axs[1].hist(phis)
    axs[1].set_title('phi')
    plt.savefig('cosThetaPhiDist')


# ----------------------FUNCTION TEST CALLS------------------------
# single_face_hex_dist_plot(exits, "y_min", "exit")
# single_face_hex_dist_plot(entries, "z_max", "entry")
# rotating_plot(entries, exits)
# all_face_hex_dist_plot(entries, 'entry')
# all_face_hex_dist_plot(exits, 'exit')
one_dim_hist_all_faces(entries, "entry")
one_dim_hist_all_faces(exits, "exit")
# costheta_phi_hist(cs, phis)
