import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi, voronoi_plot_2d
# plot
import matplotlib.pyplot as plt

def voronoi_finite_polygons_2d(vor, radius):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    #if radius is None:
        #radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



def natural_neighbor_interpolation(samples, npt):
    samples2d = samples[:, 0:2]

    radius = 10
    vor = Voronoi(samples2d)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius)
    

    vor_poly = []
    for region in regions:
        vor_poly.append(Polygon(vertices[region]))

    new_samples = np.vstack((samples2d, npt))

    vor_new = Voronoi(new_samples)
    regions, vertices = voronoi_finite_polygons_2d(vor_new, radius)

    dst_poly = Polygon(vertices[regions[-1]])
    pts_region_vertives = vertices[regions[-1]]
    
    intersect_index = []
    intersect_area = []
    for i in range(len(vor_poly)):
        if dst_poly.intersects(vor_poly[i]):
            intersect_index.append(i)
            intersect_area.append(dst_poly.intersection(vor_poly[i]).area)

    nn = samples[intersect_index, 2]
    wi = [intersect_area[i] / dst_poly.area for i in range(len(intersect_area))]
    rst_p = np.dot(wi, nn)
    
    
    return intersect_index, pts_region_vertives, rst_p


##
filepath = 'data.txt'
outputfile = 'result_' + filepath

KnownPts = []
P = [65, 137]
P0 = np.array([[65, 137]]) # target point
P0_3D = np.array([[65, 137, 0]])
Dist = []
W = []
W_sum = 0
p = 0
P_Z = 0
num = 0 # number of points
radius = 10

with open(filepath) as fp:
    for line in fp:
        num = num + 1
        values = line.split()
        KnownPts.append([float(values[0]), float(values[1]), float(values[2])])


KnownPts_array = np.asarray(KnownPts, dtype=np.float32)


vor1 = Voronoi(KnownPts_array[:,0:2])
voronoi_plot_2d(vor1)
plt.xlim(60, 80)
plt.ylim(120, 155)
plt.axis('equal')
plt.savefig('Original Vonoronoi.png')
plt.show()

plt.close()
# new:
KnownPts_array2 = np.concatenate((KnownPts_array, P0_3D))

vor = Voronoi(KnownPts_array2[:,0:2])
voronoi_plot_2d(vor)
plt.xlim(60, 80)
plt.ylim(120, 155)
plt.axis('equal')
plt.savefig('New Vonoronoi.png')
plt.show()

plt.close()

# Natural Neighbor:
intersect_index, pts_region_vertives, rst_p = natural_neighbor_interpolation(KnownPts_array, P0)

samples2d = KnownPts_array[:, 0:2]
vor = Voronoi(samples2d)
regions, vertices = voronoi_finite_polygons_2d(vor, radius)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.ishold = lambda: True  # Work-around for Matplotlib 3.0.0 incompatibility

i = 0
for region in regions:
    x = vertices[region,0]
    x = np.append(x, [x[0]], axis = 0)
    
    y = vertices[region,1]
    y = np.append(y, [y[0]], axis = 0)
    
    ax.plot(x, y, 'b:', linewidth=2)
    ax.plot(samples2d[i, 0], samples2d[i, 1], 'b+', markersize=8)
    i = i + 1

# Plot the region of the intepolated point:
x = pts_region_vertives[:,0]
x = np.append(x, [x[0]], axis = 0)
y = pts_region_vertives[:,1]
y = np.append(y, [y[0]], axis = 0)
ax.plot(x, y, 'r-', linewidth=2)
ax.plot(P0[0,0], P0[0,1], 'r*',markersize=10)
ax.axis('equal')
fig.savefig('Voronoi_finite_polygon.png')