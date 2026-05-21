"""
Triangular Surface Meshes: Unit Cube, Unit Disk, Unit Cylinder
===============================================================
Uses Plotly for interactive 3-D rendering (preferred) and falls back to
Matplotlib if Plotly is unavailable.

Run:
    pip install plotly numpy
    python mesh_solution.py
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1.  UNIT CUBE
# ─────────────────────────────────────────────────────────────────────────────
# Vertices (given in the problem statement):
#   index  (x,y,z)
#     0    (0,0,0)    1    (0,1,0)    2    (0,1,1)    3    (0,0,1)
#     4    (1,0,0)    5    (1,1,0)    6    (1,1,1)    7    (1,0,1)
cube_pts = np.array(
    [[0,0,0],[0,1,0],[0,1,1],[0,0,1],
     [1,0,0],[1,1,0],[1,1,1],[1,0,1]], dtype=np.double)

# Each face of the cube is a rectangle split into 2 triangles.
# Triangle winding is chosen so that the cross product (b-a)×(c-a) points
# *outward* (away from the cube interior at (0.5, 0.5, 0.5)).
cube_tris = np.array([
    # x=0 face  (outward normal: −x)
    [0, 3, 2], [0, 2, 1],
    # x=1 face  (outward normal: +x)
    [4, 5, 6], [4, 6, 7],
    # y=0 face  (outward normal: −y)
    [0, 4, 7], [0, 7, 3],
    # y=1 face  (outward normal: +y)
    [1, 2, 6], [1, 6, 5],
    # z=0 face  (outward normal: −z)
    [0, 1, 5], [0, 5, 4],
    # z=1 face  (outward normal: +z)
    [3, 7, 6], [3, 6, 2],
], dtype=int)

# cube_pts  : shape (8, 3)   – 8 vertices
# cube_tris : shape (12, 3)  – 12 triangles  (2 per face × 6 faces)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  UNIT DISK  (polar-coordinate construction)
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: work in polar coordinates (r, θ) on a regular rectangular grid
#   r ∈ [0, 1],  θ ∈ [0, 2π)
# then map (r, θ) → (r cosθ, r sinθ, 0).
#
# Vertices:
#   - One centre vertex (r=0).
#   - Nr rings of Nθ points each (r = r_1, …, r_Nr).
# Triangles:
#   - Fan of Nθ triangles from the centre to ring 1.
#   - Annular strip of 2·Nθ triangles between consecutive rings i and i+1.

Nr, Ntheta = 12, 36    # radial subdivisions, angular subdivisions

r_vals = np.linspace(0, 1, Nr + 1)           # r = 0, …, 1  (Nr+1 values)
t_vals = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)

# ── vertices ──
disk_pts = [[0., 0., 0.]]                     # index 0: centre
for ri in r_vals[1:]:                          # rings 1 … Nr
    for tj in t_vals:
        disk_pts.append([ri * np.cos(tj), ri * np.sin(tj), 0.])
disk_pts = np.array(disk_pts)

def d_idx(ring, sector):
    """Global index of vertex (ring, sector) in disk_pts.
    ring 0 → centre (index 0).
    ring ≥ 1 → index  1 + (ring−1)·Nθ + (sector mod Nθ).
    """
    if ring == 0:
        return 0
    return 1 + (ring - 1) * Ntheta + (sector % Ntheta)

# ── triangles ──
disk_tris = []
# central fan: centre → ring 1
for j in range(Ntheta):
    disk_tris.append([0, d_idx(1, j), d_idx(1, j + 1)])

# annular quad-strips: ring i → ring i+1  (each quad → 2 triangles)
for i in range(1, Nr):
    for j in range(Ntheta):
        a = d_idx(i,     j)
        b = d_idx(i,     j + 1)
        c = d_idx(i + 1, j + 1)
        d = d_idx(i + 1, j)
        disk_tris.append([a, b, c])
        disk_tris.append([a, c, d])

disk_tris = np.array(disk_tris, dtype=int)

# disk_pts  : shape (1 + Nr·Nθ,  3)
# disk_tris : shape (Nθ + 2·Nθ·(Nr−1), 3)  =  (Nθ·(2·Nr−1), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  UNIT CYLINDER  C = {x²+y² ≤ 1,  z ∈ [0,1]}
# ─────────────────────────────────────────────────────────────────────────────
# Decomposed into three parts:
#   (a) Bottom disk cap at z=0  (disk mesh, normal −z)
#   (b) Top    disk cap at z=1  (disk mesh, normal +z)
#   (c) Lateral side wall        (tube of Nz × Nθ quads, normal outward radial)

Nth, Nz, NR = 40, 24, 14   # angular, axial, radial subdivisions

r_c = np.linspace(0, 1, NR + 1)
t_c = np.linspace(0, 2*np.pi, Nth, endpoint=False)
z_c = np.linspace(0, 1, Nz + 1)

cyl_pts  = []
cyl_tris = []

def add_cap(z_val, normal_down):
    """Append a filled disk cap at height z_val.
    normal_down=True  → outward normal points −z  (bottom cap).
    normal_down=False → outward normal points +z  (top cap).
    """
    base = len(cyl_pts)
    cyl_pts.append([0., 0., z_val])                # centre
    for ri in r_c[1:]:
        for tj in t_c:
            cyl_pts.append([ri * np.cos(tj), ri * np.sin(tj), z_val])

    def idx(ring, sec):
        if ring == 0:
            return base
        return base + 1 + (ring - 1) * Nth + (sec % Nth)

    # Central fan
    for j in range(Nth):
        if normal_down:
            cyl_tris.append([base, idx(1, j),     idx(1, j + 1)])
        else:
            cyl_tris.append([base, idx(1, j + 1), idx(1, j)])

    # Annular strips
    for i in range(1, NR):
        for j in range(Nth):
            a, b = idx(i,     j), idx(i,     j + 1)
            c, d = idx(i + 1, j + 1), idx(i + 1, j)
            if normal_down:
                cyl_tris.append([a, b, c]);  cyl_tris.append([a, c, d])
            else:
                cyl_tris.append([a, c, b]);  cyl_tris.append([a, d, c])

add_cap(0.0, normal_down=True)   # (a) bottom
add_cap(1.0, normal_down=False)  # (b) top

# (c) Side wall: parametric (θ_j, z_i) → (cos θ_j, sin θ_j, z_i)
side_base = len(cyl_pts)
for zi in z_c:
    for tj in t_c:
        cyl_pts.append([np.cos(tj), np.sin(tj), zi])

def s_idx(zi, sec):
    return side_base + zi * Nth + (sec % Nth)

for i in range(Nz):
    for j in range(Nth):
        a, b = s_idx(i,     j), s_idx(i,     j + 1)
        c, d = s_idx(i + 1, j + 1), s_idx(i + 1, j)
        cyl_tris.append([a, b, c]);  cyl_tris.append([a, c, d])

cyl_pts  = np.array(cyl_pts)
cyl_tris = np.array(cyl_tris, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plotly_mesh(pts, tris, title, color):
    """Return a plotly Figure with a Mesh3d trace."""
    import plotly.graph_objects as go
    x, y, z = pts.T
    i, j, k = tris.T
    fig = go.Figure(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color, opacity=0.85,
        flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
        lightposition=dict(x=1, y=1, z=2)
    ))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data',
                   xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        margin=dict(l=0, r=0, t=40, b=0))
    return fig

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    fig1 = plotly_mesh(cube_pts,  cube_tris,
                       f'Unit Cube – 8 vertices, {len(cube_tris)} triangles',  '#4C9BE8')
    fig2 = plotly_mesh(disk_pts,  disk_tris,
                       f'Unit Disk – {len(disk_pts)} vertices, {len(disk_tris)} triangles', '#F4A261')
    fig3 = plotly_mesh(cyl_pts,   cyl_tris,
                       f'Unit Cylinder – {len(cyl_pts)} vertices, {len(cyl_tris)} triangles', '#57C278')

    fig1.show();  fig2.show();  fig3.show()
    print("Plotly windows opened.")

except ImportError:
    # ── Matplotlib fallback ──────────────────────────────────
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def mpl_mesh(ax, pts, tris, color, alpha, title):
        polys = [[pts[ii] for ii in tri] for tri in tris]
        col = Poly3DCollection(polys, alpha=alpha,
                               facecolor=color, edgecolor='k', linewidth=0.08)
        ax.add_collection3d(col)
        mn, mx = pts.min(0), pts.max(0)
        mid, r = (mn+mx)/2, (mx-mn).max()/2
        ax.set(xlim=(mid[0]-r, mid[0]+r), ylim=(mid[1]-r, mid[1]+r),
               zlim=(mid[2]-r, mid[2]+r),
               xlabel='x', ylabel='y', zlabel='z', title=title)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17,5.5),
                                     subplot_kw={'projection':'3d'})
    mpl_mesh(a1, cube_pts, cube_tris, '#4C9BE8', 0.80,
             f'1. Unit Cube\n8 vertices, {len(cube_tris)} triangles')
    a1.view_init(22, 35)
    mpl_mesh(a2, disk_pts, disk_tris, '#F4A261', 0.90,
             f'2. Unit Disk\n{len(disk_pts)} vertices, {len(disk_tris)} triangles')
    a2.view_init(35, 30)
    mpl_mesh(a3, cyl_pts,  cyl_tris,  '#57C278', 0.70,
             f'3. Unit Cylinder\n{len(cyl_pts)} vertices, {len(cyl_tris)} triangles')
    a3.view_init(20, 40)
    plt.suptitle('Triangular Surface Meshes', fontsize=13)
    plt.tight_layout()
    plt.show()
