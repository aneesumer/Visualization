import numpy as np
import plotly.graph_objects as go

# Vertices (as given in the problem)
points = np.array([
    [0,0,0],[0,1,0],[0,1,1],[0,0,1],   # indices 0–3  (x=0 face)
    [1,0,0],[1,1,0],[1,1,1],[1,0,1]    # indices 4–7  (x=1 face)
], dtype=np.double)

# 12 triangles – 2 per face, wound CCW when viewed from outside
# (so the cross product (b-a)×(c-a) points outward)
triangles = np.array([
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

print("points:\n", points)
print("\ntriangles:\n", triangles)

# Plotly visualisation
x, y, z = points.T
i, j, k = triangles.T

fig = go.Figure(go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='steelblue', opacity=0.85,
    flatshading=True,
    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
    lightposition=dict(x=1, y=1, z=2)
))
fig.update_layout(
    title='Unit Cube – triangulated surface mesh',
    scene=dict(aspectmode='data',
               xaxis_title='x', yaxis_title='y', zaxis_title='z')
)
fig.show()
