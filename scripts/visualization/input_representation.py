"""
Generate demonstrative plots for each type of input representation.
"""
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

def setup_ax(ax):
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def render_pointcloud(pcd, filename, title):
    pts = np.asarray(pcd.points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)

    ax.set_title(title)
    ax.set_axis_off()

    setup_ax(ax)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def render_mesh(mesh, filename, title):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        triangles,
        vertices[:, 2],
        color='lightblue',
        edgecolor='gray',
        linewidth=0.1,
        alpha=1.0
    )

    ax.set_title(title)
    setup_ax(ax)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def render_voxel(voxel_grid, filename, title):
    voxel_size = voxel_grid.voxel_size
    origin = voxel_grid.origin

    voxels = voxel_grid.get_voxels()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for v in voxels:
        # convert index tp real world coords
        coord = origin + np.array(v.grid_index) * voxel_size
        ax.bar3d(coord[0], coord[1], coord[2], voxel_size, voxel_size, voxel_size, color='blue', alpha=0.6, shade=True)

    ax.set_title(title)
    ax.set_axis_off()

    setup_ax(ax)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


mesh = o3d.io.read_triangle_mesh(os.path.expanduser("~/bunny.ply"))
mesh.rotate(
    o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2, np.pi/2, 0)),
    center=mesh.get_center()
)
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())


pcd = mesh.sample_points_poisson_disk(5000)
voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 0.05)

render_mesh(mesh, "results/input_representation/mesh.png", "Mesh")
render_pointcloud(pcd, "results/input_representation/pointcloud.png", "Point Cloud")
render_voxel(voxel, "results/input_representation/voxel.png", "Voxel Grid")
