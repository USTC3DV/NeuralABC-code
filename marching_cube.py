import numpy as np
from skimage import measure
import trimesh
import torch
import mcubes

def extract_fields(bound_min, bound_max, resolution, query_func, N=64):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    # print("pts:", pts.shape)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    # val = query_func(pts)
                    # print("val:", val.shape)

                    # print("pts:", pts)
                    # print("val<0:", val[val<0])
                    # print("val>0:", val[val>0])
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles



# def get_mesh_from_sdf(
#     udf_func: Callable[[Tensor], Tensor],
#     coords_range: Tuple[float, float],
#     max_dist: float,
#     N: int = 128,    
#     max_batch: int = 2**12,
# ):

#     verts, faces, normals, values = measure.marching_cubes_lewiner(
                                            # volume=value_grid,
                                            # gradient_direction='ascent',
                                            # level=0.)
    # mesh = trimesh.Trimesh(verts, faces, normals, process=False)
    # connected_comp = meshexport.split(only_watertight=False)
    # max_area = 0
    # max_comp = None
    # for comp in connected_comp:
    #     if comp.area > max_area:
    #         max_area = comp.area
    #         max_comp = comp
    # meshexport = max_comp

    return verts, faces