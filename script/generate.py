import numpy as np
import torch
import openmesh as om
import trimesh
import os, sys
import torch.nn.functional as F
from torch import Tensor
import open3d as o3d 
import time
import argparse

from kaolin.ops.mesh.check_sign import check_sign
from scipy.spatial import KDTree
import pickle

sys.path.append("..")

from smpl_pytorch.body_models import SMPL
from models import skip_network
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from siren.script.siren_file import SirenNet, SirenWrapper
from meshudf.meshudf import get_mesh_from_udf
from marching_cube import extract_geometry
from models.lbs_mlp import lbs_pbs, get_embedder
import cv2



def om_loadmesh(path):
    mesh = om.read_trimesh(path)
    v = torch.from_numpy(np.array(mesh.points())).float().cuda()
    f = torch.tensor(torch.from_numpy(np.array(mesh.face_vertex_indices())), dtype=torch.int64).cuda()

    return v, f

def om_loadmesh_np(path):
    mesh = om.read_trimesh(path)
    v = mesh.points()
    f = mesh.face_vertex_indices()
    mesh.request_vertex_normals()   
    mesh.update_vertex_normals()
    normals = mesh.vertex_normals()
    normals_array = np.array(normals)

    return v, f, normals_array


class Cloth_Body_Model:
    def __init__(self, sub) -> None:

        self.root = "/data5/chh/NeuralABC"
        time_sub = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.save = os.path.join(self.root, "res", sub, time_sub)

        self.mesh_sub = self.save
        os.makedirs(self.mesh_sub, exist_ok=True)

        print("sub:", sub)
        print("time_sub:", time_sub)
        print("mesh_sub:", self.mesh_sub)

        self.model_lbs_shape = skip_network.skip_connection(d_in=10+3, d_out=3, width=256, depth=8, skip_layer=[4]).cuda().eval()
        lbs_shape_checkpoints_dir = os.path.join(self.root, 'checkpoints/lbs_shape.pth')
        self.model_lbs_shape.load_state_dict(torch.load(lbs_shape_checkpoints_dir))

        self.model_lbs = skip_network.skip_connection(d_in=3, d_out=24, width=256, depth=8, skip_layer=[4]).cuda().eval()
        lbs_checkpoints_dir = os.path.join(self.root, 'checkpoints/lbs.pth')
        self.model_lbs.load_state_dict(torch.load(lbs_checkpoints_dir))

        self.coords_encoder = CoordsEncoder()
        self.smpl_model_dir = os.path.join(self.root, 'smpl_pytorch')

        extra_dir = os.path.join(self.root, 'extra-data')
        data_body = np.load(os.path.join(extra_dir, 'body_info_f.npz'))
        self.tfs_c_inv = torch.FloatTensor(data_body['tfs_c_inv']).cuda()

        self.smpl_server = SMPL(model_path=self.smpl_model_dir, gender='f').cuda()

        self.init_body_model()
        self.init_cloth_model()
        self.init_cloth_code()
        self.init_body_code()

    def cloth_skinning(self, x, w, tfs, tfs_c_inv):
        """Linear blend skinning
        Args:
            x (tensor): deformed points. shape: [B, N, D]
            w (tensor): conditional input. [B, N, J]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            tfs_c_inv (tensor): bone transformation matrices. shape: [J, D+1, D+1]
        Returns:
            x (tensor): skinned points. shape: [B, N, D]
        """

        tfs = torch.einsum('bnij,njk->bnik', tfs, tfs_c_inv)

        x_h = F.pad(x, (0, 1), value=1.0)
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

        return x_h[:, :, :3]     


    def body_skinning(self, x, w, tfs, tfs_c_inv):
        """Linear blend skinning
        Args:
            x (tensor): deformed points. shape: [B, N, D]
            w (tensor): conditional input. [B, N, J]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            tfs_c_inv (tensor): bone transformation matrices. shape: [J, D+1, D+1]
        Returns:
            x (tensor): skinned points. shape: [B, N, D]
        """
        # tfs = torch.einsum('bnij,njk->bnik', tfs, tfs_c_inv)

        x_h = F.pad(x, (0, 1), value=1.0)
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

        return x_h[:, :, :3]     


    def init_body_model(self):
        ckpt_path = os.path.join(self.root, "./checkpoints/body.pt")
        ckpt = torch.load(ckpt_path)
        hidden_dim = 512
        num_hidden_layers = 5
        body_latent_size = 10     


        sirennet = SirenNet(
            dim_in = 63,                        # input dimension, ex. 3d coor
            dim_hidden = hidden_dim,                  # hidden dimension
            dim_out = 1,                       # output dimension, ex. rgb value
            num_layers = num_hidden_layers,                    # number of layers
            w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

        body_decoder = SirenWrapper(
            sirennet,
            latent_dim = body_latent_size,
            input_dim = 63,
        )

        body_decoder.load_state_dict(ckpt["decoder"])
        self.body_decoder = body_decoder.cuda()
        self.body_decoder.eval()


    def init_cloth_model(self):
        ckpt_path = os.path.join(self.root, "./checkpoints/399.pt")
        ckpt = torch.load(ckpt_path)
        hidden_dim = 512
        num_hidden_layers = 5
        cloth_latent_size = 128

        cloth_decoder = CbnDecoder(
            self.coords_encoder.out_dim,
            cloth_latent_size,
            hidden_dim,
            num_hidden_layers,
        )    
        self.udf_max_dist =  0.1
        cloth_decoder.load_state_dict(ckpt["decoder"])
        self.cloth_decoder = cloth_decoder.cuda()   
        self.cloth_decoder.eval()

    def init_body_code(self):
        body_code_path = os.path.join(self.root, "./checkpoints/body_code.pt")
        self.body_code = torch.load(body_code_path).cuda()
    
    def init_cloth_code(self):
        cloth_code_path = os.path.join(self.root, "./checkpoints/all.pt")
        self.cloth_code = torch.load(cloth_code_path).cuda()

    def get_cloth_code(self, cloth_num=5):
        cloth_code = self.cloth_code[cloth_num].reshape(1,128)
        return cloth_code
    
    def get_body_code(self, body_num=1):
        body_code = self.body_code[body_num].reshape(1,10)
        return body_code


    def reconstruct_cloth(self, cloth_code, num):
        def udf_func(c: Tensor) -> Tensor:

            c = self.coords_encoder.encode(c.unsqueeze(0))
            p = self.cloth_decoder(c, cloth_code).squeeze(0)
            p = torch.sigmoid(p)
            p = (1 - p) * self.udf_max_dist
            return p 

        v, t = get_mesh_from_udf(
            udf_func,
            coords_range=(-1, 1),
            max_dist=self.udf_max_dist,
            N=512,
            max_batch=2**16,
            differentiable=False,
        )

        mesh_path = os.path.join(self.mesh_sub, "cloth_{}.obj".format(num))
        cloth_color_mesh = trimesh.Trimesh(v.squeeze().cpu().numpy(), t.squeeze().cpu().numpy())
        cloth_colors = np.ones((len(cloth_color_mesh.faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
        cloth_color_mesh.visual.face_colors = cloth_colors
        cloth_color_mesh.export(mesh_path)

        print("Save to:", mesh_path)

        return v, t, mesh_path
    
    def reconstruct_body(self, body_code, name):
        def udf_func(c: Tensor) -> Tensor:
            c = self.coords_encoder.encode(c.unsqueeze(0))
            p = -self.body_decoder(c, body_code).squeeze(0)
            return p

        v, t = extract_geometry(
            bound_min = torch.tensor([-1., -1.5, -0.5]).cuda(),
            bound_max = torch.tensor([1.0, 1.0, 0.5]).cuda(),
            resolution = 512,
            threshold = 0.0,
            query_func = udf_func,
        )

        mesh_path = os.path.join(self.mesh_sub, "body_{}.obj".format(name))
        body_color_mesh = trimesh.Trimesh(v, t)
        body_colors = np.ones((len(body_color_mesh.faces), 4))*np.array([255, 235, 205, 200])[np.newaxis,:]
        body_color_mesh.visual.face_colors = body_colors
        body_color_mesh.export(mesh_path)

        print("Save to:", mesh_path)

        return v, t, mesh_path


    def deform_cloth_collision(self, body_path, cloth_path, name):

        epsilon = 1e-3

        body_v, body_f = om_loadmesh(body_path)
        cloth_v, cloth_f = om_loadmesh(cloth_path)
        body_v = body_v.reshape(1,-1,3)
        cloth_v = cloth_v.reshape(1,-1,3)
        body_v_np, body_f_np, body_n_np = om_loadmesh_np(body_path)
        cloth_v_np, cloth_f_np, cloth_n_np = om_loadmesh_np(cloth_path)

        body_kdtree = KDTree(body_v_np)
        sign = check_sign(body_v, body_f, cloth_v)[0].cpu().numpy()

        for i in range(len(sign)):
            if sign[i] == True:
                cloth_vertex = cloth_v_np[i]
                _, nearest_vertex_index = body_kdtree.query(cloth_vertex)
                v_d = body_n_np[nearest_vertex_index]

                vec_norm = np.linalg.norm(v_d)
                unit_norm = v_d/vec_norm
                cloth_v_np[i] = body_v_np[nearest_vertex_index] + epsilon * unit_norm

        mesh_path = os.path.join(self.mesh_sub, "cloth_collision_{}.ply".format(name))
        cloth_color_mesh = trimesh.Trimesh(cloth_v_np, cloth_f_np)
        cloth_colors = np.ones((len(cloth_color_mesh.faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
        cloth_color_mesh.visual.face_colors = cloth_colors
        cloth_color_mesh.export(mesh_path)
        print("Save to:", mesh_path)

        return torch.from_numpy(cloth_v_np).reshape(1, -1, 3).cuda(), mesh_path

    def save_obj(self, v, f, name):
        mesh_path = os.path.join(self.mesh_sub, name)
        body_color_mesh = trimesh.Trimesh(v.detach().squeeze().cpu().numpy(), f.detach().squeeze().cpu().numpy())
        body_colors = np.ones((len(body_color_mesh.faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
        body_color_mesh.visual.face_colors = body_colors
        body_color_mesh.export(mesh_path)
        print("Save to:", mesh_path)       

    def body_lbs(self, points, beta, pose, f, name, trans=None):
        
        points = torch.tensor(points).reshape(1,-1,3).float().cuda()
        lbs_weight = self.model_lbs(points)
        lbs_weight = lbs_weight.softmax(dim=-1)
        output_smpl = self.smpl_server(betas=beta, body_pose=pose[:, 3:], global_orient=pose[:, :3], return_verts=True)
        tfs = output_smpl.T
        if trans==None:
            point_skinning = self.body_skinning(points, lbs_weight, tfs, self.tfs_c_inv) #+ trans
        else:
            point_skinning = self.body_skinning(points, lbs_weight, tfs, self.tfs_c_inv) + trans

        mesh_path = os.path.join(self.mesh_sub, "body_pose_{}.obj".format(name))
        body_color_mesh = trimesh.Trimesh(point_skinning.detach().squeeze().cpu().numpy(), f)
        body_colors = np.ones((len(body_color_mesh.faces), 4))*np.array([255, 235, 205, 200])[np.newaxis,:]
        body_color_mesh.visual.face_colors = body_colors
        body_color_mesh.export(mesh_path)
        print("Save to:", mesh_path)
        return mesh_path

    def cloth_lbs(self, points, beta, pose, f, name, trans=None):
        points = points.reshape(1, -1, 3)
        lbs_weight = self.model_lbs(points)
        lbs_weight = lbs_weight.softmax(dim=-1)
        output_smpl = self.smpl_server(betas=beta, body_pose=pose[:, 3:], global_orient=pose[:, :3], return_verts=True)
        tfs = output_smpl.T
        if trans==None:
            point_skinning = self.cloth_skinning(points, lbs_weight, tfs, self.tfs_c_inv)
        else:
            point_skinning = self.cloth_skinning(points, lbs_weight, tfs, self.tfs_c_inv) + trans
        
        mesh_path = os.path.join(self.mesh_sub, "cloth_pose_{}.obj".format(name))
        cloth_color_mesh = trimesh.Trimesh(point_skinning.detach().squeeze().cpu().numpy(), f.squeeze().cpu().numpy())
        cloth_colors = np.ones((len(cloth_color_mesh.faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
        cloth_color_mesh.visual.face_colors = cloth_colors
        cloth_color_mesh.export(mesh_path)
        print("Save to:", mesh_path)

        return mesh_path

    def deform_body(self, body_deform_code, vertices_body_T, f, name, p=None):
        num_v = vertices_body_T.shape[0]
        
        body_code_input = body_deform_code.reshape(1, -1, 10).repeat(1, num_v, 1) 
        points = torch.from_numpy(vertices_body_T).reshape(1, -1, 3).float().cuda()
        input_lbs_shape = torch.cat((points, body_code_input), dim=-1) 
        delta_shape_pred = self.model_lbs_shape(input_lbs_shape, False)
        body_deform = delta_shape_pred + points

        mesh_path = os.path.join(self.mesh_sub, "body_{}.obj".format(name))
        cloth_color_mesh = trimesh.Trimesh(body_deform.detach().squeeze().cpu().numpy(), f)
        cloth_colors = np.ones((len(cloth_color_mesh.faces), 4))*np.array([255, 235, 205, 200])[np.newaxis,:]
        cloth_color_mesh.visual.face_colors = cloth_colors
        cloth_color_mesh.export(mesh_path)
        print("Save to:", mesh_path)

        return body_deform, mesh_path

    def deform_cloth(self, body_code, vertices_garment_T, f, name, p=None):
        num_v = vertices_garment_T.shape[0]
        body_code_input = body_code.reshape(1, -1, 10).repeat(1, num_v, 1) 
        points = vertices_garment_T.reshape(1, -1, 3)
        input_lbs_shape = torch.cat((points, body_code_input), dim=-1) 
        delta_shape_pred = self.model_lbs_shape(input_lbs_shape, False)
        garment_deform = delta_shape_pred + points

        mesh_path = os.path.join(self.mesh_sub, "cloth_{}.obj".format(name))
        cloth_color_mesh = trimesh.Trimesh(garment_deform.detach().squeeze().cpu().numpy(), f.squeeze().cpu().numpy())
        cloth_colors = np.ones((len(cloth_color_mesh.faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
        cloth_color_mesh.visual.face_colors = cloth_colors
        cloth_color_mesh.export(mesh_path)
        print("Save to:", mesh_path)

        return garment_deform, mesh_path

    def get_cloth_code(self, cloth_num):
        
        cloth_code = self.cloth_code[cloth_num].reshape(1,128)

        return cloth_code
    
    def get_body_code(self, body_num):
        
        body_code = self.body_code[body_num].reshape(1, 10)

        return body_code

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="save path")
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    model = Cloth_Body_Model(args.save)
    extra_dir = '../extra-data'
    poses = torch.load(os.path.join(extra_dir, 'pose-sample.pt'))
    pose = poses[100].cuda().reshape(1,-1)

    beta = torch.tensor([-2.71335335, 0.13721677, -0.96557967, 2.15569137, -2.20862524,  2.02085724, 1.32704986, -0.18951267, -2.37554491,  0.31818445]).reshape(1, 10).float().cuda()
    
    cloth_code = model.get_cloth_code(cloth_num=5) 
    body_code = model.get_body_code(body_num=1) 
    cloth_v, cloth_t, cloth_path = model.reconstruct_cloth(cloth_code, "c")
    body_v, body_t, body_path = model.reconstruct_body(body_code, "c")


    deform_body_v, _ = model.deform_body(body_deform_code=beta, vertices_body_T=body_v, f=body_t, name="shape")
    deform_cloth_v, _ = model.deform_cloth(beta, cloth_v, cloth_t, name="shape")

    cloth_path = model.cloth_lbs(deform_cloth_v, beta, pose, cloth_t, name="pose")
    body_path = model.body_lbs(deform_body_v, beta, pose, body_t, name="pose")

    model.deform_cloth_collision(body_path, cloth_path, name="gen")



