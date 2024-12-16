'''
  @ Date: 2021-06-14 22:27:05
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 10:33:26
  @ FilePath: /EasyMocapRelease/apps/demo/smpl_from_keypoints.py
'''
# This is the script of fitting SMPL to 3d(+2d) keypoints
from easymocap.dataset import CONFIG
from easymocap.mytools import Timer
from easymocap.smplmodel import load_model, select_nf
from easymocap.mytools.reader import read_keypoints3d_all
from easymocap.mytools.file_utils import write_smpl, write_vertices
from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
from easymocap.pipeline import smpl_from_keypoints3d
import os
from os.path import join
from tqdm import tqdm
import pickle
import torch

def convert_to_standard(params: dict, body_model):
    new_poses, new_th, _ = body_model.convert_to_standard_smpl(
        params['poses'],
        params['shapes'],
        params['Rh'],
        params['Th']
    )
    params['Th'] = new_th
    params['poses'] = new_poses
    params['Rh'] = torch.ones(params['Rh'].shape) * -1
    return params

def smpl_from_skel(path, sub, out, skel3d, args):
    config = CONFIG[args.body]
    all_verts = []
    results3d, filenames = read_keypoints3d_all(skel3d)
    pids = list(results3d.keys())
    weight_shape = load_weight_shape(args.model, args.opts)
    weight_pose = load_weight_pose(args.model, args.opts)
    with Timer('Loading {}, {}'.format(args.model, args.gender)):
        body_model = load_model(args.gender, model_type=args.model)
    for idx, (pid, result) in enumerate(results3d.items()):
        body_params = smpl_from_keypoints3d(body_model, result['keypoints3d'], config, args,
                                            weight_shape=weight_shape, weight_pose=weight_pose)
        result['body_params'] = body_params

        # Add a field with the parameters converted to standard ones
        result['body_params_smpl'] = convert_to_standard(result['body_params'].copy(), body_model)

    # write for each frame
    for nf, skelname in enumerate(tqdm(filenames, desc='writing')):
        basename = os.path.basename(skelname)
        outname = join(out, basename)
        vertout = join(out, 'verts', basename)
        res = []
        res_smpl = []
        verts = []
        for idx, (pid, result) in enumerate(results3d.items()):
            frames = result['frames']
            if nf in frames:
                nnf = frames.index(nf)
                val = {'id': pid}
                vs = {'id': pid}
                params = select_nf(result['body_params'], nnf)
                vertices = body_model(return_vertices=True, return_tensor=False, **params)
                vs['vertices'] = vertices.squeeze()
                val.update(params)
                res.append(val)
                verts.append(vs)

                # Add the converted ones
                val_smpl = {'id': pid}
                params_smpl = select_nf(result['body_params_smpl'], nnf)
                val_smpl.update(params_smpl)
                res_smpl.append(val_smpl)

        all_verts.append(verts)
        write_smpl(outname, res)
        write_smpl(join(out + "_converted", basename), res_smpl)
        # write_vertices(vertout, verts)
        with open('all_verts.pkl', 'wb') as fp:
            pickle.dump(all_verts, fp)


if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser

    parser = load_parser()
    parser.add_argument('--skel3d', type=str, required=True)
    args = parse_parser(parser)
    help = """
  Demo code for fitting SMPL to 3d(+2d) skeletons:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, args.skel3d, args.out,
           args.model, args.gender, args.body)
    print(help)
    smpl_from_skel(args.path, args.sub, args.out, args.skel3d, args)
