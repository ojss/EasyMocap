'''
  @ Date: 2021-04-02 20:33:14
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-06 13:10:53
  @ FilePath: /EasyMocap/scripts/dataset/pre_panoptic.py
'''
# process script for CMU-Panoptic data

import numpy as np
import json
from glob import glob
from os.path import join
import os
from easymocap.mytools import write_camera, read_json, save_json
from easymocap.dataset import CONFIG
from xrprimer.data_structure import Keypoints
from xrprimer.transform.convention.keypoints_convention import convert_keypoints

import shutil
import re
from pathlib import Path
from typing import Tuple
from tqdm import tqdm, trange

SCALE = 100
FACE_FILE_FORMAT = "faceRecon3D_hd{frame}.json"
HAND_FILE_FORMAT = "handRecon3D_hd{frame}.json"


def extract_number_from_filename(file_name):
    m = re.search(r'_(\d+)', file_name)
    if m:
        return m.group(1)
    else:
        return None


def get_unique_ids(body_data_path: Path, hand_data_path: Path, face_data_path: Path) -> Tuple[list, dict, int]:
    """
    In one pass extracts all unique bodies and valid jsons.
    # TODO: should probably be compressed down to one function later. ONe loop for validation + extraction is possible.
    :param body_data_path:
    :param hand_data_path:
    :param face_data_path:
    :return:
    """
    ids = set()
    valid_file_list = {
        "body": [],
        "hands": [],
        "face": []
    }
    c = 0
    for json_file in tqdm(body_data_path.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

                frame_number = extract_number_from_filename(json_file.name)
                hand_file = hand_data_path / HAND_FILE_FORMAT.format(frame=frame_number)
                face_file = face_data_path / FACE_FILE_FORMAT.format(frame=frame_number)
                if not (hand_file.exists() and face_file.exists()):
                    raise FileNotFoundError
                valid_file_list["body"].append(json_file)
                valid_file_list["hands"].append(hand_file)
                valid_file_list["face"].append(face_file)
                c += 1
            for body in data.get("bodies", []):
                ids.add(body.get("id"))
        except Exception as e:
            print(f"Ignoring File: {json_file}")
    print(f"{c} Valid jsons found. Unique bodies found.")
    return sorted(list(ids)), valid_file_list, c


def convert_camera(inp, out):
    camnames = glob(join(inp, '*.json'))
    assert len(camnames) == 1, camnames
    # Load camera calibration parameters
    with open(camnames[0]) as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras_ = {cam['name']: cam for cam in calib['cameras']}
    cameras = {}
    # Convert data into numpy arrays for convenience
    for k, cam in cameras_.items():
        if cam['type'] != 'hd':
            continue
        cam['K'] = np.array(cam['K'])
        cam['dist'] = np.array(cam['distCoef']).reshape(1, -1)
        cam['R'] = np.array(cam['R'])
        cam['T'] = np.array(cam['t']).reshape((3, 1)) / SCALE
        cam = {key: cam[key] for key in ['K', 'dist', 'R', 'T']}
        cameras[k] = cam
    write_camera(cameras, out)


def copy_videos(inp, out):
    outdir = join(out, 'videos')
    os.makedirs(outdir, exist_ok=True)
    hdnames = os.listdir(join(inp, 'hdVideos'))
    for hdname in tqdm(hdnames):
        outname = join(outdir, hdname.replace('hd_', ''))
        shutil.copy(join(inp, 'hdVideos', hdname), outname)


def convert_keypoints3d(inp, out):
    bodynames = join(inp, 'hdPose3d_stage1_coco19')
    handnames = join(inp, 'hdHand3d')
    facenames = join(inp, 'hdFace3d')

    uid, valid_files, _ = get_unique_ids(Path(bodynames), Path(handnames), Path(facenames))

    out = join(out, 'keypoints3d')
    os.makedirs(out, exist_ok=True)
    names_i = CONFIG['panoptic']['joint_names']
    names_o = CONFIG['body25']['joint_names']
    commons = [i for i in names_o if i in names_i]
    idx_i = [names_i.index(i) for i in commons]
    idx_o = [names_o.index(i) for i in commons]
    use_hand = True
    use_face = True
    pid_list = set()
    if use_hand and not use_face:
        zero_body = np.zeros((25 + 21 + 21, 4))
    elif use_hand and use_face:
        zero_body = np.zeros((25 + 21 + 21 + 70, 4))
    else:
        zero_body = np.zeros((25, 4))

    for body_json, hand_json, face_json in tqdm(zip(valid_files["body"], valid_files["hands"], valid_files["face"]),
                                                total=len(valid_files["body"])):
        # bodyname = bodynames.format(i)
        uid = extract_number_from_filename(body_json.name)
        if not os.path.exists(body_json):
            continue
        bodies = read_json(body_json)
        if len(bodies['bodies']) < 3:
            continue
        if bodies['bodies'][0]['id'] < 0:
            continue
        results = []

        for data in bodies['bodies']:
            personid = data['id']
            if personid < 0: continue
            joints19 = np.array(data['joints19']).reshape(-1, 4)
            joints19[:, :3] /= SCALE
            joints19[:, 3][joints19[:, 3] < 0] = 0
            keypoints3d = zero_body.copy()
            keypoints3d[idx_o, :] = joints19[idx_i, :]
            pid_list.add(personid)
            results.append({'id': personid, 'keypoints3d': keypoints3d})
        # handname = handnames.format(i)
        hands = read_json(hand_json)
        faces = read_json(face_json)
        # breakpoint()
        lwrists = np.stack([res['keypoints3d'][7] for res in results])
        left_valid = np.zeros(len(results)) + 0.2
        rwrists = np.stack([res['keypoints3d'][4] for res in results])
        right_valid = np.zeros(len(results)) + 0.2
        for data in hands['people']:
            pid = data['id']
            if 'left_hand' in data.keys():
                left_p = np.array(data['left_hand']['landmarks']).reshape((-1, 3))
                left_v = np.array(data['left_hand']['averageScore']).reshape((-1, 1))
                left_v[left_v < 0] = 0
                left = np.hstack((left_p / SCALE, left_v))
                if left[0, -1] > 0 and (left_v > 0).sum() > 10:
                    dist = np.linalg.norm(left[:1, :3] - lwrists[:, :3], axis=1)
                    dist_min, pid = dist.min(), dist.argmin()
                    if left_valid[pid] > dist_min:
                        left_valid[pid] = dist_min
                        results[pid]['keypoints3d'][25:25 + 21, :] = left
            if 'right_hand' in data.keys():
                right_p = np.array(data['right_hand']['landmarks']).reshape((-1, 3))
                right_v = np.array(data['right_hand']['averageScore']).reshape((-1, 1))
                right_v[right_v < 0] = 0
                right = np.hstack((right_p / SCALE, right_v))
                if right[0, -1] > 0 and (right_v > 0).sum() > 10:
                    dist = np.linalg.norm(right[:1, :3] - rwrists[:, :3], axis=1)
                    dist_min, pid = dist.min(), dist.argmin()
                    if right_valid[pid] > dist_min:
                        right_valid[pid] = dist_min
                        results[pid]['keypoints3d'][25 + 21:25 + 21 + 21, :] = right
        for data in faces['people']:
            personid = data['id']
            if personid < 0: continue
            if personid not in pid_list: continue
            face_p = np.array(data['face70']['landmarks']).reshape((-1, 3)) / SCALE
            face_v = np.array(data['face70']['averageScore']).reshape((-1, 1))
            face_v[face_v < 0] = 0
            face = np.hstack((face_p, face_v))
            for r in results:
                if personid == r['id']:
                    r['keypoints3d'][25 + 21 + 21:, :] = face

            # find the correspondent people
        for res in results:
            res['keypoints3d'][:, -1][res['keypoints3d'][:, -1] > 0] = 1
            kps = Keypoints(convention='openpose_137')
            kps.set_keypoints(res['keypoints3d'][:, :3])
            kps.set_mask(res['keypoints3d'][:, -1])
            kps = convert_keypoints(keypoints=kps, dst='openpose_118', approximate=True)
            res['keypoints3d'] = np.concatenate((kps.get_keypoints(), kps.get_mask()[..., None]), axis=-1).squeeze()
        outname = join(out, '{}.json'.format(uid))
        # results = [val for key, val in results.items()]
        for res in results:
            res['keypoints3d'] = res['keypoints3d'].tolist()
        save_json(outname, results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inp', type=str)
    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.camera:
        convert_camera(args.inp, args.inp)
    convert_keypoints3d(args.inp, "./sample_data")
