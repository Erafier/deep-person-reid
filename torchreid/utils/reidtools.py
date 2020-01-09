from __future__ import print_function, absolute_import
import numpy as np
import shutil
import os.path as osp
import cv2
import json
from .tools import mkdir_if_missing

__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width


def visualize_ranked_results(
    distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10
):

    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(
                    dst, prefix + '_top' + str(rank).zfill(3)
                ) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                     osp.basename(src)
            )
            shutil.copy(src, dst)

    used_cameras = dict()
    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        if np.min(distmat[q_idx, :]) > 650:  # Если для запроса нет совпадений, то не выводим его
            continue
        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
            grid_img[:, :width, :] = qimg
        else:
            qdir = osp.join(
                save_dir, osp.basename(osp.splitext(qimg_path_name)[0])
            )
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        cams = []
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            distance = distmat[q_idx, g_idx]
            # Не берём больше 1
            if not invalid:
                if distance > 800:  # Отбрасываем совсем непохожие
                    break
                elif distance > 600:  # Проверка на непохожесть
                    border_color = (0, 0, 255)
                elif distance > 500:
                    border_color = (0, 144, 255)
                elif distance > 400:
                    border_color = (0, 255, 255)
                elif distance > 50:
                    border_color = (0, 255, 0)
                else:
                    border_color = (75, 0, 130)
                if gcamid not in cams:
                    cams.append(gcamid)
                    print(qimg_path)
                    print(gcamid)
                    print(distance)
                    matched = gpid == qpid
                    if data_type == 'image':
                        # border_color = GREEN if matched else RED
                        gimg = cv2.imread(gimg_path)
                        gimg = cv2.resize(gimg, (width, height))
                        gimg = cv2.copyMakeBorder(
                            gimg,
                            BW,
                            BW,
                            BW,
                            BW,
                            cv2.BORDER_CONSTANT,
                            value=border_color
                        )
                        cv2.putText(img=gimg,
                                    text='Camera' + str(gcamid),
                                    org=(10, 245),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.75,
                                    color=(0, 255, 255),
                                    thickness=2)
                        gimg = cv2.resize(gimg, (width, height))
                        start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        end = (
                                      rank_idx + 1
                              ) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        grid_img[:, start:end, :] = gimg
                    else:
                        _cp_img_to(
                            gimg_path,
                            qdir,
                            rank=rank_idx,
                            prefix='gallery',
                            matched=matched
                        )
                    rank_idx += 1
                    if rank_idx > topk:
                        break
        used_cameras[qimg_path_name.split('/')[-1]] = cams
        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)
    with open('static/reid/cameras.json', 'w') as f:
        json.dump(used_cameras, f)
