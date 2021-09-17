from core import utils
from core.config import cfg
import numpy as np

from imgaug import parameters as iap
from imgaug import augmenters as iaa
from argparse import RawTextHelpFormatter

import argparse
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Please enter augument image number',
                                     formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--n_aug', default=100, type=int, dest='n_aug',
                        help= 'number of augument')
    parser.add_argument('--test_size', default=0.2, type=float, dest='test_size',
                        help= 'split_test_size')

    parser.add_argument('--b_limit', default=True, type=bool, dest='b_limit',
                        help= 'flag to limit maxium number of images')
    return parser.parse_args()

def main(args):    
    ##Agumentaion 시퀀셜을 설정합니다.
    
    """A기법"""
    # seq = iaa.Sequential([
    #                 iaa.Cutout(nb_iterations=2),
    #                 iaa.Affine(rotate=(-25, 25)),
    #                 iaa.Fliplr(0.5),
    #                 iaa.GammaContrast((0.5, 2.0)),
    #                 ])

    """B기법"""
    matrix = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

    seq = iaa.Sequential([
                  iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                  iaa.ContrastNormalization
                             (iap.Choice([1.0, 1.5, 3.0],
                                         p=[0.5, 0.3, 0.2])),
                  iaa.CLAHE(tile_grid_size_px=(3, 21)),
                  iaa.Convolve(matrix=matrix)
                  ])
                  
    ##"./data/classes/car.json"의 경로에서 car.json (클래스당 카운팅 된 json 파일)을 불러옵니다.
    df_classes = pd.read_json(cfg.NEXTLAB.CLASSES_JSON, "r", encoding='UTF8')
    df_classes = df_classes.T.rename_axis('class_name').reset_index()
    print("num of class", len(df_classes))
    ##Aunmentaion을 진행합니다.

    utils.augmentaionv2(df_classes, seq, args.n_aug , args.test_size, args.b_limit)
#python launcher_augumentV2.py --n_aug 100 --test_size 0.2
if __name__ == '__main__':

    main(parse_args())