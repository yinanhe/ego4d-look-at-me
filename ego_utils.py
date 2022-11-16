from genericpath import exists
import os, torch, glob, subprocess, shutil
from PIL.Image import merge
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
from ego_metrics import run_evaluation
# from common.distributed import is_master, synchronize

import utils

def is_master():
    return utils.get_rank() == 0

def synchronize():
    torch.distributed.barrier()

class PostProcessor():
    def __init__(self, args):
        self.rank = utils.get_rank()
        self.exp_path = args.output_dir
        self.save_path = f'{self.exp_path}/tmp'
        if not os.path.exists(self.save_path) and is_master():
            os.mkdir(self.save_path)
        self.groundtruth = []
        self.prediction = []
        self.groundtruthfile = f'{self.save_path}/gt.csv.rank.{self.rank}'
        self.predctionfile = f'{self.save_path}/pred.csv.rank.{self.rank}'

    def update(self, outputs, targets):
        # postprocess outputs of one minibatch
        outputs = F.softmax(outputs.float(), dim=-1)
        # print("target shape",len(targets),"uid",len(targets[0]),flush=True)
        for idx, scores in enumerate(outputs):
            uid = targets[0][idx]
            trackid = targets[1][idx]
            frameid = targets[2][idx].item()
            x1 = targets[3][0][idx].item()
            y1 = targets[3][1][idx].item()
            x2 = targets[3][2][idx].item()
            y2 = targets[3][3][idx].item()
            label = targets[4][idx].item()
            self.groundtruth.append([uid, frameid, x1, y1, x2, y2, trackid, label])
            self.prediction.append([uid, frameid, x1, y1, x2, y2, trackid, 1, scores[1].item()])

    def save(self):
        if os.path.exists(self.groundtruthfile):
            os.remove(self.groundtruthfile)
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
        gt_df = pd.DataFrame(self.groundtruth)
        gt_df.to_csv(self.groundtruthfile, index=False, header=None)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)
        synchronize()

    def get_mAP(self):
        # merge csv
        merge_path = f'{self.exp_path}/result'
        if not os.path.exists(merge_path):
            os.mkdir(merge_path)

        gt_file = f'{merge_path}/gt.csv'
        if os.path.exists(gt_file):
            os.remove(gt_file)
        gts = glob.glob(f'{self.save_path}/gt.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(gts), gt_file)
        subprocess.call(cmd, shell=True)
        pred_file = f'{merge_path}/pred.csv'
        if os.path.exists(pred_file):
            os.remove(pred_file)
        preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.save_path)
        return run_evaluation(gt_file, pred_file)


class TestPostProcessor():
    def __init__(self, args):
        self.exp_path = args.output_dir
        self.save_path = f'{self.exp_path}/tmp'
        if not os.path.exists(self.save_path) and is_master():
            os.mkdir(self.save_path)
        self.prediction = []
        self.predctionfile = f'{self.save_path}/pred.csv.rank.{args.rank}'

    def update(self, outputs, targets):
        # postprocess outputs of one minibatch
        outputs = F.softmax(outputs.float(), dim=-1)
        for idx, scores in enumerate(outputs):
            uid = targets[0][idx]
            trackid = targets[1][idx]
            unique_id = targets[2][idx]
            self.prediction.append([uid, unique_id, trackid, 1, scores[1].item()])

    def save(self):
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)
        synchronize()

        #merge csv
        merge_path = f'{self.exp_path}/result_final'
        if not os.path.exists(merge_path) and is_master():
            os.makedirs(merge_path,exist_ok=True)
        pred_file = f'{merge_path}/pred.csv'
        preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        # shutil.rmtree(self.save_path)

