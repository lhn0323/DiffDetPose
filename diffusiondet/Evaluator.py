import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import copy
import itertools
import json
import numpy as np
import os
import math
from collections import OrderedDict
import pycocotools.mask as maskUtils
import torch
from torch import nn
from tabulate import tabulate
from diffusiondet.loss import box3d, matrix2angle, xyzs_to_xys
from prettytable import PrettyTable
from diffusiondet.util import box_ops
from read_dataset.visual_utils import project_to_image
from .detector import Data_handing
import detectron2.utils.comm as comm
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table,log_every_n_seconds, setup_logger
from collections import defaultdict
from detectron2.evaluation.evaluator import DatasetEvaluator

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]  
  
def angle2normmatrix(rotate_angle):
    if len(rotate_angle) == 0:
        return []

    bz_r_pose_rad = np.array(rotate_angle) * (math.pi / 180.0) # [nums,3]

    cos_x = np.cos(bz_r_pose_rad[:,0]) 
    sin_x = np.sin(bz_r_pose_rad[:,0])
    cos_y = np.cos(bz_r_pose_rad[:,1])
    sin_y = np.sin(bz_r_pose_rad[:,1])
    cos_z = np.cos(bz_r_pose_rad[:,2])
    sin_z = np.sin(bz_r_pose_rad[:,2])

    rotation_matrixes = np.zeros((bz_r_pose_rad.shape[0], 3, 3), dtype=np.float32)
    rotation_matrixes[:, 0, 0] = cos_y * cos_z
    rotation_matrixes[:, 0, 1] = - cos_x * sin_z + sin_x * sin_y * cos_z
    rotation_matrixes[:, 0, 2] = sin_x * sin_z + cos_x * sin_y * cos_z
    rotation_matrixes[:, 1, 0] = cos_y * sin_z
    rotation_matrixes[:, 1, 1] = cos_x * cos_z + sin_x * sin_y * sin_z
    rotation_matrixes[:, 1, 2] = - sin_x * cos_z + cos_x * sin_y * sin_z
    rotation_matrixes[:, 2, 0] = - sin_y
    rotation_matrixes[:, 2, 1] = sin_x * cos_y
    rotation_matrixes[:, 2, 2] = cos_x * cos_y
    
    return rotation_matrixes  

class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if comm.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

class Parameters:
    def __init__(self):
        self.catIds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.maxDets = [1, 10, 100]
        self.ious = {}
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True) # recall thrs 
        self.shiftThrs = np.array([1, 2, 5]) 
        self.degreeThrs = np.array([5, 10, 15]) 
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.stats = []
        self.evalImgs = []
        self.useCats = 1
        self.imgIds = []
        self.iouType = 'bbox'
            
    def evaluate(self, predictions):
        computeIoU = self.computeIoU
        self.ious = {(prediction['image_idx'], catId): computeIoU(prediction, catId) \
                        for prediction in predictions
                        for catId in self.catIds} 
        evaluateImg = self.evaluateImg
        maxDet = self.maxDets[-1]
        self.evalImgs = [evaluateImg(prediction, catId, areaRng, maxDet)
                 for catId in self.catIds 
                 for areaRng in self.areaRng 
                 for prediction in predictions] 
        computeCm5Angle = self.computeCm5Angle
        self.cm5angles = {(prediction['image_idx'], catId): computeCm5Angle(prediction, catId)
                 for prediction in predictions # per image
                 for catId in self.catIds} # per class
        evaluateTR = self.evaluateTR
        self.evaluateTRs = [evaluateTR(prediction, catId, maxDet)
                 for catId in self.catIds # per class
                 for prediction in predictions] # per image
    
    def computeIoU(self, imgs, cat):
        dt = [img for img in imgs["instances"] if img['category_id']==cat ] 
        gt = [img for img in imgs["inputs_informations"] if img['category_id']==cat ] 
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt ], kind='mergesort') 
        dt = [dt[i] for i in inds]
        if len(dt) > self.maxDets[-1]:
            dt=dt[0:self.maxDets[-1]] 
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
        iscrowd = [int(0) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd) 
        return ious   
    
    def evaluateImg(self, imgs, cat, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        dt = [img for img in imgs["instances"] if img['category_id']==cat ]
        gt = [img for img in imgs["inputs_informations"] if img['category_id']==cat ]        
        if len(gt) == 0 and len(dt) == 0:
            return None       
        
        gtind = np.argsort([0 for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(0) for o in gt]
        ious = self.ious[imgs['image_idx'], cat] if len(self.ious[imgs['image_idx'], cat]) > 0 else self.ious[imgs['image_idx'], cat]
         
        T = len(self.iouThrs) 
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G)) 
        dtm  = np.zeros((T,D)) 
        gtIg = np.array([0 for g in gt]) 
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(self.iouThrs): 
                for dind, d in enumerate(dt): 
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m = -1
                    for gind, g in enumerate(gt): 
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue 
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind] 
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgs['image_idx'],
                'category_id':  cat,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'aRng':         aRng,
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }
       
    def computeCm5Angle(self, imgs, cat):
        
        dt = [img for img in imgs["instances"] if img['category_id']==cat ]
        gt = [img for img in imgs["inputs_informations"] if img['category_id']==cat ]
        if len(gt) == 0 and len(dt) ==0:
            return []
        if len(gt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt ], kind='mergesort') 
        dt = [dt[i] for i in inds]
        if len(dt) > self.maxDets[-1]:
            dt=dt[0:self.maxDets[-1]] 
            
        gt_trans_vector = [g['gt_t'] for g in gt]
        gt_rotate_angle = [g['gt_r'] for g in gt]
        gt_rotate_matrix = angle2normmatrix(gt_rotate_angle) 
        dt_trans_vector = [d['pose'][:3] for d in dt]
        dt_rotate_angle = [d['pose'][3:] for d in dt]
        dt_rotate_matrix = angle2normmatrix(dt_rotate_angle)
        
        # The absolute value of the difference between the predicted and true values (dt,gt)  (convert m to cm by *100)
        # 1.R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
        # 2.R = R1 @ R2.transpose()
        # 3.theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
        # 3.theta *= 180 / np.pi
        trans = np.array([ 
                          np.linalg.norm(dt_t - gt_t) 
            for dt_t in np.array(dt_trans_vector)
            for gt_t in np.array(gt_trans_vector)]).reshape(len(dt),len(gt))
        rotate = np.array([ 
                           np.arccos(np.clip((np.trace(dt_r @ gt_r.transpose()) - 1) / 2, -1.0, 1.0))*180 / np.pi
            for dt_r in np.array(dt_rotate_matrix)
            for gt_r in np.array(gt_rotate_matrix)]).reshape(len(dt),len(gt))
        
        return np.concatenate((trans[:,:,np.newaxis], rotate[:,:,np.newaxis]), axis=-1)
    
    def evaluateTR(self, imgs, cat, maxDet):
        dt = [img for img in imgs["instances"] if img['category_id']==cat ]
        gt = [img for img in imgs["inputs_informations"] if img['category_id']==cat ]        
        if len(gt) == 0 and len(dt) == 0:
            return None 
        if len(gt) == 0:
            return None 
        # sort dt highest score first, sort gt ignore last
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort') 
        dt = [dt[i] for i in dtind[0:maxDet]]
        
        # load computed cm5angles[num_dts,num_gts,(trans,totate)]
        cm5angles = self.cm5angles[imgs['image_idx'], cat] if len(self.cm5angles[imgs['image_idx'], cat]) > 0 else self.cm5angles[imgs['image_idx'], cat]
        
        num_shift_thrs = len(self.shiftThrs)
        num_degree_thrs = len(self.degreeThrs)
        dt_matches = -1 * np.ones((num_shift_thrs, num_degree_thrs, len(dt)))
        gt_matches = -1 * np.ones((num_shift_thrs, num_degree_thrs, len(gt)))

        if not len(cm5angles)==0:
            for d, shift_thrs in enumerate(self.shiftThrs):                
                for s, degree_thrs in enumerate(self.degreeThrs):
                    for i in range(cm5angles.shape[0]):
                        sum_degree_shift = np.sum(cm5angles[i,:,:], axis=-1) 
                        sorted_ixs = np.argsort(sum_degree_shift) 
                        for j in sorted_ixs:
                            if gt_matches[d, s, j] > -1 :
                                continue
                            # If we reach IoU smaller than the threshold, end the loop
                            if cm5angles[i,j,0] > shift_thrs or cm5angles[i,j,1] > degree_thrs:
                                continue

                            gt_matches[d, s, j] = i 
                            dt_matches[d, s, i] = j 
                            break
       
        return {
            'image_id':     imgs['image_idx'],
            'category_id':  cat,
            'dtMatches':    dt_matches,
            'gtMatches':    gt_matches,
        }
    
    def computeProj2d(self, imgs, cat):
        
        dt = [img for img in imgs["instances"] if img['category_id']==cat ]
        gt = [img for img in imgs["inputs_informations"] if img['category_id']==cat ]
        if len(gt) == 0 and len(dt) ==0:
            return []
        return
        
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):

            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(self.iouThrs[0], self.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(self.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(self.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()
        
    def accumulate(self, p = None):
        self.accumulate_detect()
        self.accumulate_pose()
    
    def accumulate_pose(self, p = None):
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evaluateTRs:
            print('Please run evaluateTR() first')
        catId_dtMatches_all=[np.zeros((len(self.shiftThrs), len(self.degreeThrs), 0)) for _  in self.catIds]
        catId_gtMatches_all=[np.zeros((len(self.shiftThrs), len(self.degreeThrs), 0)) for _  in self.catIds]
        pose_aps = -np.ones((len(self.catIds) + 1, len(self.shiftThrs), len(self.degreeThrs)))
        for catId in self.catIds: 
            a=[x["dtMatches"] for x in self.evaluateTRs if x and x["category_id"]==catId] 
            if a:
                catId_dtMatches_all[catId] = np.concatenate(a, axis=-1)
            else:
                catId_dtMatches_all[catId] = None
            b=[x["gtMatches"] for x in self.evaluateTRs if x and x["category_id"]==catId]
            if b:
                catId_gtMatches_all[catId] = np.concatenate(b, axis=-1)
            else:
                catId_gtMatches_all[catId] = None
        
        for s, shift_thrs in enumerate(self.shiftThrs):                
            for d, degree_thrs in enumerate(self.degreeThrs):  
                for catId in self.catIds:  
                    if catId_dtMatches_all[catId] is None or catId_gtMatches_all[catId] is None:
                        continue
                    
                    catId_dtMatches = catId_dtMatches_all[catId][s, d, :]
                    catId_gtMatches = catId_gtMatches_all[catId][s, d, :]
                    precisions = np.cumsum(catId_dtMatches > -1) / (np.arange(len(catId_dtMatches)) + 1)
                    recalls = np.cumsum(catId_dtMatches > -1).astype(np.float32) / len(catId_gtMatches)
                    precisions = np.concatenate([[0], precisions, [0]])
                    recalls = np.concatenate([[0], recalls, [1]])

                    # Ensure precision values decrease but don't increase. This way, the
                    # precision value at each recall threshold is the maximum it can be
                    # for all following recall thresholds, as specified by the VOC paper.
                    for i in range(len(precisions) - 2, -1, -1):
                        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
                    # Compute mean AP over recall range
                    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
                    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
                    
                    pose_aps[catId, s, d] = ap
                a=pose_aps[:-1, s, d]  
                pose_aps[-1, s, d] = np.mean(a[a>-1]) 
        self.pose_eval = {
            'PoseAPs': pose_aps,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))
    
    def accumulate_detect(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
            
        # allows input customized parameters
        self.catIds = self.catIds if self.useCats == 1 else [-1]
        T           = len(self.iouThrs)
        R           = len(self.recThrs)
        K           = len(self.catIds) if self.useCats else 1
        A           = len(self.areaRng)
        M           = len(self.maxDets)
        precision   = -np.ones((T,R,K,A,M)) 
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        catIds = self.catIds if self.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, self.areaRng))
        setM = set(self.maxDets)
        setI = set(self.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(self.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(self.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), self.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(self.imgIds)  if i in setI]
        I0 = len(self.imgIds)
        A0 = len(self.areaRng)
        for k, k0 in enumerate(k_list): 
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list): 
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, self.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))
    
def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = comm.get_world_size()
    logger = logging.getLogger("detectron2")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        setup_logger()
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()
    
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            
            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_secondsselfr_iter = total_data_time / iters_after_start
            compute_secondsselfr_iter = total_compute_time / iters_after_start
            eval_secondsselfr_iter = total_eval_time / iters_after_start
            total_secondsselfr_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_secondsselfr_iter > 5:
                eta = datetime.timedelta(seconds=int(total_secondsselfr_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_secondsselfr_iter:.4f} s/iter. "
                        f"Inference: {compute_secondsselfr_iter:.4f} s/iter. "
                        f"Eval: {eval_secondsselfr_iter:.4f} s/iter. "
                        f"Total: {total_secondsselfr_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results  

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
    

   
class ROPEEvaluator(DatasetEvaluator):
    """
    """
    def __init__(
        self,
        distributed=True,
        output_dir=None,
        *,
        max_detsselfr_image=None,
    ):
        self._logger = logging.getLogger("detectron2")
        if not self._logger.isEnabledFor(logging.INFO):  
            setup_logger()
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = False

        if max_detsselfr_image is None:
            max_detsselfr_image = [1, 10, 100]
        else:
            max_detsselfr_image = [1, 10, max_detsselfr_image]
        self._max_detsselfr_image = max_detsselfr_image
        self._task = "bbox"
        self.ids_gt = 1
        self.ids_dt = 1

        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._predictions = []
    
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_idx": input[0] } 

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"], prediction["inputs_informations"] = self.instances_to_rope3d(instances, inputs)
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)
                
    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[ROPEEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions) 
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
    
    def _eval_predictions(self, predictions):
        """ itertools.chain(*[x["instances"] for x in predictions])
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()

        tic = time.time()
        print('Running per image evaluation...')
        # Create an instance of the Parameters class
        p = Parameters()
        p.imgIds = [x["image_idx"] for x in predictions] 
        # p._prepare(ropeGt, ropeDt) # ropeGt, ropeDt
        p.evaluate(predictions)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        
        p.accumulate()
        p.summarize()
        eval = (p)
        res = self._derive_rope_results(eval, 'bbox', CLASSES)
        self._results[self._task] = res
    
    
    def instances_to_rope3d(self, instances, inputs): 
        """
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            instances (Instances): bboxes:XYXY_ABS
            inputs (int): bboxes:XYXY_ABS

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
      
        batched_inputs = Data_handing(inputs)
        (zoom_imgs_shape,imgidx,sweep_imgs,gt_labels,gt_2dboxes,gt_3dboxes,gt_lwhs,gt_P2s,gt_translation_matrixs,gt_rotation_matrixs) = batched_inputs
    
        trans_factor = torch.tensor([[31,21,100]])
        rotate_factor = torch.tensor([[90,40,90]])
        gt_translation_vectors = gt_translation_matrixs[0]*trans_factor
        gt_rotation_angles = gt_rotation_matrixs[0]*rotate_factor
        
        boxes = instances.pred_boxes.tensor.numpy() 
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) 
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        poses_pre = instances.pred_poses*torch.cat((trans_factor,rotate_factor),dim=-1)
        poses = poses_pre.tolist()
        imageid = inputs[0][0]
        
        gt_2dboxes = BoxMode.convert(gt_2dboxes[0].numpy(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        
        if torch.cuda.is_available():
            gt_labels = [gt_label.tolist() for gt_label in gt_labels[0]] 
            gt_2dboxes = [gt_2dbox.tolist() for gt_2dbox in gt_2dboxes]
            gt_3dboxes = [gt_3dbox.tolist() for gt_3dbox in gt_3dboxes[0]]
            gt_lwhs = [gt_lwh.tolist() for gt_lwh in gt_lwhs[0]]
            gt_P2s = [gt_P2.tolist() for gt_P2 in gt_P2s[0]]
            gt_translation_vectors = [gt_translation_matrix.tolist() for gt_translation_matrix in gt_translation_vectors]
            gt_rotation_angles = [gt_rotation_angle.tolist() for gt_rotation_angle in gt_rotation_angles]
                
        has_mask = instances.has("pred_masks")
        if has_mask:
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            rles = [
                maskUtils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                rle["counts"] = rle["counts"].decode("utf-8")

        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints

        num_instance = len(instances)
        results = [] 
        for k in range(num_instance):
            id_dt = self.ids_dt +k
            bb = boxes[k] 
            area = bb[2]*bb[3]
            result = {
                "image_id": imageid,
                "category_id": classes[k],
                "bbox": boxes[k], 
                "score": scores[k],
                "pose": poses[k],
                "area":area, 
                "id":id_dt
            }
            if has_mask:
                result["segmentation"] = rles[k]
            if has_keypoints:
                # In COCO annotations,
                # keypoints coordinates are pixel indices.
                # However our predictions are floating point coordinates.
                # Therefore we subtract 0.5 to be consistent with the annotation format.
                # This is the inverse of data loading logic in `datasets/coco.py`.
                keypoints[k][:, :2] -= 0.5
                result["keypoints"] = keypoints[k].flatten().tolist()
            results.append(result)
            
        num_inputs = len(gt_labels)    
        inputs_informations = []
        for k in range(num_inputs):
            id_gt = self.ids_gt +k
            bb = gt_2dboxes[k] 
            area = bb[2]*bb[3]
            inputs_information = {
                "image_id": imageid,
                "category_id": gt_labels[k],
                "bbox":  gt_2dboxes[k],
                "gt_t": gt_translation_vectors[k],
                "gt_r":gt_rotation_angles[k],
                "area":area, 
                "id":id_gt
            }
            inputs_informations.append(inputs_information)
        self.ids_gt += num_inputs
        self.ids_dt += num_instance   
        return results, inputs_informations
    
    def _derive_rope_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
                "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
            }[iou_type]
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}


        results = {
                metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
                for idx, metric in enumerate(metrics)
            }
        self._logger.info(
                "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
            )
        pose_metrics = {"pose": ["1m5°","1m10°","1m15°","2m5°","2m10°","2m15°","5m5°","5m10°","5m15°"]}
        pose_results = {
                metric: float(coco_eval.pose_eval['PoseAPs'][-1, idx//3,idx%3] * 100 
                              if coco_eval.stats[idx] >= 0 else "nan")
                for idx, metric in enumerate(pose_metrics["pose"])
            }
        self._logger.info(
                "Evaluation results for {}: \n".format("pose") + create_small_table(pose_results)
            )
        
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
            # Compute per-category AP
            # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
            # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

            # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
            )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
        results.update({"AP-" + name: ap for name, ap in results_per_category})
        results.update({"AP-" + name: ap for name, ap in pose_results.items()})
        # pose
        pose_table = PrettyTable()
        pose_table.field_names =  ["class"] + pose_metrics['pose']
        pose_ap_values = coco_eval.pose_eval['PoseAPs'][:len(class_names),:,:].reshape(len(class_names),9)
        for idx, name in enumerate(class_names):
            a=pose_ap_values[idx]
            a=a[a>-1]
            if len(a):
                value=list(np.around(a*100, decimals=3))
            else:
                value=["nan","nan","nan","nan","nan","nan","nan","nan","nan",]
            row = [name] + value
            pose_table.add_row(row)
        self._logger.info("Per-category Pose AP: \n" + str(pose_table))   
  
        return results 