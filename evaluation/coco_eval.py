from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import mmcv
import argparse
from mmdet.datasets import build_dataloader, build_dataset
import os
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
import time
import copy
import numpy as np
import datetime
import shutil
import itertools
from terminaltables import AsciiTable
import matplotlib.pyplot as plt


def computeIoU(self, imgId, catId):
    p = self.params
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return []
    inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]

    if p.iouType == 'segm':
        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]
    elif p.iouType == 'bbox':
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
    else:
        raise Exception('unknown iouType for iou computation')

    # compute iou between each dt and gt region
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = maskUtils.iou(d, g, iscrowd)
    return ious


def evaluateImg(self, imgId, catId, aRng, score, prog_bar):
    p = self.params
    gts = self.cocoGt
    images = gts.imgs
    img_width = images[imgId]['width']
    img_height = images[imgId]['height']

    prog_bar.update()

    if p.useCats:
        gt = self._gts[imgId, catId]

        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return None
    ################################### eval at resized image ###################################
    for g in gt:
        bbox_width = g['bbox'][2]
        bbox_heigth = g['bbox'][3]

        if args.mode == 'global':
            resized_width = bbox_width * (800 / img_width)
            resized_heigth = bbox_heigth * (800 / img_height)
        elif args.mode == 'local':
            resized_width = bbox_width * (3000 / img_width)
            resized_heigth = bbox_heigth * (3000 / img_height)
        else:
            raise ValueError('Wrong mode')
        resized_area = resized_width * resized_heigth

        if g['ignore'] or (resized_area < aRng[0] or resized_area > aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0
    ################################### eval at original image ###################################
    # for g in gt:
    #     if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
    #         g['_ignore'] = 1
    #     else:
    #         g['_ignore'] = 0
    # ############################################################################################

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]

    #################################################################################
    dt = [d for d in dt if d['score'] > score]
    #################################################################################

    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:p.maxDets[-1]]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T, D))
    if not len(ious) == 0:
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d['id']
    # set unmatched detections outside of area range to ignore
    a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
    # store results for given image and category
    return {
        'image_id': imgId,
        'category_id': catId,
        'aRng': aRng,
        'score': score,
        'maxDet': p.maxDets[-1],  # 100000
        'dtIds': [d['id'] for d in dt],
        'gtIds': [g['id'] for g in gt],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'dtScores': [d['score'] for d in dt],
        'gtIgnore': gtIg,
        'dtIgnore': dtIg,
    }


def coco_evaluate(self):
    tic = time.time()
    print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))

    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    self.ious = {(imgId, catId): computeIoU(self, imgId, catId) \
                 for imgId in p.imgIds
                 for catId in catIds}

    ######################################################################################
    prog_bar = mmcv.ProgressBar(len(self.params.imgIds) * len(self.params.areaRng) * len(self.params.catIds))
    ######################################################################################
    self.evalImgs = [evaluateImg(self, imgId, catId, areaRng, p.scoceThrs, prog_bar)
                     for catId in catIds
                     for areaRng in p.areaRng
                     for imgId in p.imgIds
                     ]
    self._paramsEval = copy.deepcopy(self.params)
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))


def coco_accumulate(self, p=None):
    print('Accumulating evaluation results...')
    tic = time.time()
    if not self.evalImgs:
        print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
        p = self.params
    p.catIds = p.catIds if p.useCats == 1 else [-1]
    T = len(p.iouThrs)
    R = len(p.recThrs)
    K = len(p.catIds) if p.useCats else 1
    A = len(p.areaRng)
    M = len(p.maxDets)
    precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    recall = -np.ones((T, K, A, M))
    scores = -np.ones((T, R, K, A, M))
    Pre = []

    # create dictionary for future indexing
    _pe = self._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds) if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        Nk = k0 * A0 * I0
        for a, a0 in enumerate(a_list):
            Na = a0 * I0
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

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    Pre.append(np.mean(pr))
                    q = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t, k, a, m] = rc[-1]
                    else:
                        recall[t, k, a, m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t, :, k, a, m] = np.array(q)
                    scores[t, :, k, a, m] = np.array(ss)
    self.eval = {
        'params': p,
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'Pr': Pre,
        'recall': recall,
        'scores': scores,
    }
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))


def coco_summarize(self, args):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100000, write_handle=None):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | Score={:>4} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        if ap == 1:
            titleStr = 'Average Precision'
        elif ap == 0:
            titleStr = 'Average Recall'
        elif ap == 2:
            titleStr = 'Precision'
        else:
            raise NotImplementedError('ap should be 0,1,2')
        if ap == 1:
            typeStr = '(AP)'
        elif ap == 0:
            typeStr = '(AR)'
        elif ap == 2:
            typeStr = '(PR)'
        else:
            raise NotImplementedError('ap should be 0,1,2')
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        elif ap == 0:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        elif ap == 2:
            s = self.eval['Pr']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                ##################################################
                idx = aind[0] * len(p.iouThrs) + t[0]
                ##################################################
                s = s[idx]
            else:
                pass

        if isinstance(s, list):  # for ap==2 ,to get average precision
            mean_s = np.mean(s)
        else:
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, p.scoceThrs, areaRng, maxDets, mean_s))
        if write_handle is not None:
            write_handle.write(iStr.format(titleStr, typeStr, iouStr, p.scoceThrs, areaRng, maxDets, mean_s) + '\n')
        return mean_s

    def _summarizeDets(args):
        stats = np.zeros((3,))
        f = open(os.path.join(args.work_dir, 'result.txt'), 'a+')
        bare_name = os.path.basename(args.work_dir)
        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('Detect with {}\n'.format(args.checkpoint))
        stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)  # AP
        stats[1] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)  # AR
        stats[2] = _summarize(2, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)  # PR
        f.write('\n')
        f.close()
        print('Successfully Write Result File...')

        # for batter stored result file,create a dir to stored result,and it can be upload to github
        assert os.path.isfile(os.path.join(args.work_dir, 'result.txt'))
        out_dir = 'result'
        os.makedirs(out_dir, exist_ok=True)
        out_file_name = os.path.join(out_dir, '{}.txt'.format(bare_name))
        shutil.copy(os.path.join(args.work_dir, 'result.txt'), out_file_name)
        return stats

    def _summarizeDets2(args):
        stats = np.zeros((9,))
        f = open(os.path.join(args.work_dir, 'result.txt'), 'a+')
        bare_name = os.path.basename(args.work_dir)
        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('Detect with {}\n'.format(args.checkpoint))
        stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[1] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[2] = _summarize(2, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[3] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[4] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[5] = _summarize(2, iouThr=.75, maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[6] = _summarize(1, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[7] = _summarize(0, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[8] = _summarize(2, maxDets=self.params.maxDets[-1], write_handle=f)
        f.write('\n')
        f.close()
        print('Successfully Write Result File...')

        # for batter stored result file,create a dir to stored result,and it can be upload to github
        assert os.path.isfile(os.path.join(args.work_dir, 'result.txt'))
        out_dir = 'result'
        os.makedirs(out_dir, exist_ok=True)
        out_file_name = os.path.join(out_dir, '{}.txt'.format(bare_name))
        shutil.copy(os.path.join(args.work_dir, 'result.txt'), out_file_name)
        return stats

    def _summarizeDets3(args):
        stats = np.zeros((12,))
        f = open(os.path.join(args.work_dir, 'result.txt'), 'a+')
        bare_name = os.path.basename(args.work_dir)
        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('Detect with {}\n'.format(args.checkpoint))
        stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)  # AP
        stats[1] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)  # AR
        stats[2] = _summarize(2, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)  # PR
        f.write(' ' + '-' * 94 + '\n')
        stats[3] = _summarize(1, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[4] = _summarize(0, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[5] = _summarize(2, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[6] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[7] = _summarize(0, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[8] = _summarize(2, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[9] = _summarize(1, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[10] = _summarize(0, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[11] = _summarize(2, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write('\n')
        f.close()
        print('Successfully Write Result File...')

        # for batter stored result file,create a dir to stored result,and it can be upload to github
        assert os.path.isfile(os.path.join(args.work_dir, 'result.txt'))
        out_dir = 'result'
        os.makedirs(out_dir, exist_ok=True)
        out_file_name = os.path.join(out_dir, '{}.txt'.format(bare_name))
        shutil.copy(os.path.join(args.work_dir, 'result.txt'), out_file_name)
        return stats

    def _summarizeDets4(args):
        stats = np.zeros((36,))
        f = open(os.path.join(args.work_dir, 'result.txt'), 'a+')
        bare_name = os.path.basename(args.work_dir)
        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('Detect with {}\n'.format(args.checkpoint))
        stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[1] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[2] = _summarize(2, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[3] = _summarize(1, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[4] = _summarize(0, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[5] = _summarize(2, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[6] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[7] = _summarize(0, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[8] = _summarize(2, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[9] = _summarize(1, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[10] = _summarize(0, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[11] = _summarize(2, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[12] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[13] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[14] = _summarize(2, iouThr=.75, maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[15] = _summarize(1, iouThr=.75, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[16] = _summarize(0, iouThr=.75, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[17] = _summarize(2, iouThr=.75, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[18] = _summarize(1, iouThr=.75, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[19] = _summarize(0, iouThr=.75, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[20] = _summarize(2, iouThr=.75, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[21] = _summarize(1, iouThr=.75, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[22] = _summarize(0, iouThr=.75, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[23] = _summarize(2, iouThr=.75, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[24] = _summarize(1, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[25] = _summarize(0, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[26] = _summarize(2, maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[27] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[28] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[29] = _summarize(2, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[30] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[31] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[32] = _summarize(2, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write(' ' + '-' * 94 + '\n')
        stats[33] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[34] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        stats[35] = _summarize(2, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=f)
        f.write('\n')
        f.close()
        print('Successfully Write Result File...')

        # for batter stored result file,create a dir to stored result,and it can be upload to github
        assert os.path.isfile(os.path.join(args.work_dir, 'result.txt'))
        out_dir = 'result'
        os.makedirs(out_dir, exist_ok=True)
        out_file_name = os.path.join(out_dir, '{}.txt'.format(bare_name))
        shutil.copy(os.path.join(args.work_dir, 'result.txt'), out_file_name)
        return stats

    if len(self.params.iouThrs) == 1 and len(self.params.areaRng) == 1:
        summarize = _summarizeDets
    elif len(self.params.iouThrs) == 10 and len(self.params.areaRng) == 1:
        summarize = _summarizeDets2
    elif len(self.params.iouThrs) == 1 and len(self.params.areaRng) == 4:
        summarize = _summarizeDets3
    elif len(self.params.iouThrs) == 10 and len(self.params.areaRng) == 4:
        summarize = _summarizeDets4
    else:
        raise ValueError('wrong param, please check set_param function')
    self.stats = summarize(args)


def evaluate(args, anns):
    cocoGt = COCO(args.val_path)
    cat_ids = cocoGt.getCatIds()
    cocoDt = cocoGt.loadRes(anns)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=args.eval)

    set_param(cocoEval)
    coco_evaluate(cocoEval)
    coco_accumulate(cocoEval)
    coco_summarize(cocoEval, args)

    if args.pr_curve:
        precisions = cocoEval.eval['precision']

        if args.iou_mode == 'single':
            pr_array1 = precisions[0, :, 0, 0, 0]
            print(pr_array1)
            x = np.arange(0.0, 1.01, 0.01)
            plt.plot(x, pr_array1, label='iou=0.5')
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.grid(True)
            plt.legend(loc='lower left')
            plt.show()
            plt.savefig(os.path.join(args.work_dir, 'pr_curve.png'))
        else:
            pr_array1 = precisions[0, :, 0, 0, 0]
            pr_array2 = precisions[1, :, 0, 0, 0]
            pr_array3 = precisions[2, :, 0, 0, 0]
            pr_array4 = precisions[3, :, 0, 0, 0]
            pr_array5 = precisions[4, :, 0, 0, 0]
            pr_array6 = precisions[5, :, 0, 0, 0]
            pr_array7 = precisions[6, :, 0, 0, 0]
            pr_array8 = precisions[7, :, 0, 0, 0]
            pr_array9 = precisions[8, :, 0, 0, 0]
            pr_array10 = precisions[9, :, 0, 0, 0]

            x = np.arange(0.0, 1.01, 0.01)
            plt.plot(x, pr_array1, label='iou=0.5')
            plt.plot(x, pr_array2, label='iou=0.55')
            plt.plot(x, pr_array3, label='iou=0.6')
            plt.plot(x, pr_array4, label='iou=0.65')
            plt.plot(x, pr_array5, label='iou=0.7')
            plt.plot(x, pr_array6, label='iou=0.75')
            plt.plot(x, pr_array7, label='iou=0.8')
            plt.plot(x, pr_array8, label='iou=0.85')
            plt.plot(x, pr_array9, label='iou=0.9')
            plt.plot(x, pr_array10, label='iou=0.95')

            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.grid(True)
            plt.legend(loc='lower left')
            plt.show()
            plt.savefig(os.path.join(args.work_dir, 'pr_curve.png'))

    if args.classwise:  # Compute per-category AP
        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/
        precisions = cocoEval.eval['precision']
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = cocoGt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(
            itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print(table.table)


def det2json(dataset, results, mode):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        if mode == 'local':
            imgInfo = dataset.data_infos[idx]
            width = imgInfo['width']
            height = imgInfo['height']
            result[:, 0] = result[:, 0] * (width / args.resize_width)
            result[:, 1] = result[:, 1] * (height / args.resize_height)
            result[:, 2] = result[:, 2] * (width / args.resize_width)
            result[:, 3] = result[:, 3] * (height / args.resize_height)
            for i in result:
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = dataset.xyxy2xywh(i[:-1])
                data['score'] = float(i[4])
                data['category_id'] = 1
                json_results.append(data)
        elif mode == 'global':
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = dataset.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = dataset.cat_ids[label]
                    json_results.append(data)
    return json_results


def det(args):
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # ---------------------------------------------------------------------------------------------------------
    if args.show_feature:
        cfg.model.test_cfg.show_feature = True
        cfg.model.test_cfg.feature_dir = os.path.join(args.work_dir, 'feature')
        os.makedirs(cfg.model.test_cfg.feature_dir, exist_ok=True)

    if args.assess_proposal_quality:
        if args.mode == 'local':
            img_sacle = (3000, 3000)
        elif args.mode == 'global':
            img_sacle = (800, 800)
        else:
            raise ValueError
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_sacle,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
                ])
        ]
        cfg.data.test.pipeline = test_pipeline
        cfg.data.test.assess_proposal_quality = True
        cfg.model.test_cfg.assess_proposal_quality = True
        # ---------------------------------------------------------------------------------------------------------

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=False,
                                   shuffle=False)

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model = MMDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(model, data_loader, show=args.show, out_dir=args.outdir)
    anns = det2json(dataset, outputs, args.mode)
    if args.dump_resfile:
        mmcv.dump(anns, args.ResFile)

    return anns


def set_param(self):
    p = self.params

    # -------------------------------- 调用cocoeval时所需的参数 --------------------------------------------------
    p.maxDets = [100000]
    if args.iou_mode == 'single':
        p.iouThrs = np.array([0.5])
    elif args.iou_mode == 'multiple':
        p.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    else:
        raise ValueError('wrong iou_mode')
    p.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    p.scoceThrs = args.score
    if args.area_mode == 'single':
        p.areaRng = [[0 ** 2, 1e5 ** 2]]
    elif args.area_mode == 'multiple':
        p.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    else:
        raise ValueError('wrong area_mode')
    # ---------------------------------------------------------------------------------------------------------

    self.params = p


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    # ---------------------------------------------------------------------------------------------------------
    parser.add_argument('work_dir', type=str)
    # please point out work_dir in this placef
    parser.add_argument('--score', default=0.3, type=float)
    # drop result if result's score smaller than args.score
    parser.add_argument('--weight_file', type=str, default='epoch_30.pth')
    # choose weight file to eval
    parser.add_argument('--dataset', type=str, choices=['xview'], default='xview')
    # only support xview now
    parser.add_argument('--show', default=False, type=bool)
    # whether to draw pred box to img
    parser.add_argument('--dump_resfile', default=False, type=bool)
    # whether to save ResFile.json
    parser.add_argument('--classwise', type=bool, default=False)
    # wherther to eval classwise ap
    parser.add_argument('--assess_proposal_quality', type=bool, default=False)
    # calculate proposal number of feature maps
    parser.add_argument('--show_feature', type=bool, default=False)
    # draw feature maps
    parser.add_argument('--pr_curve', type=bool, default=False)
    # draw pr_curve

    parser.add_argument('--iou_mode', choices=['single', 'multiple'], type=str, default='single')
    # if iou_mode is single, only eval iouThr=0.5
    # else if iou_mode is multiple, eval iouThr from 0.5 to 0.95
    parser.add_argument('--area_mode', choices=['single', 'multiple'], type=str, default='multiple')
    # if area_mode is single, only eval areaRng='all'
    # else if area_mode is multiple, eval areaRng='all', 'small', 'medium', 'large'
    # it takes very long time, more than 20 minutes, use carefully
    # ---------------------------------------------------------------------------------------------------------

    parser.add_argument('--resize_width', default=3000, type=float, help='the width of image after resize')
    parser.add_argument('--resize_height', default=3000, type=float, help='the height of image after resize')

    parser.add_argument('--eval', type=str, default='bbox', nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
                        help='eval types')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'dota' in args.work_dir:
        args.dataset = 'dota'

    if args.dataset == 'xview':
        args.val_path = 'data/xview/annotations/instances_val2017.json'
    elif args.dataset == 'dota':
        args.val_path = 'data/dota/val/DOTA_val.json'
    else:
        raise NotImplementedError('Wrong dataset name, please check arg.dataset')

    bare_name = os.path.basename(args.work_dir)
    if 'global' in bare_name or 'mode1' in bare_name or 'dota' in bare_name or 'xview' in bare_name:
        args.mode = 'global'
    elif 'local' in bare_name or 'mode2' in bare_name:
        args.mode = 'local'
    elif 'mode3' in bare_name:
        args.mode = 'global'
    elif 'ABFN' in bare_name:
        args.mode = 'global'
    else:
        raise ValueError('Wrong work_dir name')
    print('Start Evaluateing {} model'.format(bare_name))

    config_file = [i for i in os.listdir(args.work_dir) if i.endswith('.py')]
    assert len(config_file) == 1, 'please ensure work_dir only have one config file'
    config_file = config_file[0]
    args.config = os.path.join(args.work_dir, config_file)
    args.ResFile = os.path.join(args.work_dir, 'ResFile.json')
    args.checkpoint = os.path.join(args.work_dir, args.weight_file)
    if args.show is True:
        args.outdir = os.path.join(args.work_dir, 'out')
        os.makedirs(args.outdir, exist_ok=True)
    else:
        args.outdir = None

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.show or not os.path.isfile(args.ResFile):
        tik = time.time()
        anns = det(args)
        tok = time.time()
        print()
        print('eval time:{}'.format(tok - tik))
        print('FPS:{}'.format(245 / (tok - tik)))
        evaluate(args, anns)
    else:
        print('Load ResFile From Local...')
        evaluate(args, args.ResFile)
