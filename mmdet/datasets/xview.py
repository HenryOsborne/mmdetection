import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset

import time
import datetime
import copy
from pycocotools import mask as maskUtils
import os
import shutil


@DATASETS.register_module()
class XviewDataset(CustomDataset):
    CLASSES = ('root',)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def det2json(self, results, mode):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            if mode == 'local':
                imgInfo = self.data_infos[idx]
                width = imgInfo['width']
                height = imgInfo['height']
                result[:, 0] = result[:, 0] * (width / 3000)
                result[:, 1] = result[:, 1] * (height / 3000)
                result[:, 2] = result[:, 2] * (width / 3000)
                result[:, 3] = result[:, 3] * (height / 3000)
                for i in result:
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(i[:-1])
                    data['score'] = float(i[4])
                    data['category_id'] = 1
                    json_results.append(data)
            elif mode == 'global':
                for label in range(len(result)):
                    bboxes = result[label]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(bboxes[i][4])
                        data['category_id'] = self.cat_ids[label]
                        json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix, mode=None):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            mode: 'global' or 'local'
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self.det2json(results, mode)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, mode=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            mode: 'global' or 'local'
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix, mode)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100000,),
                 iou_thrs=None,
                 metric_items=None,
                 mode='global',
                 args=None):
        """Evaluation in COCO protocol.

        Args:
            mode: 'global' or 'local'
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.array([0.5])
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix, mode)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            set_param(cocoEval)
            if len(cocoEval.params.areaRng) == 10:
                coco_metric_names = {
                    'AP': 0, 'AR': 1, 'PR': 2,
                    'AP_s': 3, 'AR_s': 4, 'PR_s': 5,
                    'AP_m': 6, 'AR_m': 7, 'PR_m': 8,
                    'AP_l': 9, 'AR_l': 10, 'PR_l': 11,
                }
            else:
                coco_metric_names = {
                    'AP': 0, 'AR': 1, 'PR': 2,
                }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                coco_evaluate(cocoEval, mode)
                coco_accumulate(cocoEval)
                coco_summarize(cocoEval,args)
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_evaluate(cocoEval, mode)
                coco_accumulate(cocoEval)
                coco_summarize(cocoEval, args)
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
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
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    if len(cocoEval.params.areaRng) == 10:
                        metric_items = [
                            'AP', 'AR', 'PR',
                            'AP_s', 'AR_s', 'PR_s',
                            'AP_m', 'AR_m', 'PR_m',
                            'AP_l', 'AR_l', 'PR_l',
                        ]
                    else:
                        metric_items = ['AP', 'AR', 'PR']

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


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


def evaluateImg(self, imgId, catId, aRng, score, prog_bar, mode):
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

        if mode == 'global':
            resized_width = bbox_width * (800 / img_width)
            resized_heigth = bbox_heigth * (800 / img_height)
        elif mode == 'local':
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


def coco_evaluate(self, mode):
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
    self.evalImgs = [evaluateImg(self, imgId, catId, areaRng, p.scoceThrs, prog_bar, mode)
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

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
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
        if args is not None:
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
        else:
            stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=None)  # AP
            stats[1] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=None)  # AR
            stats[2] = _summarize(2, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=None)  # PR
            stats[3] = _summarize(1, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[4] = _summarize(0, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[5] = _summarize(2, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[6] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[7] = _summarize(0, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[8] = _summarize(2, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[9] = _summarize(1, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[10] = _summarize(0, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=None)
            stats[11] = _summarize(2, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[-1], write_handle=None)
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


def set_param(self):
    p = self.params

    ############################ 调用cocoeval时所需的参数 ########################################
    area_mode = 'multiple'
    p.maxDets = [100000]
    p.iouThrs = np.array([0.5])
    p.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    p.scoceThrs = 0.3
    if area_mode == 'single':
        p.areaRng = [[0 ** 2, 1e5 ** 2]]
    elif area_mode == 'multiple':
        p.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    else:
        raise ValueError('wrong area_mode')
    ########################################################################################

    self.params = p
