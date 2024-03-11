import pickle
import time

import numpy as np
import torch
import tqdm
import os

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import time


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        #cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST=[0.25]
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0) #dict.get(key,指定值) 获得键的值，如果不存在则返回指定值。有gt但gt是0.
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test, save_to_file=True, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()


    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        #begin = time.time()
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)
        # if i<1:
        #     print(pred_dicts)
            # [{'pred_boxes': tensor([], device='cuda:0', size=(0, 7)), 'pred_scores': tensor([], device='cuda:0'),
            #   'pred_labels': tensor([], device='cuda:0', dtype=torch.int64), 'WBF': True},
            #  {'pred_boxes': tensor([], device='cuda:0', size=(0, 7)), 'pred_scores': tensor([], device='cuda:0'),
            #   'pred_labels': tensor([], device='cuda:0', dtype=torch.int64),'WBF': True},
            #  {'pred_boxes': tensor([], device='cuda:0', size=(0, 7)), 'pred_scores': tensor([], device='cuda:0'),
            # 'pred_labels': tensor([], device='cuda:0', dtype=torch.int64),'WBF': True},
            # {'pred_boxes': tensor([], device='cuda:0', size=(0, 7)),'pred_scores': tensor([], device='cuda:0'),
            # 'pred_labels': tensor([], device='cuda:0', dtype=torch.int64), 'WBF': True}]
            # [{'pred_boxes': tensor([[18.8337, 24.3252, -2.0070, 3.4818, 1.6680, 1.6483, 0.9984],
            #                         [18.9280, 24.4447, -1.9655, 3.7862, 1.7079, 1.6558, 0.9958]],
            #                        device='cuda:0'), 'pred_scores': tensor([0.6536, 0.5588], device='cuda:0'),
            #   'pred_labels': tensor([1, 1], device='cuda:0'), 'WBF': True},
            #  {'pred_boxes': tensor([[18.9405, 24.3900, -1.9791, 3.7400, 1.7139, 1.6997, 0.9851],
            #                         [19.0214, 24.4934, -1.9426, 3.9874, 1.7337, 1.6706, 0.9861],
            #                         [18.9984, 24.4370, -2.0044, 3.8919, 1.6963, 1.6571, 1.0024],
            #                         [18.8620, 24.3962, -1.9992, 3.6666, 1.7169, 1.7240, 1.0103]],
            #                        device='cuda:0'),
            #   'pred_scores': tensor([0.6219, 0.6474, 0.5695, 0.4783], device='cuda:0'),
            #   'pred_labels': tensor([1, 1, 1, 1], device='cuda:0'), 'WBF': True},
            #  {'pred_boxes': tensor([[18.9079, 24.3077, -1.9824, 3.5141, 1.6566, 1.6519, 1.0196],
            #                         [18.9988, 24.4215, -1.9506, 3.8254, 1.7026, 1.6744, 1.0253]],
            #                        device='cuda:0'), 'pred_scores': tensor([0.7373, 0.6983], device='cuda:0'),
            #   'pred_labels': tensor([1, 1], device='cuda:0'), 'WBF': True},
            #  {'pred_boxes': tensor([[18.8414, 24.2367, -1.9451, 3.3579, 1.6358, 1.6445, 0.9943],
            #                         [18.9063, 24.3521, -1.8971, 3.6095, 1.6600, 1.6206, 1.0062]],
            #                        device='cuda:0'), 'pred_scores': tensor([0.7377, 0.6988], device='cuda:0'),
            #   'pred_labels': tensor([1, 1], device='cuda:0'), 'WBF': True}]
            # print(11111111111111111111111111111111111)
            # print(type(batch_dict)) #不全为0的很多维的词典 <class 'dict'>
            # print(batch_dict)
            # for i in batch_dict:
            #     print(i)
            # print(2222222222222222222222222222222222222222)
            # print(ret_dict)# {'gt': 0, 'roi_0.25': 0, 'rcnn_0.25': 0}
            # print(333333333333333333333333)
        disp_dict = {}
        #end = time.time()
        #print(end-begin)

        statistics_info(cfg, ret_dict, metric, disp_dict)
        # logger.info('metric[gt_num:%f]' % (metric['gt_num'])) 每一个测试样本都是0
        annos = dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None)
        # if i<1:
            # print(annos)
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1) #cur_thresh=0.25,gt_num_cnt=0
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    #Average predicted number of objects(725 samples): 4.058

    pklpath = result_dir / 'result.pkl'
    if os.path.exists(pklpath):
        pklpath = result_dir / ('result_'+str(time.time())[:10]+'.pkl')

    with open(pklpath, 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    
    return ret_dict


if __name__ == '__main__':
    pass
