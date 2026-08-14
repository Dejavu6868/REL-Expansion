import os
import cv2
import argparse
import csv
import json
import numpy as np

import torch
import torch.nn as nn
from PIL import Image

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import SourceAlignedRELValPre, ValPre

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        valid_label = (label == config.background) | ((label >= 0) & (label < self.class_num))
        if not np.all(valid_label):
            invalid_values = np.unique(label[~valid_label]).tolist()
            raise ValueError('Invalid label values for {}: {}'.format(name, invalid_values))
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.class_num, pred, label)
        results_dict = {'name': name, 'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        prediction_limit = int(os.environ.get('CMX_PREDICTION_LIMIT', '0'))
        saved_predictions = getattr(self, '_saved_predictions', 0)
        if self.save_path is not None and (prediction_limit <= 0 or saved_predictions < prediction_limit):
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'
            ensure_dir(os.path.dirname(os.path.join(self.save_path, fn)))
            ensure_dir(os.path.dirname(os.path.join(self.save_path+'_color', fn)))

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = self.dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)
            self._saved_predictions = saved_predictions + 1

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        metrics_path = os.environ.get('CMX_METRICS_JSON')
        if metrics_path:
            os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
            payload = {
                'miou': float(mean_IoU),
                'pixel_accuracy': float(pixel_acc),
                'mean_pixel_accuracy': float(mean_pixel_acc),
                'frequency_weighted_iou': float(freq_IoU),
                'per_class_iou': {
                    name: float(value) for name, value in zip(self.dataset.class_names, iou)
                },
                'labeled_pixels': int(labeled),
                'correct_pixels': int(correct),
                'confusion_matrix': hist.astype(np.int64).tolist(),
                'evaluation': 'single-scale, no-flip, Area 5, epoch 32 preregistered',
            }
            temporary = '{}.tmp'.format(metrics_path)
            with open(temporary, 'w') as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write('\n')
            os.replace(temporary, metrics_path)
            csv_path = os.path.join(os.path.dirname(metrics_path), 'per_class_iou.csv')
            with open(csv_path, 'w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['class', 'iou'])
                writer.writerows(zip(self.dataset.class_names, [float(value) for value in iou]))
        per_image_path = os.environ.get('CMX_PER_IMAGE_METRICS_CSV')
        if per_image_path:
            from stage2b.metrics import write_per_image_csv
            write_per_image_csv(per_image_path, results, self.dataset.class_names)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'x_mode': getattr(config, 'x_mode', 'precomputed'),
                    'rel_impl': getattr(config, 'rel_impl', None),
                    'x_online_relplus': getattr(config, 'x_online_relplus', False),
                    'depth_root': getattr(config, 'depth_root_folder', None),
                    'depth_format': getattr(config, 'depth_format', '.png'),
                    'pose_root': getattr(config, 'pose_root_folder', None),
                    'pose_format': getattr(config, 'pose_format', '.json'),
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    if getattr(config, 'x_mode', 'precomputed') == 'rel_source_aligned':
        val_pre = SourceAlignedRELValPre()
    else:
        val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
