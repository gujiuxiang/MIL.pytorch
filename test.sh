#!/usr/bin/env bash

clear
#-----------------------------------------------------------------------------------------------------------------------
func_cls()
{
    CUDA_VISIBLE_DEVICES=1 python extract_cls_oid.py --split coco_val2014
    CUDA_VISIBLE_DEVICES=1 python extract_cls_oid.py --split coco_test2014
    CUDA_VISIBLE_DEVICES=1 python extract_cls_oid.py --split coco_train2014
    CUDA_VISIBLE_DEVICES=1 python extract_cls_oid.py --split coco_test2015
}

func_det()
{   export PYTHONPATH=$PYTHONPATH:/home/jxgu/github/MIL.pytorch/misc/models/research/object_detection
    CUDA_VISIBLE_DEVICES=1 python extract_det_oid.py --split coco_val2014
    CUDA_VISIBLE_DEVICES=1 python extract_det_oid.py --split coco_test2014
    CUDA_VISIBLE_DEVICES=1 python extract_det_oid.py --split coco_train2014
    CUDA_VISIBLE_DEVICES=1 python extract_det_oid.py --split coco_test2015
}
func_cls