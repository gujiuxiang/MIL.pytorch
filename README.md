# PyTorch implementation of Multiple-instance learning


## Updates

- [ ] Training/Testing on MS COCO
- [x] Testing on single image

## About
This repository contains the MIL implementation for the experiments in:
```
https://github.com/s-gupta/visual-concepts
```

If you want to use the original caffe implementation, pls follow the instructions below:
```bash
git clone git@github.com:s-gupta/caffe.git code/caffe
cd code/caffe
git checkout mil
make -j 16
make pycaffe
cd ../../
```

Or, you can use the transformed PyTorch model of mine. You can download them from [Baidu Pan](https://pan.baidu.com/s/1pLJkp0f). Files included in that directory are:
```
model/coco_valid1_eval.pkl
model/mil.pth
model/precision_score.pkl
```

## Test results
You can change the url in test.py to your testing image. Then run:
```python
python test.py
```

<img src="http://img1.10bestmedia.com/Images/Photos/333810/Montrose_54_990x660.jpg" alt="test0" height="" width="640">

```
['beach', 'dog', 'brown', 'standing', 'people', 'his', 'sandy', 'white', 'sitting', 'laying']
[1.0, 0.62, 0.62, 0.5, 0.45000000000000001, 0.42999999999999999, 0.37, 0.35999999999999999, 0.34000000000000002, 0.28000000000000003]
```

<img src="https://www.askideas.com/media/23/Funny-Cat-Reaction-First-Tiime-Seeing-Herself-In-The-Mirror.jpg" alt="test1" height="" width="640">

```
['cat', 'sink', 'sitting', 'black', 'bathroom', 'white', 'top', 'sits', 'counter', 'looking']
[1.0, 0.68000000000000005, 0.56999999999999995, 0.56000000000000005, 0.39000000000000001, 0.34999999999999998, 0.31, 0.26000000000000001, 0.23000000000000001, 0.22]
```
