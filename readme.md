# KITTI 道路分割

- 已有模型: Unet, Uresnet(resnet作为encoder), FPN(resnet作为encoder)
- 参数：见train.py
- 依赖：pytorch, opencv-python, pillow, albumentations, matplotlib, tqdm ...
  ### 任务

0. **实现测试指标，可视化结果**

   - 实现test.py, 计算指标和产生可视化结果
   - 指标参考官网 https://www.cvlibs.net/datasets/kitti/eval_road.php
1. **不同模型，from scratch，相同的loss，比较最好结果**

| DICE-Loss | MaxF   | AP     | PRE    | REC    | ACC    | Inference Time |
| --------- | ------ | ------ | ------ | ------ | ------ | -------------- |
| Unet      | 0.9295 | 0.0229 | 0.9295 | 0.9327 | 0.9821 | 5.88           |
| UResnet   | 0.8727 | 0.0229 | 0.792  | 0.9409 | 0.9615 | 5.75           |
| FPN       | 0.8705 | 0.0229 | 0.8027 | 0.9217 | 0.9616 | 5.38           |

2. **相同模型，不同loss（例如下面几个或者其他的），比较最好结果**

| MODEL - ?     | MaxF | AP | PRE | REC |
| ------------- | ---- | -- | --- | --- |
| iou           |      |    |     |     |
| bce           |      |    |     |     |
| bce+dice      |      |    |     |     |
| anything else |      |    |     |     |

3. **使用imagenet pre-trained参数初始化模型与否，比较最好结果**

|             | MaxF | AP | PRE | REC |
| ----------- | ---- | -- | --- | --- |
| UResnet     |      |    |     |     |
| FPN         |      |    |     |     |
| UResnet+pre |      |    |     |     |
| FPN+pre     |      |    |     |     |

- 参考：
  - https://www.kaggle.com/code/hossamemamo/kitti-road-segmentation-pytorch-unet-from-scratch
  - https://github.com/gasparian/multiclass-semantic-segmentation
