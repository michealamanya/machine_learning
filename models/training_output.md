Downloading FER2013 dataset...
Using Colab cache for faster access to the 'fer2013' dataset.
FACIAL EXPRESSION RECOGNITION - HIGH ACCURACY EDITION
Device: cuda
Using pretrained ResNet-18: True
Label smoothing: 0.1
TTA enabled: True
TTA transforms: 5
Loading datasets...
Loaded 28709 images from /kaggle/input/fer2013/train
Loaded 7178 images from /kaggle/input/fer2013/test

Class weights for balancing: tensor([1.0266, 9.4066, 1.0010, 0.5684, 0.8260, 0.8491, 1.2934],
       device='cuda:0')
Training samples: 24403
Validation samples: 4306
Test samples: 7178

Class distribution:
  Angry: 3995
  Disgust: 436
  Fear: 4097
  Happy: 7215
  Neutral: 4965
  Sad: 4830
  Surprise: 3171
Initializing ResNet-18 model...
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
100%|██████████| 44.7M/44.7M [00:00<00:00, 199MB/s]
Total parameters: 11,310,151

----------------------------------------------------------------------
Starting training...
----------------------------------------------------------------------
Epoch [  1/100] | Loss: 2.0452/1.9250 | Acc: 22.90%/41.34% | LR: 0.001000
  Best model saved! (Val Acc: 41.34%)
Epoch [  2/100] | Loss: 1.8810/1.8662 | Acc: 36.24%/38.11% | LR: 0.001000
Epoch [  3/100] | Loss: 1.8386/1.8625 | Acc: 40.08%/41.11% | LR: 0.001000
Epoch [  4/100] | Loss: 1.8148/1.8344 | Acc: 41.45%/45.22% | LR: 0.001000
  Best model saved! (Val Acc: 45.22%)
Epoch [  5/100] | Loss: 1.7991/1.8039 | Acc: 41.90%/46.52% | LR: 0.001000
  Best model saved! (Val Acc: 46.52%)
Epoch [  6/100] | Loss: 1.7860/1.7686 | Acc: 43.24%/47.47% | LR: 0.001000
  Best model saved! (Val Acc: 47.47%)
Epoch [  7/100] | Loss: 1.7769/1.8208 | Acc: 44.31%/39.78% | LR: 0.001000
Epoch [  8/100] | Loss: 1.7699/1.7876 | Acc: 45.11%/50.37% | LR: 0.001000
  Best model saved! (Val Acc: 50.37%)
Epoch [  9/100] | Loss: 1.7538/1.7741 | Acc: 46.68%/47.93% | LR: 0.001000
Epoch [ 10/100] | Loss: 1.7472/1.7843 | Acc: 46.88%/44.61% | LR: 0.000500
Epoch [ 11/100] | Loss: 1.6927/1.6912 | Acc: 50.42%/49.51% | LR: 0.000500
Epoch [ 12/100] | Loss: 1.6779/1.6993 | Acc: 51.50%/49.61% | LR: 0.000500
Epoch [ 13/100] | Loss: 1.6609/1.6774 | Acc: 52.07%/53.44% | LR: 0.000500
  Best model saved! (Val Acc: 53.44%)
Epoch [ 14/100] | Loss: 1.6552/1.6824 | Acc: 52.67%/51.93% | LR: 0.000500
Epoch [ 15/100] | Loss: 1.6474/1.6751 | Acc: 52.86%/54.30% | LR: 0.000500
  Best model saved! (Val Acc: 54.30%)
Epoch [ 16/100] | Loss: 1.6372/1.6803 | Acc: 53.65%/53.09% | LR: 0.000500
Epoch [ 17/100] | Loss: 1.6265/1.6694 | Acc: 54.26%/53.25% | LR: 0.000500
Epoch [ 18/100] | Loss: 1.6245/1.6668 | Acc: 54.58%/55.62% | LR: 0.000500
  Best model saved! (Val Acc: 55.62%)
Epoch [ 19/100] | Loss: 1.6174/1.6661 | Acc: 55.10%/54.09% | LR: 0.000500
Epoch [ 20/100] | Loss: 1.6091/1.6646 | Acc: 55.69%/53.39% | LR: 0.000500
Epoch [ 21/100] | Loss: 1.5997/1.6343 | Acc: 56.06%/56.13% | LR: 0.000500
  Best model saved! (Val Acc: 56.13%)
Epoch [ 22/100] | Loss: 1.5900/1.6584 | Acc: 56.53%/51.67% | LR: 0.000500
Epoch [ 23/100] | Loss: 1.5889/1.6637 | Acc: 56.26%/54.11% | LR: 0.000500
Epoch [ 24/100] | Loss: 1.5833/1.6574 | Acc: 56.82%/55.90% | LR: 0.000500
Epoch [ 25/100] | Loss: 1.5802/1.6226 | Acc: 56.92%/55.99% | LR: 0.000500
Epoch [ 26/100] | Loss: 1.5674/1.6418 | Acc: 57.22%/55.88% | LR: 0.000500
Epoch [ 27/100] | Loss: 1.5697/1.6792 | Acc: 58.04%/55.06% | LR: 0.000500
Epoch [ 28/100] | Loss: 1.5696/1.6368 | Acc: 57.44%/54.76% | LR: 0.000500
Epoch [ 29/100] | Loss: 1.5626/1.6409 | Acc: 57.91%/56.48% | LR: 0.000250
  Best model saved! (Val Acc: 56.48%)
Epoch [ 30/100] | Loss: 1.5364/1.5906 | Acc: 59.19%/57.66% | LR: 0.000250
  Best model saved! (Val Acc: 57.66%)
Epoch [ 31/100] | Loss: 1.5166/1.6068 | Acc: 59.72%/58.10% | LR: 0.000250
  Best model saved! (Val Acc: 58.10%)
Epoch [ 32/100] | Loss: 1.5058/1.6032 | Acc: 60.25%/58.73% | LR: 0.000250
  Best model saved! (Val Acc: 58.73%)
Epoch [ 33/100] | Loss: 1.5112/1.5800 | Acc: 60.55%/59.50% | LR: 0.000250
  Best model saved! (Val Acc: 59.50%)
Epoch [ 34/100] | Loss: 1.5048/1.6032 | Acc: 60.62%/58.22% | LR: 0.000250
Epoch [ 35/100] | Loss: 1.4948/1.5894 | Acc: 61.14%/59.48% | LR: 0.000250
Epoch [ 36/100] | Loss: 1.4853/1.6087 | Acc: 61.75%/56.90% | LR: 0.000250
Epoch [ 37/100] | Loss: 1.4774/1.5886 | Acc: 61.57%/60.13% | LR: 0.000125
  Best model saved! (Val Acc: 60.13%)
Epoch [ 38/100] | Loss: 1.4684/1.5793 | Acc: 62.51%/60.50% | LR: 0.000125
  Best model saved! (Val Acc: 60.50%)
Epoch [ 39/100] | Loss: 1.4545/1.5737 | Acc: 62.91%/59.22% | LR: 0.000125
Epoch [ 40/100] | Loss: 1.4504/1.5761 | Acc: 63.02%/59.24% | LR: 0.000125
Epoch [ 41/100] | Loss: 1.4518/1.5744 | Acc: 63.44%/60.43% | LR: 0.000125
Epoch [ 42/100] | Loss: 1.4428/1.5886 | Acc: 63.80%/59.52% | LR: 0.000125
Epoch [ 43/100] | Loss: 1.4412/1.5757 | Acc: 63.81%/60.13% | LR: 0.000063
Epoch [ 44/100] | Loss: 1.4330/1.5654 | Acc: 63.89%/60.92% | LR: 0.000063
  Best model saved! (Val Acc: 60.92%)
Epoch [ 45/100] | Loss: 1.4315/1.5728 | Acc: 64.39%/60.47% | LR: 0.000063
Epoch [ 46/100] | Loss: 1.4314/1.5737 | Acc: 64.34%/61.01% | LR: 0.000063
  Best model saved! (Val Acc: 61.01%)
Epoch [ 47/100] | Loss: 1.4151/1.5855 | Acc: 65.06%/60.75% | LR: 0.000063
Epoch [ 48/100] | Loss: 1.4233/1.5683 | Acc: 64.55%/60.31% | LR: 0.000031
Epoch [ 49/100] | Loss: 1.4130/1.5655 | Acc: 64.98%/59.85% | LR: 0.000031
Epoch [ 50/100] | Loss: 1.4109/1.5668 | Acc: 64.82%/60.61% | LR: 0.000031
Epoch [ 51/100] | Loss: 1.4162/1.5612 | Acc: 65.21%/60.75% | LR: 0.000031
Epoch [ 52/100] | Loss: 1.4122/1.5710 | Acc: 64.94%/60.57% | LR: 0.000031
Epoch [ 53/100] | Loss: 1.4106/1.5510 | Acc: 65.02%/60.45% | LR: 0.000031
Epoch [ 54/100] | Loss: 1.4015/1.5737 | Acc: 65.46%/60.64% | LR: 0.000031
Epoch [ 55/100] | Loss: 1.4005/1.5693 | Acc: 65.52%/61.12% | LR: 0.000031
  Best model saved! (Val Acc: 61.12%)
Epoch [ 56/100] | Loss: 1.4029/1.5568 | Acc: 65.81%/60.94% | LR: 0.000031
Epoch [ 57/100] | Loss: 1.4032/1.5674 | Acc: 65.48%/59.94% | LR: 0.000016
Epoch [ 58/100] | Loss: 1.3952/1.5457 | Acc: 65.57%/62.01% | LR: 0.000016
  Best model saved! (Val Acc: 62.01%)
Epoch [ 59/100] | Loss: 1.3889/1.5610 | Acc: 66.54%/61.12% | LR: 0.000016
Epoch [ 60/100] | Loss: 1.3977/1.5649 | Acc: 65.73%/61.26% | LR: 0.000016
Epoch [ 61/100] | Loss: 1.3973/1.5807 | Acc: 66.02%/60.61% | LR: 0.000016
Epoch [ 62/100] | Loss: 1.3985/1.5588 | Acc: 65.94%/60.61% | LR: 0.000008
Epoch [ 63/100] | Loss: 1.3993/1.5630 | Acc: 65.73%/61.05% | LR: 0.000008
Epoch [ 64/100] | Loss: 1.3942/1.5628 | Acc: 66.04%/61.70% | LR: 0.000008
Epoch [ 65/100] | Loss: 1.3968/1.5637 | Acc: 65.75%/61.10% | LR: 0.000008
Epoch [ 66/100] | Loss: 1.3953/1.5675 | Acc: 66.04%/59.80% | LR: 0.000004
Epoch [ 67/100] | Loss: 1.3894/1.5584 | Acc: 66.18%/61.05% | LR: 0.000004
Epoch [ 68/100] | Loss: 1.3900/1.5594 | Acc: 65.84%/60.96% | LR: 0.000004

 Early stopping at epoch 68
Training complete! Best val accuracy: 62.01%
Evaluating on test set...
Standard Test Accuracy: 64.07%

Running Test-Time Augmentation (5 augmentations)...
TTA Test Accuracy: 64.70% (+0.63%)

----------------------------------------------------------------------
Classification Report:
              precision    recall  f1-score   support

       angry     0.5680    0.5846    0.5761       958
     disgust     0.5448    0.6577    0.5959       111
        fear     0.5380    0.3320    0.4106      1024
       happy     0.8777    0.8416    0.8593      1774
     neutral     0.5376    0.7429    0.6238      1233
         sad     0.5659    0.4579    0.5062      1247
    surprise     0.6828    0.8315    0.7499       831

    accuracy                         0.6470      7178
   macro avg     0.6164    0.6355    0.6174      7178
weighted avg     0.6476    0.6470    0.6390      7178

Confusion matrix saved: confusion_matrix.png
Training curves saved: training_metrics.png
Exporting model to ONNX...
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipython-input-245357860.py in <cell line: 0>()
    548     dummy_input = torch.randn(1, 3, 48, 48).to(Config.DEVICE)
    549 
--> 550     torch.onnx.export(
    551         model,
    552         dummy_input,

2 frames
/usr/local/lib/python3.12/dist-packages/torch/onnx/_internal/exporter/_core.py in <module>
     16 from typing import Any, Callable, Literal
     17 
---> 18 import onnxscript
     19 import onnxscript.evaluator
     20 from onnxscript import ir

ModuleNotFoundError: No module named 'onnxscript'

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
-----------------------------------