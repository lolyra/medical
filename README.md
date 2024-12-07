# Enhancing Hybrid CNN-ViT Models with Fisher Vector Encoding for Scalable Medical Image Classification

In order to execute the code, it is necessary to have Python>=3.9 installed as well as the packages listed in `requirements.txt`.

Fine-tuning of a model can be performed running

```
python train_model.py -d <dataset-name>
```

Similarly, the GMM can be estimated by

```
python train_gmm.py -d <dataset-name>
```

and the classifier can be trained by

```
python train_classifier.py -d <dataset-name>
```

Tests of the fine-tuned backbone and classifier can be performed by

```
python test_model.py -d <dataset-name>
python test_classifier.py -d <dataset-name>
```

Fine-tuning and testing other models in the literature. Models must be implemented in [timm library](https://github.com/huggingface/pytorch-image-models).

```
python test.py -d <dataset-name> -m <model-name>
```

Parameters can be adjusted in `variables.py`.
