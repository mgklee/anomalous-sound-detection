# Anomalous Sound Detection

This repository contains our work for the final project of the course "AI Convergence Capstone Design," offered by Professor [Jung-Woo Choi](https://www.sound.kaist.ac.kr/) of the [Department of Electrical Engineering at KAIST](https://ee.kaist.ac.kr/). We tackle [DCASE2020 Challenge Task 2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds): Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring. This task focuses on developing systems capable of identifying anomalous machine operating sounds without prior exposure to anomalous samples during training.

## Datasets

* [development dataset](https://zenodo.org/record/3678171)
* [additional training dataset](https://zenodo.org/record/3727685)
* (optional) [evaluation dataset](https://zenodo.org/record/3841772)

The first two datasets should be merged and organized as follows.

```
dev_data/
├── fan/
    ├── train/
    ├── test/
├── pump/
    ├── train/
    ├── test/
├── slider/
    ├── train/
    ├── test/
├── ToyCar/
    ├── train/
    ├── test/
├── ToyConveyor/
    ├── train/
    ├── test/
├── valve/
    ├── train/
    ├── test/
```

## Training

To train a model, first specify the path to the dataset (`dev_data` in the above example) in the `root_path` of the `config.yaml` file, and then run:
```
python train.py
```

## Evaluation

To evaluate the trained model, run:
```
python eval.py
```

## Results

| Machine Type |   AUC(%)   |   pAUC(%)  |
|--------------|:----------:|:----------:|
| fan          |   98.140   |   94.260   |
| pump         |   96.541   |   89.220   |
| slider       |   99.421   |   97.048   |
| ToyCar       |   96.349   |   89.594   |
| ToyConveyor  |   82.940   |   68.800   |
| valve        |   99.986   |   99.926   |
| **Average**  | **95.563** | **89.808** |

## References

* S. Choi and J. -W. Choi, "Noisy-Arcmix: Additive Noisy Angular Margin Loss Combined With Mixup For Anomalous Sound Detection," _ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, Seoul, Korea, Republic of, 2024, pp. 516-520.
* D. Kong, H. Yu and G. Yuan, "Multi-Spectral and Multi-Temporal Features Fusion With SE Network for Anomalous Sound Detection," in _IEEE Access_, vol. 12, pp. 167262-167277, 2024.
