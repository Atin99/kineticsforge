# Kaggle and Colab Training Workspace

This folder is the only place intended for long neural training runs.

Fast local data check:

```bash
python -m training.colab_kaggle.kaggle_colab_data_bootstrap --profile smoke
python -m training.colab_kaggle.kaggle_colab_train --task all --profile quick
```

Main free GPU run:

```bash
pip install -r training/colab_kaggle/requirements_kaggle.txt
python -m training.colab_kaggle.kaggle_colab_data_bootstrap --profile foundation --allow-download --max-real-gb 0.45
python -m training.colab_kaggle.kaggle_colab_train --task all --profile standard --device auto
```

Long run:

```bash
python -m training.colab_kaggle.kaggle_colab_train --task cathode --profile deep --device auto
python -m training.colab_kaggle.kaggle_colab_train --task bms --profile deep --device auto
python -m training.colab_kaggle.kaggle_colab_train --task recycling --profile deep --device auto
```

Outputs go to `checkpoints/` and run ledgers go to `training/colab_kaggle/runs/`.
