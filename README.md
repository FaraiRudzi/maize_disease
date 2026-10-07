# Maize Leaf Check

A web system that identifies maize leaf blight and maize streak virus from a smartphone photo, gives treatment advice,
and shows extension officers where cases are being reported.

Stack: Python, TensorFlow (MobileNetV2 transfer learning), Flask, SQLite.

## What is in this folder

| File or folder | Purpose |
|---|---|
| `download_dataset.py` | Lists and downloads the Makerere image archives through Harvard Dataverse's public API |
| `prepare_dataset.py` | Cleans, resizes and splits the Makerere dataset into train, validation and test folders |
| `train_model.py` | Trains the classifier and writes the model, metrics, confusion matrix and training curves |
| `evaluate_local.py` | Tests the trained model on your own Zimbabwean photos |
| `app.py`, `predictor.py`, `db.py`, `config.py` | The web application |
| `advice.json` | Treatment advice shown to farmers (draft: have an extension officer review it) |
| `templates/`, `static/` | Web pages and styling |
| `tests/` | Automated tests for the web app (`python -m unittest discover -s tests`) |
| `database/schema.sql` | The database table, for your documentation |

## 1. Set up

```
python -m venv venv
venv\Scripts\activate          (Windows)      or      source venv/bin/activate   (Linux, Mac)
pip install -r requirements.txt
```

## 2. Get the dataset from the Dataverse API

Dataset: Makerere University Maize Image Dataset (Babirye et al., 2022), 15,821 images labelled healthy, maize leaf
blight (mlb) or maize streak virus (msv). Licence: CC0 1.0. Official page (Harvard Dataverse):
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LPGHKK

The project includes `download_dataset.py`, which uses Harvard Dataverse's public API. No API key is needed. First
list the available image archives and sizes:

```
python download_dataset.py
```

Download all six class archives (approximately 12.7 GB; check the actual size printed by the API first):

```
python download_dataset.py --download --classes all
```

The downloader saves archives under `dataset/raw/healthy`, `dataset/raw/maize_leaf_blight`, and
`dataset/raw/maize_streak_virus`. It resumes interrupted transfers and checks each file against the MD5 checksum
provided by Dataverse. To download only some archives, use for example `--download --classes healthy blight`.

The API files are RAR archives. Install 7-Zip, then extract each archive into its existing class folder (for example,
extract `Healthy_1.rar` into `dataset/raw/healthy`). Keep the extracted images under the corresponding class folder.
The preparation script finds class folders recursively. It also supports your own labeled image folders directly under
`dataset`, including common diseases and nutrient deficiencies.

Training cannot start from the archive files themselves. The `dataset_split` directory is created by the next step;
it does not exist until you have extracted image files and run dataset preparation.

## 3. Prepare the dataset

For the API data, pass `dataset/raw`; for labeled class folders already under `dataset`, pass `dataset`:

```
python prepare_dataset.py --src dataset --dst dataset_split
```

Recognized folder names include `Common rust`, `Gray leaf spot`, `Healthy`, `Magnessium`, `Maize streak virus`,
`Nitrogen`, `Northern Leaf Blight`, `Phosphorous`, and `Pottasium`. Nutrient folders are normalized to
`magnesium_deficiency`, `nitrogen_deficiency`, `phosphorus_deficiency`, and `potassium_deficiency`.
Northern leaf blight remains a distinct class from maize leaf blight. The script removes exact duplicate files, shrinks
images to 512 px on the longest side, and makes 70/15/15 train/validation/test splits per class. Check the printed
counts for every class before training.

## 4. Train

```
python train_model.py --data dataset_split --out model
```

Stage 1 trains a new classification layer on a frozen MobileNetV2, stage 2 fine-tunes the last layers. Outputs in `model/`:
`maize_model.keras`, `class_names.json`, `metrics.json`, `confusion_matrix.png`, `training_curves.png`.
Use these for your results chapter. Quick trial: add `--epochs 1 --fine-tune-epochs 1`.

## 5. Run the web app

```
python app.py
```

Open http://127.0.0.1:5000. To let phones on the same Wi-Fi reach it: `set HOST=0.0.0.0` (Windows) or `HOST=0.0.0.0 python app.py`.

Officer dashboard: set a password first, then sign in at `/dashboard`:

```
set DASHBOARD_PASSWORD=choose-a-password
set SECRET_KEY=a-long-random-string
```

Other settings (environment variables): `CONFIDENCE_THRESHOLD` (default 0.70, below this the app says "not sure"),
`UPLOAD_DIR`, `MODEL_PATH`.

## 6. Test on Zimbabwean photos

Collect photos, have an extension officer confirm the labels, and put them in `local_test/healthy`,
`local_test/maize_leaf_blight` and `local_test/maize_streak_virus`. Then:

```
python evaluate_local.py --folder local_test
```

Compare the result with the test-set result from step 4. That gap is a finding for your report.

## Notes you should read

- **Not run here:** The web app and dataset script were tested with automated tests and sample data. `train_model.py`,
  `evaluate_local.py` and model loading need TensorFlow and the real dataset, so run them yourself and fix any version
  issue that shows up.
- **Advice text:** `advice.json` is general guidance, not an agronomist's recommendation. Have it reviewed before farmers use it.
- **Scope of the model:** It knows three classes only. A photo of anything else (another disease, a pest, a non-maize plant) will
  still get one of the three labels, so the confidence threshold and the "ask an officer" message matter.
- **Privacy:** Photos are re-encoded on upload, which removes GPS data stored inside the file. Locations are rounded to about
  1 km. Obtain consent before collecting photos from farmers, and do not store farmer names.
- **Dataset origin:** The photos come from Uganda. Performance on Zimbabwean fields can differ, which is why step 6 exists.

## Citation

Babirye, C., Nakatumba-Nabende, J., Namanya, G., Mutebi, C., Ebellu, M., Murungi, J., Tobius, S., Ssemwogerere, J.,
Nakayima, A., Nabagereka, D., Asasira, J., & Kanyesigye, R. (2022). Makerere University Maize Image Dataset [Data set].
Harvard Dataverse. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LPGHKK
