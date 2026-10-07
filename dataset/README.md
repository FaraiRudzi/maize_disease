From the project root, download the archives using:

    python download_dataset.py --download --classes all

Extract each RAR archive into its corresponding class folder under dataset/raw, then run:

    python prepare_dataset.py --src dataset/raw --dst dataset_split
