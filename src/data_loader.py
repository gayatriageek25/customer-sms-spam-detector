import os
import zipfile
import urllib.request
import pandas as pd

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
ZIP_FILE = os.path.join(DATA_DIR, "smsspamcollection.zip")

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_sms_dataset():
    """Download and load the UCI SMS Spam Collection dataset."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("Downloading SMS Spam Collection dataset...")

        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)

        print("Extracting dataset...")

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    return df
