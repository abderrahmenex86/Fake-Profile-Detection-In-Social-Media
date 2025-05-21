import glob
import os

import pandas as pd
from tqdm import tqdm

from ...preprocessing.weibo.process_html_file import process_html_file


def process_all_files(directory):
    all_data = []

    html_files = glob.glob(os.path.join(directory, "*.html"))
    for file_path in tqdm(html_files, desc="Processing HTML files"):
        file_data = process_html_file(file_path)
        if file_data:
            all_data.append(file_data)

    return pd.DataFrame(all_data)
