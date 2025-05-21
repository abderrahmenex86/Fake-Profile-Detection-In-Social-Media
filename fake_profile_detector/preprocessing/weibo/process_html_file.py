import os

from bs4 import BeautifulSoup

from ...preprocessing.weibo.extract_config_data import extract_config_data
from ...preprocessing.weibo.extract_post_content import extract_post_content
from ...preprocessing.weibo.extract_profile_info import extract_profile_info


def process_html_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        config_data = extract_config_data(html_content)
        post_content = extract_post_content(soup)
        profile_info = extract_profile_info(soup)

        all_data = {
            "file_name": os.path.basename(file_path),
            "account_content": post_content,
            **config_data,
            **profile_info,
        }

        return all_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"file_name": os.path.basename(file_path), "error": str(e)}
