import re


def extract_config_data(html_content):
    config_pattern = re.compile(r"\$CONFIG\s*=\s*\{\s*(.*?)\}\s*;", re.DOTALL)
    match = config_pattern.search(html_content)

    config_data = {}
    if match:
        config_text = match.group(1)

        lines = config_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and ":" in line:
                # Handle the line to extract key and value
                if line.endswith(";"):
                    line = line[:-1]

                key_val = line.split(":", 1)
                if len(key_val) == 2:
                    key, value = key_val

                    # Clean up the key
                    key = (
                        key.strip()
                        .replace("$CONFIG[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace('"', "")
                    )

                    # Clean up the value
                    value = value.strip().strip("'").strip(";").strip("'").strip('"')

                    config_data[key] = value

    return config_data
