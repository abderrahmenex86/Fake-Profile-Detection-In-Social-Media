def extract_post_content(soup):
    post_divs = soup.find_all("div", class_="WB_text W_f14")

    all_content = []
    for div in post_divs:
        text = div.get_text(strip=True)
        if text:
            all_content.append(text)

    return " ".join(all_content)
