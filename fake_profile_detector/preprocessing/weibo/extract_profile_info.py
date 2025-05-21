def extract_profile_info(soup):
    profile_info = {}

    try:
        counter_div = soup.find("div", class_="PCD_counter")
        if counter_div:
            counts = counter_div.find_all("strong", class_="W_f18")
            if len(counts) >= 3:
                profile_info["following_count"] = counts[0].get_text(strip=True)
                profile_info["followers_count"] = counts[1].get_text(strip=True)
                profile_info["posts_count"] = counts[2].get_text(strip=True)
    except:
        pass

    try:
        description = soup.find("div", class_="pf_intro")
        if description:
            profile_info["profile_description"] = description.get_text(strip=True)
    except:
        pass

    try:
        info_items = soup.find_all("li", class_="item S_line2 clearfix")
        for item in info_items:
            icon = item.find("em", class_="W_ficon")
            if icon and icon.get_text(strip=True):
                icon_type = icon.get_text(strip=True)
                text = item.find("span", class_="item_text W_fl").get_text(strip=True)

                if "地" in icon_type or "place" in str(icon.get("class", [])):
                    profile_info["location"] = text
                elif "学" in icon_type or "edu" in str(icon.get("class", [])):
                    profile_info["education"] = text
                elif "职" in icon_type or "bag" in str(icon.get("class", [])):
                    profile_info["occupation"] = text
                elif "邮" in icon_type or "email" in str(icon.get("class", [])):
                    profile_info["email"] = text
    except:
        pass

    return profile_info
