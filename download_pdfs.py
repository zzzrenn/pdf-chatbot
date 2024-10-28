import os

import requests
from bs4 import BeautifulSoup

base_url = "https://www.nice.org.uk"


save_dir = "documents"
os.makedirs(save_dir, exist_ok=True)
visited = set()

QUERY = "Hypertension"
DOWNLOAD_LIMIT = 5

count = 0
page = 1
while True:
    url = requests.get(base_url + f"/search?q={QUERY}&ndt=Guidance&gst=Published&pa={page}")
    print("Scrapping page ", page, "...")
    soup = BeautifulSoup(url.content, "lxml")
    start_count = count
    for a in soup.find_all("a", href=True):
        mystr = a["href"]
        # print(mystr)
        if mystr.startswith("/guidance"):
            tag = mystr.split("/")[2].lower()
            if tag in visited:
                continue

            visited.add(tag)
            url_child = requests.get(base_url + mystr)
            soup_child = BeautifulSoup(url_child.content, "lxml")
            found_pdf = False
            for a_child in soup_child.find_all("a", href=True):
                mystr_child = a_child["href"]
                if "pdf" in mystr_child:
                    if mystr_child.startswith("https"):
                        url_pdf = requests.get(mystr_child)
                    else:
                        url_pdf = requests.get(base_url + mystr_child)
                    pdf_file_name = os.path.basename(mystr_child)
                    with open(save_dir + os.sep + pdf_file_name + ".pdf", "wb") as f:
                        f.write(url_pdf.content)
                    print(f"Downloaded {pdf_file_name}")
                    count += 1
                    found_pdf = True
                    break
            if not found_pdf:
                print("no pdf was found in this tag: ", tag)
        if count == DOWNLOAD_LIMIT:
            break
    page += 1
    if count - start_count != 15:
        print(f"Missing {15- count - start_count} pdfs!")
    if count == start_count or count == DOWNLOAD_LIMIT:
        break
print("Total downloaded pdf count: ", count)
