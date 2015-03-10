import random
import time

__author__ = 'niklas'

import urllib.request
import shutil
import os

dict_urls = []

for i in range(1, 16):
    url = "http://www.manythings.org/vocabulary/lists/l/words.php?f=noll{0:0=2d}".format(i)
    dict_urls.append(url)

for i, url in enumerate(dict_urls):
    print("fetching file number", i, "at URL: ", url)
    try:
        randint = random.randint(1, 3)
        print("sleeping for ", randint, " seconds")
        time.sleep(randint)
        file_name, headers = urllib.request.urlretrieve(url)
        shutil.copy(file_name, os.path.join("data", "dictionaries", "dict{}.html".format(i)))
    except urllib.error.ContentTooShortError:
        print("failed, content too short")