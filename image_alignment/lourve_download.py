'ttps://collections.louvre.fr/en/artwork/image/download/{}/{}.zip'

import requests
import time
import os
from PIL import Image
from concurrent.futures.thread import ThreadPoolExecutor
import random

lourve_base = 'https://collections.louvre.fr/en/artwork/image/download/{}/0'

urls = []
for i in range(100000, 200000):
    urls.append(lourve_base.format(i))

def download_image(url):
    # start = time.time()
    print(url)
    r = requests.get(url, stream=True)
    if r.status_code != 404:
        save_path = '~/lourve_images/{}'.format(url.split('/')[-2])
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
        # end = time.time()
        # print(end - start)

# for i in urls:
#     download_image(i)
#     print(i)




if __name__ == "__main__":
    resize_images(os.path.expanduser('/raw_data/bikini_jpg'))
    # start = time.time()
    # with ThreadPoolExecutor() as exector:
    #     exector.map(download_image, urls)
    # end = time.time()
    # print(end - start)
