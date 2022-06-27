from PIL import Image
import urllib.request
import time
from io import BytesIO


def url_to_image(url):
    url = url

    # time check
    start = time.time()

    # request.urlopen()
    # HTTP Error 403: Forbidden 에러 때문에 하단의 소스 한 줄을 추가해주었다.
    req = urllib.request.Request(url, headers = {"User-Agent" : "Mozilla/5.0"})
    res = urllib.request.urlopen(req).read()

    # 이미지 다운로드 시간 체크
    print(time.time() - start)


    # Image open
    urlopen_img = Image.open(BytesIO(res))

    return urlopen_img