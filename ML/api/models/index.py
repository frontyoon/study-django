from io import BytesIO
from djanog_fn import _output
#from test import url_to_image

img_pth = 'C:\\Project\\study-django\\ML\\api\\models\\img\\jung.jpg'

url = "s3://consolationbucket//b7db6140-7978-465b-be05-f1f2fe058db4.png"
url1 = "https://consolationbucket.s3.ap-northeast-2.amazonaws.com/b7db6140-7978-465b-be05-f1f2fe058db4.png"

#img = url_to_image(url)

model_pth = 'C:\\Project\\study-django\\ML\\api\\models\\best_model.pth'

print(_output(url1, model_pth))

