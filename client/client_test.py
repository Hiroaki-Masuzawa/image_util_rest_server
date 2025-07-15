import requests

url = "http://localhost:8008/maskrcnn/predict"
image_path = "input.jpg"

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

response_dict = response.json()

for i in range(len(response_dict['predictions'])):
    for k in ['class', 'confidence', 'bbox', 'x', 'y', 'width', 'height']:
        print(response_dict['predictions'][i][k])