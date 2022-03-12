from flask import request, Flask
import base64
import cv2
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def get_frame():
    # 解析图片数据
    img = base64.b64decode(str(request.form['image']))
    image_data = np.fromstring(img, np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    cv2.imwrite(r'/work/Codes/layoutlmft/examples/XFUND-DATA-Gartner/0.jpg', image_data)
    print(image_data)
    return 'koukou'


if __name__ == "__main__":
    app.run("localhost", port=10004)
