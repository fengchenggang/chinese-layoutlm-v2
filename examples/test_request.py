import requests
import base64

# 将图片数据转成base64格式
with open(r'/work/Codes/layoutlmft/examples/XFUND-DATA-Gartner/zh.val/zh_val_0.jpg', 'rb') as f:
    img = base64.b64encode(f.read()).decode()
image = []
image.append(img)
res = {"image": image}
# 访问服务
r = requests.post("http://localhost:10003", data=res)
print(r.content)
