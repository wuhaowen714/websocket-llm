## 使用方式
安装依赖
```
pip3 install websockets
```

修改server.py中9-11行，填写正确的dev_ids, bmodel_path, tokenizer_path 
```python
dev_ids = [0, 1, 2]
bmodel_path = './chatglm3-6b_int4_1dev_1024_nosample.bmodel'
tokenizer_path = '../token_config'
```
启动websocket服务
```bash
python3 server.py
```

调用websocket服务，可参考client.py
```bash
python3 client.py
```

## 接口形式
传入参数为json形式的字符串
```bash
{
    "id": 0, 
    "question": "以下是中国关于Accountant考试的单项选择题，请选出其中的正确答案。\n\n下列关于资本结构理论的说法中，不正确的是____。 \nA. 代理理论、权衡理论、有企业所得税条件下的MM理论，都认为企业价值与资本结构有关。\nB. 按照优序融资理论的观点，考虑信息不对称和逆向选择的影响，管理者偏好首选留存收益筹资，然后是发行新股筹资，最后是债务筹资。 \nC. 权衡理论是对有企业所得税条件下的MM理论的扩展。\nD. 代理理论是对权衡理论的扩展\n答案：", 
    "is_decode": False,
    "forward_times": 3,
    "is_predict_option": False
}
```
```
id - 问题id（必填）
question - 问题内容（必填）
is_decode - 是否decode（选填，默认为True）
forward_times - forward次数 （选填，默认为0）0代表一直到推理结束
is_predict_option: 是否预测选项（默认为False）如果为True，is_decode, forward_times参数就失效了
```

返回参数形式为
```bash
{
    "id": 0, 
    "answer": "\n 选项B中的说法是不正确的。按照优序融资理论的观点，管理者偏好首选债务筹资，然后是发行新股筹资，最后是留存收益筹资。而不是首选留存收益筹资，然后是发行新股筹资，最后是债务筹资。", 
    "ftl": 1.1623857021331787
}
```
```
id - 问题id
answer - 回答内容
ftl - first token的推理时间
```

如果使用predict_option，传入参数如下：
```bash
{
    "id": 0, 
    "question": "以下是中国关于Accountant考试的单项选择题，请选出其中的正确答案。\n\n下列关于资本结构理论的说法中，不正确的是____。 \nA. 代理理论、权衡理论、有企业所得税条件下的MM理论，都认为企业价值与资本结构有关。\nB. 按照优序融资理论的观点，考虑信息不对称和逆向选择的影响，管理者偏好首选留存收益筹资，然后是发行新股筹资，最后是债务筹资。 \nC. 权衡理论是对有企业所得税条件下的MM理论的扩展。\nD. 代理理论是对权衡理论的扩展\n答案：", 
    "is_predict_option": True
}
```

返回参数为
```bash
{
    "id": 0, 
    "answer": "B", 
    "ftl": 1.1623857021331787
}
```

如需要获取平均tps，发送message为"tps"
```python
await websocket.send("tps")
tps = await websocket.recv()
```
