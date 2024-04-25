import asyncio
import websockets
import json
import pandas as pd

# 替换为你的WebSocket服务器URI
uri = "ws://localhost:8765"


question_num = 0

def build_request(raw):
    question = f"{raw['question']} A.{raw['A']}。B.{raw['B']}。 C.{raw['C']}。D.{raw['D']}"
    request = {
        "id": int(raw['id']),
        "question": question
    }
    return request


## 发送消息
async def send_messages(websocket):
    # 读取CSV文件到DataFrame
    df = pd.read_csv('accountant_test.csv', encoding='utf-8')
    question_num = df.shape[0]
    for i in range(question_num):
        request = build_request(df.loc[i])
        await websocket.send(json.dumps(request, ensure_ascii=False))
        await asyncio.sleep(1)  # 等待一段时间再发送下一条消息

## 处理消息
async def receive_messages(websocket):
    cnt = 0
    while True:
        response = await websocket.recv()
        print(f"收到: {response}")
        response = json.loads(response)
        # 对response 进行下一步处理

        cnt += 1
        if (cnt == question_num):
            break

async def main(uri):
    async with websockets.connect(uri) as websocket:
        # 创建发送和接收消息的协程
        sender = asyncio.create_task(send_messages(websocket))
        receiver = asyncio.create_task(receive_messages(websocket))
        # 等待两个协程完成
        await asyncio.gather(sender, receiver)



# 运行客户端
asyncio.run(main(uri))