import asyncio
import websockets
import json
import pandas as pd
import os
from tqdm import tqdm

# 替换为你的WebSocket服务器URI
uri = "ws://172.26.166.90:8765"


question_num = 0

def load_json(json_path):
    with open(json_path, 'r') as f:
        res = json.load(f)
    return res

def dump_json(dic, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(dic, json_file)
    return

def construct_request(subject, test_row, id):
    sys_pattern = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。\n\n"
    test_pattern = "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案："

    res = sys_pattern.format(subject)
    res = res + test_pattern.format(test_row.question, test_row.A, test_row.B, test_row.C, test_row.D)
    request = {
        "id": int(id),
        "question": res,
        "is_decode": True,
        "forward_times": 3,
        "is_predict_option": True
    }
    return request

## 发送消息
async def send_messages(websocket, question_num, test_df, subject):
    # 读取CSV文件到DataFrame
    for i in range(question_num):
        request = construct_request(subject, test_df.loc[i], i)
        await websocket.send(json.dumps(request, ensure_ascii=False))
        #await asyncio.sleep(0.3) # 等待一段时间再发送下一条消息

## 处理消息
async def receive_messages(websocket, question_num, subject, subject_dict, res):
    for i in tqdm(range(question_num)):
        response = await websocket.recv()
        response = json.loads(response)
        # 对response 进行下一步处理
        pred =  response['answer'][-1]
        subject_dict[str(i)] = pred
    res[subject] = subject_dict
    await websocket.send("tps")
    tps = await websocket.recv()
    print(f"tps: {tps}")

async def main(uri):
    # define params
    example_num = 0
    dev_path = "ceval-exam/dev"
    test_path = "ceval-exam/test"
    submit_path ="submisstion.json"
    subject_path = "subject_mapping.json"
    subject_map = load_json(subject_path)
    
    res = {}
    subject_num = len(os.listdir(test_path))
    print(f"Subject numbers: {subject_num}")
    sub_cnt = 0
    
    for dev_csv_file, test_csv_file in zip(os.listdir(dev_path), os.listdir(test_path)):
        dev_csv_path = os.path.join(dev_path, dev_csv_file)
        test_csv_path = os.path.join(test_path, test_csv_file)
        dev_df = pd.read_csv(dev_csv_path)
        test_df = pd.read_csv(test_csv_path)

        subject = test_csv_file.replace("_test.csv", "")
        subject_zh = subject_map[subject][1]
        dev_row = [dev_df.loc[i] for i in range(example_num)]

        subject_dict = {}
        sub_cnt += 1
        print("======================================================")
        print("======================================================")
        print("Current subject:", subject, ", subject no.", sub_cnt)
        print("======================================================")
        print("======================================================")
        question_num = len(test_df)
        async with websockets.connect(uri) as websocket:
            # 创建发送和接收消息的协程
            sender = asyncio.create_task(send_messages(websocket, question_num, test_df, subject))
            receiver = asyncio.create_task(receive_messages(websocket, question_num, subject, subject_dict, res))
            # 等待两个协程完成
            await asyncio.gather(sender, receiver)
        print("cur_res: ", res)
        
    dump_json(res, submit_path)



# 运行客户端
asyncio.run(main(uri))