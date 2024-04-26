import asyncio
import websockets
import json
import sophon.sail as sail
from chatglm3 import ChatGLM3
from transformers import AutoTokenizer
from logger import init_logger

dev_ids = [0, 1, 2]
bmodel_path = '../../models/BM1684X/chatglm3-6b_int4.bmodel'
tokenizer_path = '../token_config'

port = 8765
logger = init_logger('chatglm3')
client_pool = []

token_len_total = 0
next_duration_total = 1e-8 

def init_client_pool():
    for dev_id in dev_ids:
        engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.DEVIO)
        handle = sail.Handle(dev_id)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        client = ChatGLM3(handle, engine, tokenizer)
        client_pool.append({"client": client, "status": 0})


async def consumer_handler(queue):
    while True:
        for i in range(len(client_pool)):
            if (client_pool[i]["status"] == 0):
                message, websocket = await queue.get()
                client_pool[i]["status"] = 1
                asyncio.create_task(process_message(client_pool[i], message, websocket))  # 处理消息

        await asyncio.sleep(1)

async def process_message(client_item, message, websocket):
    try:
        params = json.loads(message)
        is_decode = params.get('is_decode', True)
        forward_times = params.get('forward_times', 0)
        loop = asyncio.get_running_loop()
        answer, ftl, next_token_len, next_duration = await loop.run_in_executor(
            None,  # None 表示使用默认的线程池执行器
            client_item["client"].chat, 
            params["question"],  
            [],
            is_decode,
            forward_times
        )
        response = {
            "id": params["id"],
            "answer": answer,
            "ftl": ftl
        }
        global token_len_total, next_duration_total
        token_len_total += next_token_len
        next_duration_total += next_duration

        logger.info(f"chat done: {json.dumps(response, ensure_ascii=False)}")
        logger.info(f"token_len_total: {token_len_total}, next_duration_total: {next_duration_total}")
        # 处理完成，发送响应给客户端
        await websocket.send(json.dumps(response, ensure_ascii=False))
    except Exception as e:
        logger.error(e)
    finally:
        client_item["status"] = 0



async def handler(websocket, path, queue):
    async for message in websocket:
        logger.info(f"get message: {message}")
        if (message == "tps"):
            await websocket.send(str((token_len_total / next_duration_total) * len(dev_ids)))
        else:
            # 将消息和websocket连接放入队列
            asyncio.create_task(queue.put((message, websocket)))

async def main():
    init_client_pool()
    queue = asyncio.Queue()  
    # 启动消费者协程，这个协程将在服务器生命周期内一直运行
    consumer_task = asyncio.create_task(consumer_handler(queue))

    # 启动WebSocket服务器
    async with websockets.serve(lambda ws, path: handler(ws, path, queue), "0.0.0.0", port):
        logger.info(f"server start, port: {port}")
        await asyncio.Future()  # 保持服务器运行

asyncio.run(main())  # 运行主函数