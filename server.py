import asyncio
import websockets
import json
import sophon.sail as sail
from chatglm3 import ChatGLM3_BS
from transformers import AutoTokenizer
from logger import init_logger

dev_ids = [0]
bmodel_path = '/disk/haowen/bmodels/chatglm3-6b_int8_4bs_1k.bmodel'
tokenizer_path = './token_config'
batch_size = 4

port = 8765
logger = init_logger('chatglm3')
client_pool = []


token_len_total = 0
next_duration_total = 1e-8
ftl_total = 0
batch_num = 1e-8

def init_client_pool():
    for dev_id in dev_ids:
        engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.DEVIO)
        handle = sail.Handle(dev_id)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        client = ChatGLM3_BS(handle, engine, tokenizer)
        client_pool.append({"client": client, "status": 0})

async def get_data_with_timeout(queue, timeout):
    try:
        data = await asyncio.wait_for(queue.get(), timeout=timeout)
        return data
    except asyncio.TimeoutError:
        # 如果没有获取到数据，则捕获超时异常
        return None


async def consumer_handler(queue):
    while True:
        for i in range(len(client_pool)):
            if (client_pool[i]["status"] == 0):
                questions = []
                ids = []
                message_option = {}
                websocket = None
                for _ in range(batch_size):
                    data = await get_data_with_timeout(queue, timeout=0.5)
                    if data != None:
                        message, websocket = data
                        params = json.loads(message)
                        questions.append(params["question"])
                        ids.append(params["id"])
                        message_option["is_decode"] = params.get('is_decode', True)
                        message_option["forward_times"] = params.get('forward_times', 0)
                        message_option["is_predict_option"] = params.get('is_predict_option', False)
                
                if (len(questions) == 0):
                    break
                client_pool[i]["status"] = 1
                asyncio.create_task(process_message(client_pool[i], questions, ids, message_option, websocket))  # 处理消息
        await asyncio.sleep(1)

async def process_message(client_item, questions, ids, message_option, websocket):
    try:
        is_decode = message_option["is_decode"]
        forward_times = message_option["forward_times"]
        is_predict_option = message_option["is_predict_option"]
        loop = asyncio.get_running_loop()
        global token_len_total, next_duration_total, ftl_total, batch_num
        if (is_predict_option):
            answer_options, ftl, next_token_len, next_duration = await loop.run_in_executor(
                None,
                client_item["client"].predict_option,
                questions,
                []
            )
            token_len_total += next_token_len
            next_duration_total += next_duration

            for i in range(len(answer_options)):
                response = {
                    "id": ids[i],
                    "answer": answer_options[i],
                    "ftl": ftl
                }
                batch_num += 1
                ftl_total += ftl
                logger.info(f"chat done: {json.dumps(response, ensure_ascii=False)}")
                await websocket.send(json.dumps(response, ensure_ascii=False))
                # logger.info(f"token_len_total: {token_len_total}, next_duration_total: {next_duration_total}")
        else:
            answers, ftl, next_token_len, next_duration = await loop.run_in_executor(
                None,  # None 表示使用默认的线程池执行器
                client_item["client"].chat, 
                questions,  
                [],
                is_decode,
                forward_times
            )
            token_len_total += next_token_len
            next_duration_total += next_duration
            for i in range(len(answers)):
                response = {
                    "id": ids[i],
                    "answer": answers[i],
                    "ftl": ftl
                }
                batch_num += 1
                ftl_total += ftl
                logger.info(f"chat done: {json.dumps(response, ensure_ascii=False)}")
                # logger.info(f"token_len_total: {token_len_total}, next_duration_total: {next_duration_total}")
                # # 处理完成，发送响应给客户端
                await websocket.send(json.dumps(response, ensure_ascii=False))
    except Exception as e:
        logger.error(e)
    finally:
        client_item["status"] = 0



async def handler(websocket, path, queue):
    async for message in websocket:
        logger.info(f"get message: {message}")
        if (message == "tps"):
            logger.info(f"tps: {(token_len_total / next_duration_total) * len(dev_ids)}")
            await websocket.send(str((token_len_total / next_duration_total) * len(dev_ids)))
        elif (message == "ftl"):
            logger.info(f"ftl: {ftl_total / batch_num}")
            await websocket.send(str(ftl_total / batch_num))
        else:
            # 将消息和websocket连接放入队列
            await asyncio.create_task(queue.put((message, websocket)))

async def main():
    init_client_pool()
    queue = asyncio.Queue()  
    # 启动消费者协程，这个协程将在服务器生命周期内一直运行
    consumer_task = asyncio.create_task(consumer_handler(queue))

    # 启动WebSocket服务器
    async with websockets.serve(lambda ws, path: handler(ws, path, queue), "0.0.0.0", port):
        logger.info(f"server start, port: {port}")
        await asyncio.Future()  # 保持服务器运行

asyncio.run(main(), debug=True)  # 运行主函数