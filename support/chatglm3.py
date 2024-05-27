    #===----------------------------------------------------------------------===#
    #
    # Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
    #
    # SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
    # third-party components.
    #
    #===----------------------------------------------------------------------===#
import sophon.sail as sail
import time
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import argparse

#convert sail_dtype to numpy dtype
def type_convert(sail_dtype):
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    
    raise TypeError("only support float32 and int32 right now")

def fp16_cast(arr:np.ndarray): #这个接口的作用在于把np.float16假冒成np.uint16传进Tensor，sail update_data如果能接收传输二进制，那就不需要这个了。
    """
    reinterpret an array with int16 instead of float16, because pybind11 do not support float16.
    """
    if arr.dtype == np.float16:
        return arr.view(np.uint16)
    else:
        return arr
    
class ChatGLM3_BS:
    def __init__(self, handle, args):
        # load tokenizer
        self.sp = AutoTokenizer.from_pretrained(args.token_config, trust_remote_code=True)
        self.handle = handle
        # warm up
        self.sp.decode([0]) 
        self.EOS = self.sp.eos_token_id

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = sail.Engine(args.bmodel, 0, sail.IOMode.DEVIO)

        self.graph_names = self.net.get_graph_names()
        
        # initialize glm parameters
        self.NUM_LAYERS = (len(self.graph_names) - 5) // 2
        self.first_hidden_input_shape = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        self.SEQLEN, self.batch_size, self.HIDDEN_SIZE = self.first_hidden_input_shape
        
        self.is_greedy_sample = True
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_sample = "greedy_head" if self.is_greedy_sample else "penalty_sample_head"

        # tensors:
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0)
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, None, False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0)
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, None, False)

        # forward_first: hidden_state
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_first: position_id_tensor 和 attention_mask_tensor
        self.first_pid = self.init_sail_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_sail_tensor(self.name_blocks[0], 2)
    
        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_sail_tensor(self.name_blocks_cache[0], 2)

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False)

        # forward_first: key_tensor 和 value_tensor
        self.past_key_output = []
        self.past_value_output = []

        # forward_next: cache block的kv tensor名
        self.cache_key_input = []
        self.cache_key_output = []
        self.cache_value_input = []
        self.cache_value_output = []

        for i in range(self.NUM_LAYERS):
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0], 2, None, False))
            self.past_key_output[i]["data"].memory_set(0)
            self.past_value_output[i]["data"].memory_set(0)

            
            self.cache_key_input.append({"name": self.net.get_input_names(self.name_blocks_cache[0])[3]})
            self.cache_key_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False))

            self.cache_value_input.append({"name": self.net.get_input_names(self.name_blocks_cache[0])[4]})
            self.cache_value_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        # sample tensor
        self.sample_input = {"name": self.net.get_input_names(self.name_sample)[0]}
        self.sample_output = self.init_sail_tensor(self.name_sample, 0, None, False)


    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        if is_input:
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True) 
        return tensor
        
    def forward_first(self, tokens):
        # Keep
        input_ids = np.zeros([self.batch_size, self.SEQLEN], type_convert(self.first_embed_input["dtype"]))
        self.token_length = np.zeros(self.batch_size, np.int16)
        for i in range (len(tokens)):
            self.token_length[i] = len(tokens[i])
            input_ids[i, self.SEQLEN - len(tokens[i]):] = tokens[i]

        position_id = np.zeros([self.batch_size, self.SEQLEN], type_convert(self.first_pid["dtype"]))
        for i in range(self.batch_size):
            for j in range(self.token_length[i]):
                position_id[i][self.SEQLEN - self.token_length[i] + j] = j
        
        attention_mask = np.full([self.batch_size, self.SEQLEN, self.SEQLEN], -10000, type_convert(self.first_attention["dtype"])) 
        for k in range(self.batch_size):
            count = 1
            for i in range(self.SEQLEN - self.token_length[k], self.SEQLEN):
                attention_mask[k][i][self.SEQLEN - self.token_length[k]: self.SEQLEN - self.token_length[k] + count] = 0
                count += 1
        # embedding
        self.first_embed_input["data"].update_data(fp16_cast(input_ids))
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}
        
        
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        tmp = self.first_embed_output["data"].asnumpy()
        tmp = np.transpose(tmp, (1, 0, 2))
        self.first_hidden_tensor = sail.Tensor(self.handle, self.first_hidden_input["shape"], self.first_hidden_input["dtype"], False, True)
        self.first_hidden_tensor.update_data(fp16_cast(tmp))

        # self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(fp16_cast(position_id.reshape(self.first_pid["shape"])))
        self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"])))
        
        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor, 
                                self.first_pid["name"]: self.first_pid["data"], 
                                self.first_attention["name"]: self.first_attention["data"]}

        for i in range(self.NUM_LAYERS):        
            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.past_key_output[i]["data"],
                                    self.past_value_output[i]["name"]: self.past_value_output[i]["data"]}
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
            

        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1] * self.batch_size
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                    (self.SEQLEN-1)* copy_len,  
                                    0, 
                                    copy_len)
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}

        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        # sample
        input_sample_tensor = {self.sample_input["name"]: self.lm_output["data"]}
        output_sample_tensor = {self.sample_output["name"]: self.sample_output["data"]}
        self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)
        return self.sample_output["data"].asnumpy()

    def forward_next(self, ):
        attention_mask = np.zeros([self.batch_size, self.SEQLEN + 1], type_convert(self.next_attention["dtype"]))
        for k in range(self.batch_size):
            for i in range(self.SEQLEN - self.token_length[k] + 1):
                attention_mask[k][i] = -10000.0
        position_id = np.zeros([self.batch_size, 1], type_convert(self.next_pid["dtype"]))

        for i in range (self.batch_size):
            position_id[i][0] = self.token_length[i] - 1

        # embedding
        self.next_embed_input["data"] = self.sample_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(fp16_cast(position_id.reshape(self.next_pid["shape"])))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.past_key_output[i]["data"],
                                        self.cache_value_output[i]["name"]: self.past_value_output[i]["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        # sample
        input_sample_tensor = {self.sample_input["name"]: self.lm_output["data"]}
        output_sample_tensor = {self.sample_output["name"]: self.sample_output["data"]}
        self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)
        return self.sample_output["data"].asnumpy() #int32
    
    def forward_next_without_topk(self):
        attention_mask = np.zeros([self.batch_size, self.SEQLEN + 1], type_convert(self.next_attention["dtype"]))
        for k in range(self.batch_size):
            for i in range(self.SEQLEN - self.token_length[k] + 1):
                attention_mask[k][i] = -10000.0
        position_id = np.zeros([self.batch_size, 1], type_convert(self.next_pid["dtype"]))

        for i in range (self.batch_size):
            position_id[i][0] = self.token_length[i] - 1

        # embedding
        self.next_embed_input["data"] = self.sample_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(fp16_cast(position_id.reshape(self.next_pid["shape"])))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.past_key_output[i]["data"],
                                        self.cache_value_output[i]["name"]: self.past_value_output[i]["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return self.lm_output["data"].asnumpy()        
    
    def predict_option(self, inputs, history=[]):
        input_tokens = []
        for question in inputs:
            input_tokens.append(self.sp.build_chat_input(question, history=history, role="user"))
        first_start = time.time()
        token = self.forward_first(input_tokens)
        first_end = time.time()
        token = self.forward_next()
        logits = self.forward_next_without_topk()
        next_end = time.time()
        option_prompt = {316: "A", 347: "B", 319: "C", 367: "D"}
        option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        answers = []
        for i in range(len(inputs)):
            score = []
            for key in option_prompt.keys():
                score.append(logits[i][key])
            max_score = max(score)
            answers.append(option_map[score.index(max_score)])
        return answers, first_end-first_start, 2 * len(inputs), next_end-first_end

    def chat(self, inputs, history=[], is_decode = True, forward_times=0):
        input_tokens = []
        for question in inputs:
            input_tokens.append(self.sp.build_chat_input(question, history=history, role="user"))
        first_start = time.time()
        pre_token = self.forward_first(input_tokens)
        first_end = time.time()
        token = pre_token
        tokens = token.tolist()
        tok_num = 1
        for i in range(len(inputs)):
            self.token_length[i] += 1
        
        status = [True for i in range(len(inputs))]
        end_cnt = 0
        while end_cnt < len(inputs) and forward_times != tok_num:
            token = self.forward_next()
            for i in range(len(inputs)):
                if (status[i] and (token[i] == self.EOS or self.token_length[i] >= self.SEQLEN)):
                    status[i] = False
                    end_cnt += 1
                if (status[i]):
                    tokens[i].append(token[i][0])
                    self.token_length[i] += 1
            tok_num += 1
        next_end = time.time()        
        first_duration = first_end-first_start
        answers = []
        for item in tokens:
            if is_decode:
                answers.append(self.sp.decode(item))
            else:
                answers.append(item)
        return answers, first_duration, (tok_num - 1) * len(inputs), next_end - first_end


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--name', type=str, default='chatglm3', help='name of model, default chatglm3')
    parser.add_argument('--bmodel', type=str, default='./models/chatglm3-6b_int8_4bs_1k.bmodel', help='path of bmodel')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size of the model, default 1')
    parser.add_argument('--token_config', type=str, default='./token_config/', help='path of tokenizer')
    parser.add_argument('--dev_id', type=list, default=[0], help='dev ids, default 0')
    parser.add_argument('--port', type=int, default=8765, help='port of the service, default 8765')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    handle = sail.Handle(0)
    client = ChatGLM3_BS(handle, args)
    inputs = []
    batch_size = 4
    df = pd.read_csv('accountant_test.csv', encoding='utf-8')
    for i in range(batch_size):
        raw = df.loc[i]
        question = f"以下是中国关于Accountant考试的单项选择题，请选出其中的正确答案。\n\n{raw['question']} \nA. {raw['A']}。\nB. {raw['B']}。 \nC. {raw['C']}。\nD. {raw['D']}\n答案："
        inputs.append(question)
    output = client.predict_option(inputs)
    print(output)



