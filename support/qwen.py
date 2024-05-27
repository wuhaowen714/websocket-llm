#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import argparse
import time
from tokenization_util import make_context
from transformers import AutoTokenizer
import numpy as np

#convert sail_dtype to numpy dtype
def type_convert(sail_dtype):
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    if sail_dtype == sail.Dtype.BM_BFLOAT16: # 后续需要修改bf16的接口,现在先用fp16的代替
        return np.float16
    
    raise TypeError("only support float32 and int32 right now")

def fp16_cast(arr:np.ndarray): #这个接口的作用在于把np.float16假冒成np.uint16传进Tensor，sail update_data如果能接收传输二进制，那就不需要这个了。(后续需要改成bf16的接口)
    """
    reinterpret an array with int16 instead of float16, because pybind11 do not support float16.
    """
    if arr.dtype == np.float16:
        return arr.view(np.uint16)
    else:
        return arr

class Qwen:
    def __init__(self, args):
        self.handles = [sail.Handle(i) for i in args.dev_id]
        print("Load " + args.token + " ...")
        self.sp = AutoTokenizer.from_pretrained(args.token, trust_remote_code=True)
        
        # warm up
        self.sp.decode([0]) 
        self.EOS = self.sp.im_end_id

        # load bmodel
        self.net = sail.EngineLLM(args.bmodel, args.dev_id)
        self.dev_num = len(args.dev_id)
        self.graph_names = self.net.get_graph_names()

        # initialize qwen parameters
        self.NUM_LAYERS = (len(self.graph_names) - 3) // 2
        # _, self.SEQLEN, self.HIDDEN_SIZE = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        _, self.SEQLEN, self.HIDDEN_SIZE = self.net.get_input_shape("block_0", 0)

        # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        # import pdb;pdb.set_trace()
        # initialize tensors (inputs & outputs)
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_sail_tensor(self.name_embed,
                                                        input_idx = [i for i in range(self.dev_num)],
                                                        )
        
        self.first_embed_output = self.init_sail_tensor(self.name_embed,
                                                        input_idx = [i for i in range(self.dev_num)],
                                                        input_type = False
                                                        )

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache,
                                                    input_idx = [i for i in range(self.dev_num)],
                                                    )
        
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache,
                                                        input_idx = [i for i in range(self.dev_num)],
                                                        input_type = False
                                                        )

        # forward_first: hidden_state
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0],
                                                        input_idx = [i for i in range(0, self.dev_num*3, 3)],
                                                        malloc=False)

        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 
                                                        input_idx = [i for i in range(0, self.dev_num*3, 3)],
                                                        input_type = False, 
                                                        malloc=False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0],
                                                        input_idx = [i for i in range(0, self.dev_num*5, 5)],
                                                        malloc=False)

        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 
                                                        input_idx = [i for i in range(0, self.dev_num*3, 3)],
                                                        input_type = False, 
                                                        malloc=False)

        # forward_first: position_id_tensor and attention_mask_tensor
        self.first_pid = self.init_sail_tensor(self.name_blocks[0],
                                                input_idx = [i for i in range(1, self.dev_num*3, 3)],
                                                )
        self.first_attention_mask = self.init_sail_tensor(self.name_blocks[0], 
                                                     input_idx = [i for i in range(2, self.dev_num*3, 3)],
                                                     ) 
        
        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0],
                                                input_idx = [i for i in range(1, self.dev_num*5, 5)],
                                                )
        self.next_attention_mask = self.init_sail_tensor(self.name_blocks_cache[0],
                                                    input_idx = [i for i in range(2, self.dev_num*5, 5)],
                                                    )

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0],
                                                 input_idx = [i for i in range(1, self.dev_num*3, 3)],
                                                 input_type = False)

        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0],
                                                 input_idx = [i for i in range(2, self.dev_num*3, 3)],
                                                 input_type = False)

        # forward_first: key_tensor and value_tensor
        self.past_key_output = []
        self.past_value_output = []

        # forward_next: kv cache block 
        self.cache_key_input = []
        self.cache_key_output = []
        self.cache_value_input = []
        self.cache_value_output = []
        # import pdb;pdb.set_trace()
        for _ in range(self.NUM_LAYERS):
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0],
                                                              input_idx = [i for i in range(1, self.dev_num*3, 3)],
                                                              input_type = False))

            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0],
                                                                input_idx = [i for i in range(2, self.dev_num*3, 3)], 
                                                                input_type = False))

            self.cache_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 
                                                              input_idx = [i for i in range(3, self.dev_num*5, 5)],
                                                              malloc=False))

            self.cache_key_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 
                                                              input_idx = [i for i in range(1, self.dev_num*3, 3)],
                                                              input_type = False,
                                                              malloc=False))

            self.cache_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 
                                                              input_idx = [i for i in range(4, self.dev_num*5, 5)],
                                                              malloc=False))

            self.cache_value_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 
                                                              input_idx = [i for i in range(2, self.dev_num*3, 3)],
                                                              input_type = False,
                                                              malloc=False))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm,
                                               input_idx = [0],
                                               repeat_num = 1
                                               )
                                               
        self.lm_output = self.init_sail_tensor(self.name_lm, 
                                               input_idx = [0],
                                               repeat_num = 1,
                                               input_type = False)
        # import pdb;pdb.set_trace()
        self.token_length = 0

    def init_sail_tensor(self, name, input_idx, shape=None, input_type=True, repeat_num=0, malloc=True):
        Tensors = []
        repeat_num = self.dev_num if repeat_num == 0 else 1
        for i in range(repeat_num):
            tensor = {}
            if input_type:            
                tensor["name"] = self.net.get_input_names(name)[input_idx[i]]
                # tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
                # tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
                tensor["shape"] = self.net.get_input_shape(name, input_idx[i]) if shape is None else shape
                tensor["dtype"] = self.net.get_input_dtype(name, input_idx[i])
                tensor["data"] = sail.Tensor(self.handles[i], tensor["shape"], tensor["dtype"], False, True) if malloc else None
            else:
                tensor["name"] = self.net.get_output_names(name)[input_idx[i]]
                # tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
                # tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
                tensor["shape"] = self.net.get_output_shape(name, input_idx[i]) if shape is None else shape
                tensor["dtype"] = self.net.get_output_dtype(name, input_idx[i])
                tensor["data"] = sail.Tensor(self.handles[i], tensor["shape"], tensor["dtype"], False, True) if malloc else None
            Tensors.append(tensor)

        return Tensors

    # inference for the first token
    def forward_first(self, token):
        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input[0]["dtype"]))
        input_ids[:min(self.SEQLEN, len(token))] = token
        input_ids = input_ids.reshape(1, -1)
        self.token_length = len(token)
        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid[0]["dtype"])) 
        for i in range(self.token_length):
            position_id[i] = i

        attention_mask = np.ones(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention_mask[0]["dtype"])) * (-10000.0)
        
        for i in range(self.token_length):
            for j in range(self.SEQLEN):
                if (j <= i):
                    attention_mask[i*self.SEQLEN + j] = 0
        
        # embedding
        input_embed_tensors = {}
        output_embed_tensors = {}
        for i in range(self.dev_num):
            self.first_embed_input[i]["data"].update_data(input_ids)
            input_embed_tensors[self.first_embed_input[i]["data"].device_id()] = self.first_embed_input[i]["data"]
            output_embed_tensors[self.first_embed_output[i]["data"].device_id()] = self.first_embed_output[i]["data"]
        import pdb;pdb.set_trace()
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)
        
        # blocks
        input_blocks_tensors = {}
        for i in range(self.dev_num):
            self.first_hidden_input[i]["data"] = self.first_embed_output[i]["data"]
            self.first_hidden_input[i]["data"].reshape(self.first_hidden_input[i]["shape"])
            self.first_pid[i]["data"].update_data(position_id.reshape(self.first_pid[i]["shape"]))
            self.first_attention_mask[i]["data"].update_data(attention_mask.reshape(self.first_attention_mask[i]["shape"]).astype(np.uint16))
            input_blocks_tensors[3*i] = self.first_hidden_input[i]["data"]
            input_blocks_tensors[3*i + 1] = self.first_pid[i]["data"]
            input_blocks_tensors[3*i + 2] = self.first_attention_mask[i]["data"]
         
        for i in range(self.NUM_LAYERS):
            output_blocks_tensors = {}
            for idx in range(self.dev_num):
                output_blocks_tensors[3*idx] = self.first_hidden_input[idx]["data"]
                output_blocks_tensors[3*idx + 1] = self.past_key_output[i][idx]["data"]
                output_blocks_tensors[3*idx + 2] = self.past_value_output[i][idx]["data"]
            import pdb;pdb.set_trace()
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
            import pdb;pdb.set_trace()   
        
        # get the last token info as Lm head input
        import pdb;pdb.set_trace()
        copy_len = self.first_hidden_input[0]["shape"][-1]
        self.lm_input[0]["data"].sync_d2d(self.first_hidden_input[0]["data"],
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {self.lm_input[0]["data"].device_id(): self.lm_input[0]["data"]}
        output_lm_tensors = {self.lm_output[0]["data"].device_id(): self.lm_output[0]["data"]}
        
        # Lm_head Inference
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        # import pdb;pdb.set_trace()
        return self.lm_output[0]["data"].asnumpy()[0][0][0]

    # The following tokens prediction
    def forward_next(self, ):
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention_mask[0]["dtype"]))
        for i in range(self.token_length-1, self.SEQLEN):
            attention_mask[i] = -10000.0
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid[0]["dtype"]))

        # embedding
        input_embed_tensors = {}
        output_embed_tensors = {}

        next_token = np.reshape(self.lm_output[0]["data"].asnumpy(), self.next_embed_input[0]["shape"])
        for i in range(self.dev_num):
            self.next_embed_input[i]["data"].update_data(next_token)
            input_embed_tensors[self.next_embed_input[i]["data"].device_id()] = self.next_embed_input[i]["data"]
            output_embed_tensors[self.next_embed_input[i]["data"].device_id()] = self.next_embed_output[i]["data"]
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)
        
        # block
        input_blocks_tensors = {}
        for i in range(self.dev_num):
            self.next_hidden_input[i]["data"] = self.next_embed_output[i]["data"]
            self.next_hidden_input[i]["data"].reshape(self.next_hidden_input[i]["shape"])

            self.next_pid[i]["data"].update_data(position_id.reshape(self.next_pid[i]["shape"]))
            self.next_attention_mask[i]["data"].update_data(attention_mask.reshape(self.next_attention_mask[i]["shape"]).astype(np.uint16))
            input_blocks_tensors[5*i] = self.next_hidden_input[i]["data"]
            input_blocks_tensors[5*i + 1] = self.next_pid[i]["data"]
            input_blocks_tensors[5*i + 2] = self.next_attention_mask[i]["data"]

        for i in range(self.NUM_LAYERS):
            output_blocks_tensors = {}
            for idx in range(self.dev_num):
                input_blocks_tensors[5*idx + 3] = self.past_key_output[i][idx]["data"]
                input_blocks_tensors[5*idx + 4] = self.past_value_output[i][idx]["data"]

                output_blocks_tensors[3*idx] = self.next_hidden_input[idx]["data"]
                output_blocks_tensors[3*idx + 1] = self.present_key[idx]["data"]
                output_blocks_tensors[3*idx + 2] = self.present_value[idx]["data"]
            # import pdb;pdb.set_trace()
            self.net.process(self.name_blocks_cache[i], input_blocks_tensors, output_blocks_tensors)  

            # update kv_cache()
            unit_size = self.present_key[0]["shape"][-1]*self.present_key[0]["shape"][-2]
            # self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.token_length-1)*unit_size, unit_size)
            # self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.token_length-1)*unit_size, unit_size)
            for idx in range(self.dev_num):
                self.past_key_output[i][idx]["data"].sync_d2d(self.present_key[idx]["data"],
                                                            0, 
                                                            (self.token_length-1)*unit_size,
                                                            unit_size)

                self.past_value_output[i][idx]["data"].sync_d2d(self.present_value[idx]["data"], 
                                                            0,
                                                            (self.token_length-1)*unit_size, 
                                                            unit_size)

        input_lm_tensors = {self.lm_input[0]["data"].device_id(): self.lm_input[0]["data"]}
        output_lm_tensors = {self.lm_output[0]["data"].device_id(): self.lm_output[0]["data"]}

        # Lm_head Inference
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return self.lm_output[0]["data"].asnumpy()[0][0][0]

    def chat_stream(self, input, history, system='You are a helpful assistant.'):
        input_tokens = make_context(self.sp, query=input, max_window_size=self.SEQLEN)
        if (len(input_tokens) > self.SEQLEN / 3):
            yield '##INPUT_TOO_LONG'
            return

        tok_num = 0
        tokens = make_context(self.sp, query=input, 
                        history=history,
                        system=system,
                        max_window_size=self.SEQLEN,
                        chat_format="chatml")
        while (len(tokens) > self.SEQLEN / 2):
            if (len(history) > 0):
                history = history[1:]
            else:
                system = ''
            tokens = make_context(self.sp, query=input, 
                            history=history,
                            system=system,
                            max_window_size=self.SEQLEN,
                            chat_format="chatml")
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()
        while token != self.EOS and self.token_length < self.SEQLEN:
            diff = self.sp.decode([token])
            yield diff
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            tok_num += 1
            token = self.forward_next()
        
        if self.token_length >= self.SEQLEN:
            yield '##TOKEN_LENGTH_MAX'
            return
        
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration
        print('\n\n')
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def app(client):
    history = []
    while True:
        input_str = input("\nQuestion: ")
        if input_str == "exit":
            break
        print("\nAnswer: ")
        assistant_msg = ''
        for response in client.chat_stream(input_str, history):
            assistant_msg = response
            print(response, flush=True, end='')
        history.append([input_str, assistant_msg])

def main(args):
    qwen = Qwen(args)
    app(qwen)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='../qwen-72b_int4_seq1024_8dev.bmodel', help='path of bmodel')
    parser.add_argument('--token', type=str, default='./token_config/', help='path of tokenizer')
    parser.add_argument('--dev_id', type=list, default=[0,1,2,3,4,5,6,7], help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')
