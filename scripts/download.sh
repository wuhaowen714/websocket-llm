#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

# models
if [ ! -d "./models" ]; 
then
    # chatglm3
    mkdir -p ./models/chatglm3
    pushd ./models/chatglm3
    ## TODO: update model source here
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1684X/chatglm3-6b_int4.bmodel
    mkdir token_config
    pushd token_config
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/token_config.zip
    unzip token_config.zip
    rm token_config.zip
    popd
    popd
    echo "ChatGLM3 downloaded!"

    # qwen
    mkdir -p ./models/qwen
    pushd ./models/qwen
    ## TODO: update model source here
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/models/BM1684X/chatglm3-6b_int4.bmodel
    mkdir token_config
    pushd token_config
    python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/token_config.zip
    unzip token_config.zip
    rm token_config.zip
    popd
    popd
    echo "Qwen downloaded!"

else
    echo "Models folder exist! Remove it if you need to update."
fi