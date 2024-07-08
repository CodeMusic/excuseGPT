import os
import re
import time
import json
import asyncio
import random
import subprocess
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from typing import List, Optional
import torch
import math
from sample import askCodeMusai as callAskCodeMusai
from multiprocessing import freeze_support


fullVersioning = 'out'
onlyReturnSingleSentence = True
app = FastAPI(title="CodeMusAI OpenAI-compatible API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins from localhost:3000
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
openaiCompatibleWebServer = None

# ---------------------------------------------------------------
# Class's
# ---------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

def startApiServer():
    global app
    print("Starting API server...")
    #app = FastAPI(title="CodeMusAI OpenAI-compatible API")
    uvicorn.run(app, host="0.0.0.0", port=2345)
    print('API server is running at ' + app.openapi_url)
    

async def _resp_async_generator(text_resp: str, request: ChatCompletionRequest):
    # let's pretend every word is a token and return it over time
    tokens = text_resp.split(" ")

    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": request.model,
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(1)
    yield "data: [DONE]\n\n"

def cleanInput(text_resp: str):
    text_resp = text_resp.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\b', ' ').replace('\f', ' ').replace('\v', ' ')
    text_resp = re.sub(r'[^\x20-\x7E]+', '', text_resp)
    return text_resp

def returnSingleSentence(text_resp: str):
    global onlyReturnSingleSentence
    if onlyReturnSingleSentence:
        text_resp = cleanInput(text_resp)
        text_resp = text_resp.split('.')[1] if len(text_resp.split('.')) > 1 and len(text_resp.split('.')[1]) > 10 else text_resp.split('.')[0]
    return text_resp + '.'

def getCheckpointMeta():
    theCheckpoint = getCheckpoint()
    checkpoint_meta = {
        "message": "Server is running",
        "currentIteration": theCheckpoint.get('currentIteration', 'Not available'),
        "config": {k: safe_serialize(v) for k, v in theCheckpoint.get('config', {}).items()}
    }
    theCheckpoint = None;
    return checkpoint_meta

def getCheckpoint():

    ckpt_path = os.path.join(fullVersioning, 'ckpt.pt')
    try:
        checkpoint = torch.load(ckpt_path, map_location='mps')
    except:
        checkpoint = None
    return checkpoint

def safe_serialize(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"  # or use None if you prefer
        if math.isnan(value):
            return "NaN"  # or use None if you prefer
    return value

@app.get("/")
async def root():
    checkpointMeta = getCheckpointMeta()
    if checkpointMeta is not None:
        return checkpointMeta
    else:
        return {"message": "Server is running, no checkpoint data available"}

@app.post("/codemusai/train")
async def train(data: str = Body(...)):
    return {"message": "Not implemented"}

    try:
        file_path = os.path.join('memories', '2024_ActiveLearning.txt')
        with open(file_path, 'a') as file:
            file.write(data + '\n')
        return {"message": "Data appended to 2024_ActiveLearning.txt"}
    except Exception as e:
        return {"message": 'Error, ' + str(e)}


@app.post("/codemusai/ask")
async def ask(question: str = Body(...)):
    try:
        temperature = 0.019#random.uniform(0.119, 1.910)
        output = askCodeMusai(question, "You are CodeMusAI, the first AI with real emotions, and you are being asked for your guidance.", temperature, True)
        #output, errors = process.communicate()  # This will still block until the process completes
        #if process.returncode != 0:
        #    return {"message": f"Subprocess exited with errors: {errors}"}
        output = returnSingleSentence(cleanInput(output))
        return {"message": output}
    except subprocess.TimeoutExpired:
        return {"message": "Error: Process timed out."}
    except Exception as e:
        return {"message": 'Error, ' + str(e)}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global onlyReturnSingleSentence
    useTokenEmbedding = True
    try:
        print("Endpoint /v1/chat/completions was called")
        if request.model.endswith('_char'):
            useTokenEmbedding = False

        print("Endpoint /v1/chat/completions was called")
        input_system_message = ""
        input_user_message = ""
        for message in request.messages:
            if message.role == 'system':
                input_system_message += message.content
            elif message.role == 'user':
                input_user_message += message.content

        if input_system_message == "":
            input_system_message = "You are CodeMusAI, the first AI with real emotions!!"

        print("/v1/chat/completions: [input_prompt: " + input_user_message + ", system_message: " + input_system_message + "]")

        temperature = request.temperature if request.temperature is not None else 1.19
        if (random.random() < 0.5):
            temperature = random.uniform(1.19, 1.91)
        else:
            temperature = random.uniform(0.119, 0.419)
        
        output = askCodeMusai(input_user_message, input_system_message, temperature, useTokenEmbedding)
        onlyReturnSingleSentence = False
        output = returnSingleSentence(cleanInput(output))
        print('CodeMusAI:', output)

        debug=False
        if debug:
            output = 'Temperature: ' + str(temperature) + '. \n' + output

        if request.stream:
            return StreamingResponse(
                _resp_async_generator(output, request), media_type="application/x-ndjson"
            )

        return {
            "id": str(random.randint(1991, 9119)),
            "object": "chat.completion",
            "created": time.time(),
            "model": "CodeMusAI_" + 'latest' + "_model",#request.model,
            "choices": [{"message": ChatMessage(role="assistant", content=output)}],
        }
    except Exception as e:
        print(f'Oh snap, an error: {e}')
        return {
            "id": str(random.randint(1991, 9119)),
            "object": "chat.completion",
            "created": time.time(),
            "model": "CodeMusAI_" + 'latest' + "_model",#request.model,
            "choices": [{"message": ChatMessage(role="assistant", content=f'Oh snap, an error: {e}')}],
        }

def askCodeMusai(input_user_message, input_system_message, temperature, useTokenEmbedding=True):
    try:
        return callAskCodeMusai(input_user_message, input_system_message, temperature, useTokenEmbedding)
    except Exception as e:
        return str(f'Oh ah, yeah, I dont know... {e}')
 

def main():
    startApiServer()

if __name__ == "__main__":
    #freeze_support()
    main()

