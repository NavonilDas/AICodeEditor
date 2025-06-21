from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

app = FastAPI()

def transform_message(messages):
    if not isinstance(messages, list):
        return []
    transformed = []
    for message in messages:
        if not "message" in message:
            continue
        if messages["message"] == "":
            continue
        transformed.append({
            'role': message.get('role', 'user').lower(), 
            'content': message["message"]
        })
    return transformed

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if "messages" in data:
                messages = transform_message(data["messages"])
                if len(messages) <= 0:
                    await websocket.send_text("Incorrect Input provided to model")
                question = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                question = tokenizer(question, return_tensors="pt").to(model.device)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
                _ = model.generate(
                    **question, 
                    max_new_tokens=1500, 
                    do_sample=False, 
                    top_k=50, 
                    top_p=0.95, 
                    num_return_sequences=1, 
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )

                for text in streamer:
                    await websocket.send_text(text)


    except WebSocketDisconnect:
        print("Client Disconnected")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)