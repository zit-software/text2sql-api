import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = T5ForConditionalGeneration.from_pretrained(
    "thangved/text2sql").to(device)  # type: ignore
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def predict(context, question):
    inputs = tokenizer(f"query for: {question}? ",
                       f"tables: {context}",
                       max_length=200,
                       padding="max_length",
                       truncation=True,
                       pad_to_max_length=True,
                       add_special_tokens=True)

    input_ids = torch.tensor(
        inputs["input_ids"], dtype=torch.long).to(device).unsqueeze(0)
    attention_mask = torch.tensor(
        inputs["attention_mask"], dtype=torch.long).to(device).unsqueeze(0)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=128)
    answer = tokenizer.decode(
        outputs.flatten(), skip_special_tokens=True)  # type: ignore
    return answer


class Text2SqlReq(BaseModel):
    context: str
    question: str


class Text2SqlRes(BaseModel):
    answer: str


class StatusRes(BaseModel):
    status: int


@app.post('/text2sql', summary='Text 2 SQL', tags=['Text 2 SQL'], response_model=Text2SqlRes)
async def text2sql(body: Text2SqlReq):
    answer = predict(body.context, body.question)

    return Text2SqlRes(answer=answer)


@app.get('/status', summary='Check server status', tags=['Status'], response_model=StatusRes)
async def status():
    return StatusRes(status=200)
