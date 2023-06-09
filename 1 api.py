from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import TextGenerator
from prompt import PromptManager

class TextGenerationInput(BaseModel):
    email: str

class TextGenerationOutput(BaseModel):
    classification_text: str

model_id = "gpt2"  # replace with your model
model = TextGenerator(model_id)

prompt_text = "Your static prompt text here"  # replace with your prompt
prompt_manager = PromptManager(prompt_text)

app = FastAPI()

@app.post("/predict", response_model=TextGenerationOutput)
async def predict(input_data: TextGenerationInput):
    try:
        email_with_prompt = prompt_manager.append_prompt(input_data.email)
        classification_text = model.generate(email_with_prompt)
        return TextGenerationOutput(classification_text=classification_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
