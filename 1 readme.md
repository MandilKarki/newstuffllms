# Email Classification Service

This service uses a transformer model to generate text based on email bodies and a static prompt. The text after the last "classification label" in the generated text is returned as the classification of the email.

## Project Structure

- `model.py`: Contains the TextGenerator class which is responsible for loading the transformer model and generating text based on a prompt.
- `prompt.py`: Contains the PromptManager class which is used to append the static prompt to each incoming email.
- `api.py`: Contains the FastAPI application and the /predict API route. 
- `main.py`: This is the entry point of the application, it starts the FastAPI server.

## Instructions

1. Install the required dependencies with pip: `pip install fastapi uvicorn transformers`
2. Run the FastAPI server: `python main.py`
3. Send a POST request to http://localhost:8000/predict with a JSON body like this: `{"email": "Your email body here"}`

The service will respond with the generated classification text.
