import os
from fastapi import FastAPI
from controller import metadata_controller
from services.answering.answering_service import AnsweringService

app = FastAPI(
    title="Palantir Ai Api",
    description="API for retrieving data from vector database",
    version="1.0.0",
)

app.include_router(metadata_controller.router)
def run_chatbot():
    chatbot_service = AnsweringService()
    print("ChatBot: Hello! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break
        response = chatbot_service.get_fireworks_response(user_input)
        print(f"ChatBot: {response}")


if __name__ == "__main__":

    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

    if os.getenv('CHAT_BOT_OPERATIONAL').lower() != 'down':
        run_chatbot()