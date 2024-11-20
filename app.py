from services.answering.answering_service import AnsweringService
from util.pdfreader import PdfReader
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
    run_chatbot()
