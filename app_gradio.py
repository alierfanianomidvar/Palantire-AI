import gradio as gr
from services.answering.answering_service import AnsweringService


# Define the chatbot logic
def chatbot_response(user_input):
    chatbot_service = AnsweringService()  # Initialize the chatbot service
    if user_input.lower() == "exit":
        return "Goodbye!"
    response = chatbot_service.get_fireworks_response(user_input)
    return response

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot_response,  # Function to process user input
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="text",
    title="Palantír - Your Tolkien Guide",
    description=""
                "Welcome to Palantír, your AI-powered guide to the world of J.R.R. Tolkien." +
                "Ask questions about Middle-earth, including The Lord of the Rings, The Hobbit, " +
                "The Silmarillion, and other related lore. " +
                "Type your questions below and get answers steeped in the legendarium! " +
                "Type 'exit' to end the conversation.",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
