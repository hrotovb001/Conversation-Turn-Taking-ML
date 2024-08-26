from input import get_utterance
from politeness import speaker_done_probability
from utils import clean_utterance
import openai
import threading

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Other constants
done_threshold = 0.2


def chat_with_gpt(messages):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content


def get_user_input(lock, input_ready, user_input):
    while True:
        utterance = get_utterance()
        with lock:
            user_input.append(utterance)
            input_ready.set()  # Signal that input is ready


if __name__ == "__main__":
    print("The conversation is ready")

    # Initialize the conversation history
    conversation_history = []
    user_input = []
    input_ready = threading.Event()
    lock = threading.Lock()

    # Start a new thread for getting user input
    input_thread = threading.Thread(target=get_user_input, args=(lock, input_ready, user_input))
    input_thread.start()

    while True:
        input_ready.wait()  # Wait for user input

        with lock:
            current_input = ' '.join(user_input)  # Combine all user inputs

        print(f"\rYou: {current_input}", end="", flush=True)

        prev_utterance = conversation_history[-1]["content"] if conversation_history else ""
        prev_utterance = clean_utterance(prev_utterance)
        cur_utterance = clean_utterance(current_input)

        done_probability = speaker_done_probability(prev_utterance, cur_utterance)
        print(f" ({round(float(done_probability) * 100)}% done prob)", end="", flush=True)

        if done_probability < done_threshold:
            input_ready.clear()  # Reset the event for the next input
            continue

        print()

        # Add the user's message to the conversation history
        conversation_history.append({"role": "user", "content": current_input})

        # Get the model's response
        response = chat_with_gpt(conversation_history)

        # Add the model's message to the conversation history
        conversation_history.append({"role": "assistant", "content": response})

        print(f"GPT: {response}")

        input_ready.clear()  # Reset the event for the next input
        user_input.clear()
