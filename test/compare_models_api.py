import os
import json
import random
from together import Together

DATASET_PATH = "../data/modelmatch_dataset.json"
MODEL_1_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_2_NAME = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"

def load_dataset(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset.get("conversations", [])
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []

def get_together_response(client: Together, model_name: str, messages: list):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Together API for {model_name}: {e}")
        return None

def main():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable not set.")
        return

    client = Together(api_key=api_key)
    conversations = load_dataset(DATASET_PATH)

    if not conversations:
        print("No conversations found in the dataset.")
        return

    selected_conversation = random.choice(conversations)
    
    print(f"Selected Conversation ID: {selected_conversation.get('id', 'N/A')}")
    print(f"Target Model in Dataset: {selected_conversation.get('target_model_name', 'N/A')}:{selected_conversation.get('target_model_version', 'N/A')}")
    print(f"Quantization: {selected_conversation.get('quantization_type', 'N/A')}")
    print("-" * 50)

    user_prompts = selected_conversation.get("user_prompts", [])
    if not user_prompts:
        print("Selected conversation has no user prompts.")
        return

    history_model1 = []
    history_model2 = []

    for i, user_prompt_text in enumerate(user_prompts):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {user_prompt_text}")

        current_messages_model1 = history_model1 + [{"role": "user", "content": user_prompt_text}]
        response_model1 = get_together_response(client, MODEL_1_NAME, current_messages_model1)
        if response_model1:
            print(f"\n{MODEL_1_NAME}:")
            print(response_model1)
            history_model1.append({"role": "user", "content": user_prompt_text})
            history_model1.append({"role": "assistant", "content": response_model1})
        else:
            print(f"\n{MODEL_1_NAME}: No response or error.")

        current_messages_model2 = history_model2 + [{"role": "user", "content": user_prompt_text}]
        response_model2 = get_together_response(client, MODEL_2_NAME, current_messages_model2)
        if response_model2:
            print(f"\n{MODEL_2_NAME}:")
            print(response_model2)
            history_model2.append({"role": "user", "content": user_prompt_text})
            history_model2.append({"role": "assistant", "content": response_model2})
        else:
            print(f"\n{MODEL_2_NAME}: No response or error.")

        print("-" * 50)

if __name__ == "__main__":
    main()