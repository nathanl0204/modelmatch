import json
import random
import torch
from transformers import pipeline

DATASET_PATH = "../data/modelmatch_dataset.json"

def load_dataset(filepath: str) -> dict:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        print(f"Please ensure '{filepath}' is correct relative to the script location.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        exit(1)

def get_random_conversation(dataset: dict) -> dict:
    if not dataset.get("conversations"):
        print("Error: No conversations found in the dataset.")
        exit(1)
    return random.choice(dataset["conversations"])

def main():
    model1_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model2_name = "HuggingFaceH4/zephyr-7b-beta"

    generation_params = {
        "max_new_tokens": 200,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
    }

    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    conversation = get_random_conversation(dataset)

    print(f"\nSelected conversation ID: {conversation.get('id', 'N/A')}")
    print(f"Source dataset for this conversation: {conversation.get('source_dataset', 'N/A')}")
    print(f"Target model specified in dataset: {conversation.get('target_model_name', 'N/A')}:{conversation.get('target_model_version', 'N/A')}")
    print("\nUser prompts in this conversation:")
    for i, prompt_text in enumerate(conversation['user_prompts']):
        print(f"  {i+1}. {prompt_text[:150]}{'...' if len(prompt_text) > 150 else ''}")
    

    print(f"\nLoading model 1: {model1_name}...")
    try:
        generator1 = pipeline(
            "text-generation",
            model=model1_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model {model1_name}: {e}")
        print("Make sure you have 'accelerate' installed (`pip install accelerate`) for device_map='auto'.")
        print("Ensure the model name is correct, you have internet access, and accepted any model licenses on HuggingFace Hub.")
        exit(1)
    
    print(f"\nLoading model 2: {model2_name}...")
    try:
        generator2 = pipeline(
            "text-generation",
            model=model2_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model {model2_name}: {e}")
        exit(1)
    
    history1_messages = []
    history2_messages = []

    print("\n--- Starting conversation simulation ---")
    for user_prompt_text in conversation['user_prompts']:
        print(f"\n👤 User: {user_prompt_text}")

        messages_for_model1 = history1_messages + [{"role": "user", "content": user_prompt_text}]

        try:
            output1_chat = generator1(messages_for_model1, **generation_params)
            assistant1_response = output1_chat[0]['generated_text'][-1]['content']
        except Exception as e:
            print(f"Error during generation with {model1_name}: {e}")
            assistant1_response = "[Error generating response]"
        
        print(f"🤖 {model1_name}: {assistant1_response}")

        history1_messages.append({"role": "user", "content": user_prompt_text})
        history1_messages.append({"role": "assistant", "content": assistant1_response})

        messages_for_model2 = history2_messages + [{"role": "user", "content": user_prompt_text}]

        try:
            output2_chat = generator2(messages_for_model2, **generation_params)
            assistant2_response = output2_chat[0]['generated_text'][-1]['content']
        except Exception as e:
            print(f"Error during generation with {model2_name}: {e}")
            assistant2_response = "[Error generating response]"
        
        print(f"🤖 {model2_name}: {assistant2_response}")

        history2_messages.append({"role": "user", "content": user_prompt_text})
        history2_messages.append({"role": "assistant", "content": assistant2_response})
    
    print("\n--- End of conversation simulation ---")

if __name__ == "__main__":
    main()