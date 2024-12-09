from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import sys
from datasets import load_dataset
import os
from openai import OpenAI
import json


os.system('clear')

def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)

def evaluate_state(data_entry, model="gpt-4o-mini"):
    client = OpenAI() 
    msgs = [
        {
            "role": "system",
            "content": """
                    You are a helpful AI assistant that helps evaluate an agent at problem solving.
                    Given a question and a final goal state, evaluate the agent's Chain of Thought.
                    You should provide a score from 100 to -100 with 100 being an extremely logical
                    reasoning step in the path to the goal state and -100 being an extremely illogical reasoning step 
                    on the path to the goal state. 
                    be reasonable with your scores.
            """,
        },
        {
            "role": "user",
            "content": str(
                "QUESTION/START STATE: "
                + str(data_entry.get("question", ""))
                + "\nAGENT CHAIN OF THOUGHT: "
                + str(data_entry.get("steps", ""))
                + "\nCORRECT GOAL STATE: "
                + str(data_entry.get("final_answer", ""))
            )
        },
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=msgs,
        functions=[
            {
                "name": "eval_chain_of_thought",
                "description": "function to score each Thought in chain of thought",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chain_of_thought": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "thought_id": { "type": "string" },
                                    "thought_score": {
                                        "type": "integer", 
                                        "description": "the score from 100 to -100"
                                    }
                                },
                                "required": ["thought_id", "thought_score"]
                            },
                            "description": "A list of dictionaries with string keys and integer values."
                        }
                    }
                },
            }
        ],
    )

    data = json.loads(response.choices[0].message.function_call.arguments)
    
    thought_scores = data['chain_of_thought']
    score_dict = {score['thought_id']: score['thought_score'] for score in thought_scores}
    
    # Enrich the steps with their scores
    enriched_steps = []
    for i, step in enumerate(data_entry['steps']):
        thought_id = f"thought_{i}"  # Align with zero-based indexing
        score = score_dict.get(thought_id, 0)  # Default score to 0 if not found

        enriched_steps.append({
            "id": thought_id,
            "txt": step[f'thought_{i}'],
            "score": score / 100
        })
    
    combined_data = {
        "question": data_entry["question"],
        "steps": enriched_steps,
        "final_answer": data_entry["final_answer"]
    }

    return combined_data

class OnlineRewardDataset(Dataset):
    def __init__(self, questions, model, tokenizer, num_samples=100, max_length=512):
        self.questions = questions
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly select a question
        question_data = random.choice(self.questions)
        user_question = "Solve this mystery below:\n" + question_data['start_state']
        correct_answer = question_data['solution']
        
        # Generate reasoning response
        messages = [
            {"role": "system", "content": "You are a helpful crime fighting detective"}, 
            {"role": "user", "content": user_question}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_reasoning_prompt=True)
  
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
            )
        
        reasoning_response = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).split("\n\n")

        # Generate progressively expanding subsets of reasoning
        cumulative_reasons = []
        for i in range(len(reasoning_response)):
            cumulative_reasons.append({f"thought_{i}": reasoning_response[i]})
            
        data = {'steps':cumulative_reasons, 'question':user_question, 'final_answer':correct_answer}
        results = evaluate_state(data)

        examples = []
        total_string=""
        for idx_step, step in enumerate(results['steps']):
            total_string += "\n\n" + step['txt']
            # Prepare input for each reasoning step
            msgs = [
                {'role':"user", 'content':results['question']},
                {'role':'reasoning','content':total_string}
            ]

            step_prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False)
            encoded = self.tokenizer(
                step_prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Store user_question, reasoning_response (at this step), and correct_answer for debugging
            examples.append({
                'input_ids': encoded['input_ids'][0],
                'attention_mask': encoded['attention_mask'][0],
                'reward': torch.tensor(step['score'], dtype=torch.float),
                'user_question': results['question'],
                'reasoning_response': total_string,
                'correct_answer': correct_answer
            })

        # Return list of examples for this item
        return examples

def custom_collate_fn(batch):
    # batch is a list of lists (each item from __getitem__ returns multiple examples)
    # Flatten this list
    flat_examples = [ex for ex_list in batch for ex in ex_list]

    input_ids = torch.stack([ex['input_ids'] for ex in flat_examples], dim=0)
    attention_mask = torch.stack([ex['attention_mask'] for ex in flat_examples], dim=0)
    reward = torch.stack([ex['reward'] for ex in flat_examples], dim=0)

    user_question = [ex['user_question'] for ex in flat_examples]
    reasoning_response = [ex['reasoning_response'] for ex in flat_examples]
    correct_answer = [ex['correct_answer'] for ex in flat_examples]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'reward': reward,
        'user_question': user_question,
        'reasoning_response': reasoning_response,
        'correct_answer': correct_answer
    }

def prepare_model(model_name, device="auto"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
        load_in_8bit=True
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model

def train_model(model, tokenizer, questions, 
                num_epochs=25,
                samples_per_epoch=10,
                batch_size=1,
                learning_rate=2e-5,
                max_grad_norm=1.0,
                warmup_steps=100,
                gradient_accumulation_steps=4):
    
    dataset = OnlineRewardDataset(questions, model, tokenizer, num_samples=samples_per_epoch)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    device = next(model.parameters()).device
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
  
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rewards = batch['reward'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            base_loss = outputs.loss
            reward_tensor = rewards.view(-1, 1, 1).expand_as(outputs.logits)
            loss = base_loss * (1 + reward_tensor)
            loss = loss.mean() / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Print example from batch
            if batch_idx % 10 == 0:
                print("\nExample interaction:")
                print(f"Q: {batch['user_question'][0]}")
                print(f"A: {batch['reasoning_response'][0]}")
                print(f"Correct (unused): {batch['correct_answer'][0]}")
                print(f"Reward: {batch['reward'][0].item():.3f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        print(f"Reward stats - Mean: {reward_mean:.4f}, Std: {reward_std:.4f}")
    
    return model

def main():
    model_name = "KingNish/Reasoning-Llama-3b-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = prepare_model(model_name)
    

    questions = load_dataset("danikhan632/OpenMystery")['train']
    trained_model = train_model(
        model, 
        tokenizer, 
        questions,
        samples_per_epoch=1,  # fewer samples for demo
        batch_size=1,
        gradient_accumulation_steps=1
    )
    
    trained_model.save_pretrained("online_trained_reward_model_lora")

if __name__ == "__main__":
    main()
