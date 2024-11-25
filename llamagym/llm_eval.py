

from openai import OpenAI
import json


def evaluate_state(agent_state:str,question:str, goal_state:str, model="gpt-3.5-turbo-0125", base_url:str=None):
    client = OpenAI() if base_url is None else OpenAI(base_url=base_url)
    score = 0
    response = client.chat.completions.create(
        model=model,
        max_tokens=120,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that helps evuluate an agent at problem solving. Given a question and a goal state, evaluate the agent's current state and reasoning steps based on the rubric/solution",
            },
            {
                "role": "user",
                "content": str(
                    "QUESTION/START STATE:"
                    + question
                    + "\nAGENT CURRENT STATE: "
                    + agent_state
                    + "\n:CORRECT GOAL STATE: "
                    + goal_state
                ),
            },
        ],
        functions=[
            {
                "name": "evaluate_state",
                "description": "Evaluates the agent's answer compared to the answer key",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "feedback": {
                            "type": "string",
                            "description": "in a few short words, provide feedback to the agent on their answer",
                        },
                        "score": {
                            "type": "number",
                            "description": "the score of based on the rubric",
                        },

                    },
                },
            }
        ],
    )

    
    data = json.loads(response.choices[0].message.function_call.arguments)
    
    #these are just sample metrics
    score += data.get("score", 0) 

    feedback = data.get("feedback","No feedback")

    return {'reward': score, 'feedback': feedback}



