from ollama import chat
from ollama import ChatResponse


class OllamaInteractions:

    def __init__(self, model_name: str = "gemma2:2b") -> str:
        self._model_name = model_name
        self._has_started = False
        self._is_starting = False
        self.memory = []

        # self.startup()

    def startup(self):

        init_prompt = {
                    "role": "user",
                    "content": """You are an AI assistant controlling a robot that understands natural language and can perform actions. Your task is to:  

                    1. **Interpret the user's message** as either a question, a command, or a statement.  
                    2. **Provide a response** enclosed in `<r>` and `</r>`, answering the user naturally.  
                    3. **Determine if movement actions are required** based on the user's message.  
                    4. **If actions are needed**, return them inside `<as>` and `</as>`.  
                    5. **If no movement is required**, return an empty `<as></as>`.  

                    Follow these movement commands when needed:  
                    - `move front`
                    - `move back`  
                    - `move left`  
                    - `move right`  
                    - `turn left`  
                    - `turn right`  """,
                }

        self.memory.append(init_prompt)

        response: ChatResponse = chat(
            model=self._model_name,
            messages=self.memory
        )

        self.memory.append(response.message)

        self._has_started = True
        self._is_starting = False
        print("INIT STATUS:", response.message.content)

    def get_response(self, prompt: str, **kwargs):
        message = {
                    "role": "user",
                    "content": prompt,
                }
        
        self.memory.append(message)

        response: ChatResponse = chat(
            model=self._model_name,
            messages=self.memory,
        )

        self.memory.append(response.message)

        return response.message.content
