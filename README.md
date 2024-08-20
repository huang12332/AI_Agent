
# AI Agent with Llama3.1 and GoogleSerperAPIWrapper

## Overview

This project demonstrates the integration of an AI agent with the Llama3.1 language model and the GoogleSerperAPIWrapper tool. The agent is capable of performing various tasks such as translation, chaining with prompt templates, and tool calling. The project is implemented in a Jupyter Notebook and can be easily extended or modified for different use cases.

## Features

- **Language Model Initialization**: The Llama3.1 model is initialized with specific parameters such as temperature.
- **Translation**: The agent can translate English sentences to Chinese.
- **Chaining with Prompt Templates**: The agent can chain the language model with a prompt template to perform more complex tasks.
- **Tool Calling**: The agent can call external tools like the GoogleSerperAPIWrapper to fetch real-time data.
- **Agent Execution**: The agent can be executed in a streaming mode to handle real-time requests.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/ai-agent-llama3.1.git
    cd ai-agent-llama3.1
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up API Keys**:

    Obtain API keys for the GoogleSerperAPIWrapper and set them in your environment variables or in the notebook.

## Usage

1. **Open the Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

2. **Run the Notebook**:

    Open `AI Agent with Llama3.1, tools GoogleSerperAPIWrapper .ipynb` and run the cells sequentially.

### Example Tasks:

- **Translation**: Translate English sentences to Chinese.
- **Chaining**: Chain the model with a prompt template to perform a task.
- **Tool Calling**: Use the GoogleSerperAPIWrapper to fetch data.

## Examples

### Translation

```python
from langchain_core.messages import AIMessage

messages = [
    ("system", "You are a helpful assistant that translates English to Chinese, Translate the user sentence."),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
```

### Chaining with Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant that translate {input_language} to {output_language}."),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "Chinese",
        "input": "I Love programming",
    }
)
```

### Tool Calling

```python
from langchain_ollama import ChatOllama

def validate_user(user_id: int, addresses: List) -> bool:
    return True

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke(
    "Could you validate user 123? They previously lived at 123 Fake St in Boston MA and 234 Pretend Boulevard in Houston TX."
)
print(result.tool_calls)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## Acknowledgements

- **LangChain**
- **Llama3.1 Model**
- **GoogleSerperAPIWrapper**
```