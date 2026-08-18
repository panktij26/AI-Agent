import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tools import TOOLS_SCHEMA, AVAILABLE_TOOLS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Please set it in your .env file.")

client = OpenAI(api_key=api_key)

def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """
    Executes the agent loop:
    1. Sends input to LLM with tool definitions.
    2. Executes function if requested.
    3. Feeds result back to LLM until a final response is generated.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous AI Agent. "
                "Break down problems logically and use available tools to retrieve facts or calculate answers."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        # Check if the LLM requested any tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"⚙️ [Tool Calling] Executing: {fn_name}({fn_args})")

                if fn_name in AVAILABLE_TOOLS:
                    tool_result = AVAILABLE_TOOLS[fn_name](**fn_args)
                else:
                    tool_result = json.dumps({"error": f"Tool {fn_name} not found"})

                # Return tool output to the model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                })
        else:
            # Model generated a final response without further tool calls
            return message.content

    return "Agent reached maximum iteration limit without resolving the task."

if __name__ == "__main__":
    print("🤖 AI Agent Initialized. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            response = run_agent(user_input)
            print(f"\nAgent: {response}\n{'-'*50}\n")
        except KeyboardInterrupt:
            break
