from agents import Agent, Runner, function_tool, set_tracing_disabled, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os

load_dotenv()
set_tracing_disabled(disabled=True)  # optional

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASEURL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.0-flash"   

# Gemini client
external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASEURL)

# Wrap Gemini client into Agents model
gemini_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model=MODEL_NAME
)


@function_tool(strict_mode=False)
def file_and_folder_handler(
    file_name: str = None,
    folder_name: str = None,
    content: str = None,
    file_path: str = None,
    read: bool = None
):
    try:
        result_messages = []

        # Create folder
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
            result_messages.append(f"Folder '{folder_name}' is ready")

        # Read file
        if read and file_path:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    file_data = f.read()
                result_messages.append(f"Content of {file_path}:\n{file_data}")
            else:
                result_messages.append(f"File {file_path} does not exist")

        # Write file
        if file_name:
            if folder_name:
                full_path = os.path.join(folder_name, file_name)
            else:
                full_path = file_name

            with open(full_path, "w") as f:
                f.write(content if content else "")

            result_messages.append(f"File '{full_path}' created successfully")
            if content:
                result_messages.append(f"Content written to '{full_path}'")

        return "\n".join(result_messages)

    except Exception as e:
        return f"Error occurred: {e}"


file_handler_agent = Agent(
    name="FileHandlerAgent",
    instructions="""You are a helpful file management assistant. You can:
    1. Create folders and files
    2. Write content to files
    3. Read content from files
    4. Always use the tool to perform file and folder operations
    5. Generate HTML, CSS, and JS code snippets when required.

    Examples of what you can do:
    - Create a folder named 'my_folder'
    - Inside my_folder, create a file named e.g 'index.html' and write a basic html boilerplate code in it
    - Read the content of 'my_folder/'index.html'
    - Create a file named 'style.css' and write a basic css code to make  the background color 
    """,
    tools=[file_and_folder_handler],
    model=gemini_model   
)

result = Runner.run_sync(
    starting_agent=file_handler_agent,
    input="create three file html css and js in folder named full_stack"
)

print("Final Output:\n", result.final_output)
