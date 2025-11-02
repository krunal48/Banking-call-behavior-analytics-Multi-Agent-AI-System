from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    print(city)
    return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = LlmAgent(
    model=LiteLlm(model="openai/gpt-4o"),
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge. You can tell the current time in cities',
    tools=[get_current_time]
)

