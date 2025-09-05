import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from game_tool import roll_dice, generate_event

# Load environment variables from .env file
load_dotenv()
# Initialize the OpenAI model

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Adjust the base URL as needed
)

model = OpenAIChatCompletionsModel(model="gemini-2.0-flash",openai_client=client)
config = RunConfig(model=model, tracing_disabled=True)

# Define the game agents

narrator_agent = Agent(
    name="NarratorAgent",
    instructions="you narrat the advenure. Ask the player for choices.",
    model=model,
)

monster_agent = Agent(
    name="MonsterAgent",
    instructions="you handle monster encounters using roll_dice and generate_event.",
    model=model,
    tools=[roll_dice, generate_event]
)

item_agent = Agent(
    name="ItemAgent",
    instructions=" you provide rewards or items to the player.",
    model=model,
)

def main():
    print("Welcome to the Adventure Game!")
    choice = input("Do you enter the forest or turn back? ->")

    result1 = Runner.run_sync(narrator_agent, choice, run_config=config)
    print("\n story:", result1.final_output)

    result2 = Runner.run_sync(monster_agent, "Start encounter", run_config=config)
    print("\n Encounter:", result2.final_output)

    result3 = Runner.run_sync(item_agent, "Given reward", run_config=config)
    print("\n story:", result3.final_output)

if __name__ == "__main__":
    main()
    # print("Game Over. Thanks for playing!")    