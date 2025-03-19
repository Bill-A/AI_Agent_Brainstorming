# -*- coding: utf-8 -*-
"""
# **AI Agents with CrewAI - Workshop Notebook**

This notebook will guide you through building a simple AI agent workflow using the CrewAI framework.  We'll cover the basics of defining agents, tasks, and crews, and then explore how to use AI Studio (MakerSuite) to refine your prompts for optimal performance.

**IMPORTANT:**  Before running this notebook, you will need to obtain your own API key from Google AI Studio and set it as an environment variable. **DO NOT** hardcode your API key into the notebook!

**Learning Objectives:**

*   Understand the core concepts of AI agents and CrewAI.
*   Learn how to define agents with roles, goals, and backstories.
*   Learn how to create tasks for agents to complete.
*   Learn how to create and run a CrewAI workflow.
*   Understand the importance of prompt engineering and how to use AI Studio to refine prompts.
*   (Optional) Explore advanced CrewAI features such as tool integration.

"""

# --- Install required libraries ---
# Use !pip install within a Colab notebook to install packages.
# %pip install crewai
# %pip install google-generativeai

# --- Set up your Google API Key ---
# IMPORTANT:  Get your own API key from Google AI Studio and replace 'YOUR_API_KEY' below.
# DO NOT commit your API key to a public repository!
import os

if not os.environ.get("GOOGLE_API_KEY"):  # Check if it's already set
    os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" #Attendees must change this line to their own key
    print("Please replace 'YOUR_API_KEY' with your actual Google API key.")
else:
    print("Google API key already set in environment variables.")


# --- Define the Research Agent ---
from crewai import Agent

researcher = Agent(
    role="AI Research Analyst",
    goal="Provide accurate and insightful research on specific topics.",
    backstory="A highly skilled AI researcher with a knack for finding the most relevant information.",
    verbose=True,
    allow_delegation=False, # This agent can't delegate tasks
    tools=[] # Will add tools in a later step
)

print("Research Agent created!")

# --- Define the Writing Agent ---
writer = Agent(
    role="Content Writer",
    goal="Craft compelling and well-structured articles based on research.",
    backstory="An experienced content writer with a passion for clear and engaging communication.",
    verbose=True,
    allow_delegation=True # This agent can delegate tasks to other agents
)

print("Writing Agent Created")

# --- Define the Task for the Research Agent ---
from crewai import Task

research_task = Task(
    description="Conduct thorough research on the latest advancements in large language models and provide a detailed summary.",
    agent=researcher  # Assign the task to the research agent
)

print("Research Task Created!")

# --- Define the Task for the Writing Agent ---
writing_task = Task(
    description="Write a compelling blog post summarizing the research on large language models.  The blog post should be engaging and informative, targeting a general audience.",
    agent=writer  # Assign the task to the writing agent
)

print("Writing Task Created")

# --- Create the Crew ---
from crewai import Crew

my_crew = Crew(
    agents=[researcher, writer],  # Add the agents to the crew
    tasks=[research_task, writing_task],  # Add the tasks to the crew
    verbose=2,  # Show detailed output during execution
)

print("Crew Created!")

# --- Run the Crew ---
print("Starting the Crew...")
result = my_crew.kickoff()

# --- Display the Result ---
print("\n\n--- Final Result ---")
print(result)

"""
### **Prompt Engineering - Improving the Research Task**

The quality of the output depends heavily on the prompts you provide to your agents.  Let's use AI Studio (MakerSuite) to refine the prompt for the research task.

1.  **Go to AI Studio:**  Navigate to the Google AI Studio website.
2.  **Create a New Prompt:** Create a new text prompt.
3.  **Copy the Initial Prompt:** Copy the `description` from the `research_task` above:
    ```
    Conduct thorough research on the latest advancements in large language models and provide a detailed summary.
    ```
4.  **Paste it into AI Studio:** Paste the prompt into the prompt box in AI Studio.
5.  **Run and Evaluate:** Run the prompt and evaluate the output.  Does it provide the information you're looking for?
6.  **Iterate and Refine:**
    *   Modify the prompt slightly and run it again.
    *   Try different phrasing, adding more context, or specifying the desired format of the response.
    *   For example, you could refine the prompt to be more specific:
        ```
        Conduct thorough research on the latest advancements in large language models in the last 6 months.  Focus on models exceeding 100 billion parameters, and summarize their architecture, training data, and performance benchmarks in a concise report.
        ```
7.  **Update the Task:**  Once you're happy with the output from AI Studio, copy the refined prompt and update the `description` in the `research_task` in the notebook.  Run the crew again to see the improved results.

**Exercise:** Try refining the prompt for the `writing_task` as well!
"""

"""
### **Adding a Tool (Optional - Requires Tavily API Key)**

Tools allow your agents to interact with the real world.  Let's add the Tavily search tool to the research agent so it can search the web.

**Prerequisites:**

1.  **Get a Tavily API Key:**  Sign up for a free Tavily API key at [https://tavily.com/](https://tavily.com/).
2.  **Set the TAVILY_API_KEY environment variable.**

"""

# --- Install Tavily (a web search tool) ---
# Use !pip install within a Colab notebook to install packages.
# %pip install tavily-python
# import os #Uncomment this line if you've not imported os yet.
from crewai import Agent
from tavily import TavilyClient

#---Get an api key from Tavily
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY" #Replace this with yours
#---set it as an environemnt variable
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# --- Create the Tavily Search Tool ---
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- Update the Agent to Use the Tool ---
researcher.tools = [tavily_client.search] #Assign the search tool

print("Research Agent updated with the Tavily Search Tool!")

# --- Run the Crew Again ---
# Remember to rerun all cells including the definition of the crew before running this section.
print("Starting the Crew with Tavily...")
result = my_crew.kickoff()

# --- Display the Result ---
print("\n\n--- Final Result (with Tavily) ---")
print(result)

"""
### **Experiment and Extend**

Now it's your turn to experiment and extend the capabilities of your AI agents!

**Exercises:**

1.  **Add a new agent to the crew:**  For example, you could add an "Editor" agent to review and improve the writing.
2.  **Modify the goal of an existing agent:**  See how changing the goal affects the agent's behavior.
3.  **Add a tool to the writing agent:**  Perhaps a grammar checker tool.
4.  **Refine the prompts to improve the output quality:**  Experiment with different phrasing and instructions.
5.  **Explore other CrewAI features:**  Check out the CrewAI documentation for more advanced features such as memory management and agent communication.

**Further Resources:**

*   **CrewAI Documentation:** [https://www.crewai.com/](https://www.crewai.com/) (Replace with the actual documentation link when it's available)
*   **Google AI Studio:** [https://makersuite.google.com/app/home](https://makersuite.google.com/app/home) (or whatever the current URL is)
*   **Tavily API:** [https://tavily.com/](https://tavily.com/)

Have fun building your own AI agent workflows!
"""
