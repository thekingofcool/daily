---
layout: post
title: Day 1b - Agent Architectures
date: 2025-11-10
author: thekingofcool
description: ""
categories: thoughts
---

##### Copyright 2025 Google LLC.


```python
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# 🚀 Multi-Agent Systems & Workflow Patterns

**Welcome to the Kaggle 5-day Agents course!**

In the previous notebook, you built a **single agent** that could take action. Now, you'll learn how to scale up by building **agent teams**.

Just like a team of people, you can create specialized agents that collaborate to solve complex problems. This is called a **multi-agent system**, and it's one of the most powerful concepts in AI agent development.

In this notebook, you'll:

- ✅ Learn when to use multi-agent systems in [Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
- ✅ Build your first system using an LLM as a "manager"
- ✅ Learn three core workflow patterns (Sequential, Parallel, and Loop) to coordinate your agent teams

**ℹ️ Note: No submission required!**

This notebook is for your hands-on practice and learning only. You **do not** need to submit it anywhere to complete the course.

## 📖 Get started with Kaggle Notebooks

If this is your first time using Kaggle Notebooks, welcome! You can learn more about using Kaggle Notebooks [in the documentation](https://www.kaggle.com/docs/notebooks).

Here's how to get started:

**1. Verify Your Account (Required)**

To use the Kaggle Notebooks in this course, you'll need to verify your account with a phone number.

You can do this in your [Kaggle settings](https://www.kaggle.com/settings).

**2. Make Your Own Copy**

To run any code in this notebook, you first need your own editable copy.

Click the `Copy and Edit` button in the top-right corner.

![Copy and Edit button](https://storage.googleapis.com/kaggle-media/Images/5gdai_sc_1.png)

This creates a private copy of the notebook just for you.

**3. Run Code Cells**

Once you have your copy, you can run code.

Click the ▶️ Run button next to any code cell to execute it.

![Run cell button](https://storage.googleapis.com/kaggle-media/Images/5gdai_sc_2.png)

Run the cells in order from top to bottom.

**4. If You Get Stuck**

To restart: Select `Factory reset` from the `Run` menu.

For help: Ask questions on the [Kaggle Discord](https://discord.com/invite/kaggle) server.

### Section 1

## ⚙️ Setup

### Install dependencies

The Kaggle Notebooks environment includes a pre-installed version of the [google-adk](https://google.github.io/adk-docs/) library for Python and its required dependencies, so you don't need to install additional packages in this notebook.

To install and use ADK in your own Python development environment outside of this course, you can do so by running:

```
pip install google-adk
```

### 1.1 Configure your Gemini API Key

This notebook uses the [Gemini API](https://ai.google.dev/gemini-api/docs), which requires authentication.

**1. Get your API key**

If you don't have one already, create an [API key in Google AI Studio](https://aistudio.google.com/app/api-keys).

**2. Add the key to Kaggle Secrets**

Next, you will need to add your API key to your Kaggle Notebook as a Kaggle User Secret.

1. In the top menu bar of the notebook editor, select `Add-ons` then `Secrets`.
2. Create a new secret with the label `GOOGLE_API_KEY`.
3. Paste your API key into the "Value" field and click "Save".
4. Ensure that the checkbox next to `GOOGLE_API_KEY` is selected so that the secret is attached to the notebook.

**3. Authenticate in the notebook**

Run the cell below to complete authentication.


```python
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")
```

    ✅ Gemini API key setup complete.


### 1.2 Import ADK components

Now, import the specific components you'll need from the Agent Development Kit and the Generative AI library. This keeps your code organized and ensures we have access to the necessary building blocks.


```python
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("✅ ADK components imported successfully.")
```

    ✅ ADK components imported successfully.


---
### Section 2

## 🤔 Why Multi-Agent Systems? + Your First Multi-Agent

**The Problem: The "Do-It-All" Agent**

Single agents can do a lot. But what happens when the task gets complex? A single "monolithic" agent that tries to do research, writing, editing, and fact-checking all at once becomes a problem. Its instruction prompt gets long and confusing. It's hard to debug (which part failed?), difficult to maintain, and often produces unreliable results.

**The Solution: A Team of Specialists**

Instead of one "do-it-all" agent, we can build a **multi-agent system**. This is a team of simple, specialized agents that collaborate, just like a real-world team. Each agent has one clear job (e.g., one agent *only* does research, another *only* writes). This makes them easier to build, easier to test, and much more powerful and reliable when working together.

To learn more, check out the documentation related to [LLM agents in ADK](https://google.github.io/adk-docs/agents/llm-agents/).

**Architecture: Single Agent vs Multi-Agent Team**

<!--
```mermaid
graph TD
    subgraph Single["❌ Monolithic Agent"]
        A["One Agent Does Everything"]
    end

    subgraph Multi["✅ Multi-Agent Team"]
        B["Root Coordinator"] -- > C["Research Specialist"]
        B -- > E["Summary Specialist"]

        C -- >|findings| F["Shared State"]
        E -- >|summary| F
    end

    style A fill:#ffcccc
    style B fill:#ccffcc
    style F fill:#ffffcc
```
-->

<img width="800" src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day1/multi-agent-team.png" alt="Multi-agent Team" />

### 2.1 Example: Research & Summarization System

Let's build a system with two specialized agents:

1. **Research Agent** - Searches for information using Google Search
2. **Summarizer Agent** - Creates concise summaries from research findings


```python
# Research Agent: Its job is to use the google_search tool and present findings.
research_agent = Agent(
    name="ResearchAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
    tools=[google_search],
    output_key="research_findings", # The result of this agent will be stored in the session state with this key.
)

print("✅ research_agent created.")
```

    ✅ research_agent created.



```python
# Summarizer Agent: Its job is to summarize the text it receives.
summarizer_agent = Agent(
    name="SummarizerAgent",
    model="gemini-2.5-flash-lite",
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Read the provided research findings: {research_findings}
Create a concise summary as a bulleted list with 3-5 key points.""",
    output_key="final_summary",
)

print("✅ summarizer_agent created.")
```

    ✅ summarizer_agent created.


Refer to the ADK documentation for more information on [guiding agents with clear and specific instructions](https://google.github.io/adk-docs/agents/llm-agents/).

Then we bring the agents together under a root agent, or coordinator:


```python
# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
root_agent = Agent(
    name="ResearchCoordinator",
    model="gemini-2.5-flash-lite",
    # This instruction tells the root agent HOW to use its tools (which are the other agents).
    instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
3. Finally, present the final summary clearly to the user as your response.""",
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[
        AgentTool(research_agent),
        AgentTool(summarizer_agent)
    ],
)

print("✅ root_agent created.")
```

    ✅ root_agent created.


Here we're using `AgentTool` to wrap the sub-agents to make them callable tools for the root agent. We'll explore `AgentTool` in-detail on Day 2.

Let's run the agent and ask it about a topic:


```python
runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("What are the latest advancements in quantum computing and what do they mean for AI?")
```

    
     ### Created new session: debug_session_id
    
    User > What are the latest advancements in quantum computing and what do they mean for AI?


    WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
    WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.


    ResearchCoordinator > Quantum computing's unique capabilities, like superposition in qubits, offer unprecedented speed and efficiency, enabling AI to overcome current limitations. Quantum AI (QAI) is expected to accelerate AI performance, enhance complex simulations (e.g., for drug discovery), solve previously intractable problems, and improve the efficiency of AI models. The synergy between AI and quantum computing is driving breakthroughs in specific AI fields like NLP and autonomous systems, while AI is also helping to advance quantum computing itself. Major tech companies are investing in quantum computing, with early prototypes and services emerging, suggesting significant future growth and industry transformation.


You've just built your first multi-agent system! You used a single "coordinator" agent to manage the workflow, which is a powerful and flexible pattern.

‼️ However, **relying on an LLM's instructions to control the order can sometimes be unpredictable.** Next, we'll explore a different pattern that gives you guaranteed, step-by-step execution.

---

### Section 3
## 🚥 Sequential Workflows - The Assembly Line

**The Problem: Unpredictable Order**

The previous multi-agent system worked, but it relied on a **detailed instruction prompt** to force the LLM to run steps in order. This can be unreliable. A complex LLM might decide to skip a step, run them in the wrong order, or get "stuck," making the process unpredictable.

**The Solution: A Fixed Pipeline**

When you need tasks to happen in a **guaranteed, specific order**, you can use a `SequentialAgent`. This agent acts like an assembly line, running each sub-agent in the exact order you list them. The output of one agent automatically becomes the input for the next, creating a predictable and reliable workflow.

**Use Sequential when:** Order matters, you need a linear pipeline, or each step builds on the previous one.

To learn more, check out the documentation related to [sequential agents in ADK](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/).

**Architecture: Blog Post Creation Pipeline**

<!--
```mermaid
graph LR
    A["User Input: Blog about AI"] -- > B["Outline Agent"]
    B -- >|blog_outline| C["Writer Agent"]
    C -- >|blog_draft| D["Editor Agent"]
    D -- >|final_blog| E["Output"]

    style B fill:#ffcccc
    style C fill:#ccffcc
    style D fill:#ccccff
```
-->

<img width="1000" src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day1/sequential-agent.png" alt="Sequential Agent" />

### 3.1 Example: Blog Post Creation with Sequential Agents

Let's build a system with three specialized agents:

1. **Outline Agent** - Creates a blog outline for a given topic
2. **Writer Agent** - Writes a blog post
3. **Editor Agent** - Edits a blog post draft for clarity and structure


```python
# Outline Agent: Creates the initial blog post outline.
outline_agent = Agent(
    name="OutlineAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Create a blog outline for the given topic with:
    1. A catchy headline
    2. An introduction hook
    3. 3-5 main sections with 2-3 bullet points for each
    4. A concluding thought""",
    output_key="blog_outline", # The result of this agent will be stored in the session state with this key.
)

print("✅ outline_agent created.")
```

    ✅ outline_agent created.



```python
# Writer Agent: Writes the full blog post based on the outline from the previous agent.
writer_agent = Agent(
    name="WriterAgent",
    model="gemini-2.5-flash-lite",
    # The `{blog_outline}` placeholder automatically injects the state value from the previous agent's output.
    instruction="""Following this outline strictly: {blog_outline}
    Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
    output_key="blog_draft", # The result of this agent will be stored with this key.
)

print("✅ writer_agent created.")
```

    ✅ writer_agent created.



```python
# Editor Agent: Edits and polishes the draft from the writer agent.
editor_agent = Agent(
    name="EditorAgent",
    model="gemini-2.5-flash-lite",
    # This agent receives the `{blog_draft}` from the writer agent's output.
    instruction="""Edit this draft: {blog_draft}
    Your task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.""",
    output_key="final_blog", # This is the final output of the entire pipeline.
)

print("✅ editor_agent created.")
```

    ✅ editor_agent created.


Then we bring the agents together under a sequential agent, which runs the agents in the order that they are listed:


```python
root_agent = SequentialAgent(
    name="BlogPipeline",
    sub_agents=[outline_agent, writer_agent, editor_agent],
)

print("✅ Sequential Agent created.")
```

    ✅ Sequential Agent created.


Let's run the agent and give it a topic to write a blog post about:


```python
runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("Write a blog post about the potential benefits of Unity Software for the world")
```

    
     ### Created new session: debug_session_id
    
    User > Write a blog post about the potential benefits of Unity Software for the world
    OutlineAgent > Here's a blog outline about the potential benefits of Unity Software for the world:
    
    ## Headline: Beyond Games: How Unity Software is Reshaping Our World
    
    ### Introduction Hook:
    Imagine a world where you can visualize complex medical procedures before they happen, train surgeons in a risk-free environment, or even walk through the blueprint of your future home. This isn't science fiction; it's the rapidly expanding reality powered by tools like Unity Software. While widely known for its dominance in game development, Unity's impact stretches far beyond entertainment, offering transformative solutions across a multitude of industries and promising significant benefits for humanity.
    
    ### Main Sections:
    
    1.  **Democratizing Creation: Empowering Anyone to Build the Future**
        *   **Accessibility & Ease of Use:** Unity's user-friendly interface and extensive asset store lower the barrier to entry, allowing individuals and smaller teams with limited resources to bring complex ideas to life.
        *   **Cross-Platform Deployment:** Developers can build once and deploy across a vast array of devices – from PCs and consoles to mobile phones, VR/AR headsets, and web browsers – reaching a global audience with a single project.
        *   **Fostering Innovation:** By making powerful creation tools widely available, Unity encourages a more diverse pool of creators to innovate, leading to a richer ecosystem of applications and experiences.
    
    2.  **Revolutionizing Education and Training: Learning by Doing, Safely**
        *   **Immersive Learning Environments:** Unity enables the creation of realistic simulations for training in fields like medicine, aviation, engineering, and manufacturing, allowing for hands-on practice without real-world risks.
        *   **Engaging Educational Content:** From interactive science lessons to historical recreations, Unity can make learning more dynamic, engaging, and effective for students of all ages.
        *   **Skill Development for the Future:** As industries increasingly adopt digital tools, proficiency in Unity-based development can equip individuals with highly sought-after skills for the modern workforce.
    
    3.  **Transforming Industries: Beyond the Screen**
        *   **Architecture, Engineering, and Construction (AEC):** Visualize designs in 3D, conduct virtual walkthroughs, and identify potential issues early in the development process, saving time and resources.
        *   **Automotive and Transportation:** Develop realistic driving simulators for autonomous vehicle testing, design vehicle interiors virtually, and create engaging marketing experiences.
        *   **Manufacturing and Industrial Design:** Create digital twins of factories and products for optimization, train workers on complex machinery safely, and streamline prototyping processes.
        *   **Healthcare Advancements:** From surgical planning and medical visualization to patient rehabilitation and remote diagnostics, Unity is enabling groundbreaking healthcare solutions.
    
    ### Concluding Thought:
    Unity Software is more than just a game engine; it's a powerful platform for imagination and innovation. As its capabilities continue to expand and its accessibility grows, we can expect to see even more profound benefits emerge, shaping how we learn, work, heal, and interact with the world around us, truly pushing the boundaries of what's possible for the betterment of society.
    WriterAgent > ## Beyond Games: How Unity Software is Reshaping Our World
    
    Ever imagined visualizing a complex surgery before it happens, training pilots in a completely safe virtual cockpit, or walking through the digital blueprint of your dream home? This isn't science fiction; it's the expanding reality brought to life by tools like Unity Software. While celebrated for its game development prowess, Unity's impact is now rippling across industries, promising transformative benefits for us all.
    
    At its core, Unity democratizes creation. Its accessible interface and vast asset store empower individuals and small teams to build incredible experiences, regardless of budget. Build once, and deploy everywhere – from phones and PCs to cutting-edge VR/AR headsets. This widespread availability fuels a diverse wave of innovation, enriching our digital world with new applications and ideas.
    
    Education and training are also undergoing a revolution. Unity crafts immersive simulations for high-stakes fields like medicine and aviation, allowing for invaluable hands-on practice without real-world risk. Learning becomes more engaging, whether it's exploring interactive historical events or mastering complex scientific concepts. This prepares a future workforce with essential digital skills.
    
    Beyond these realms, Unity is reshaping industries. Architects can virtually tour buildings before construction, car manufacturers test autonomous systems in realistic simulators, and factories can create "digital twins" for optimization. Even healthcare benefits, with advancements in surgical planning, patient rehabilitation, and remote diagnostics.
    
    Unity Software is far more than a game engine; it's a catalyst for imagination and progress. As its capabilities grow, expect to see even more profound ways it shapes how we learn, work, heal, and interact with our world, pushing the boundaries of what's possible for the betterment of society.
    EditorAgent > ## Beyond Games: How Unity Software is Reshaping Our World
    
    Ever imagined visualizing a complex surgery before it happens, training pilots in a completely safe virtual cockpit, or walking through the digital blueprint of your dream home? This isn't science fiction; it's the expanding reality brought to life by tools like Unity Software. While celebrated for its game development prowess, Unity's impact is now rippling across industries, promising transformative benefits for us all.
    
    At its core, Unity democratizes creation. Its accessible interface and vast asset store empower individuals and small teams to build incredible experiences, regardless of budget. Developers can **build once and deploy everywhere** – from phones and PCs to cutting-edge VR/AR headsets. This widespread availability fuels a diverse wave of innovation, enriching our digital world with new applications and ideas.
    
    Education and training are also undergoing a revolution. Unity crafts immersive simulations for high-stakes fields like medicine and aviation, allowing for invaluable hands-on practice without real-world risk. Learning becomes more engaging, whether it's exploring interactive historical events or mastering complex scientific concepts. This prepares a future workforce with essential digital skills.
    
    Beyond these realms, Unity is reshaping industries. Architects can virtually tour buildings before construction, car manufacturers test autonomous systems in realistic simulators, and factories can create "digital twins" for optimization. Even healthcare benefits, with advancements in surgical planning, patient rehabilitation, and remote diagnostics.
    
    Unity Software is far more than a game engine; it's a catalyst for imagination and progress. As its capabilities grow, expect to see even more profound ways it shapes how we learn, work, heal, and interact with our world, pushing the boundaries of what's possible for the betterment of society.


👏 Great job! You've now created a reliable "assembly line" using a sequential agent, where each step runs in a predictable order.

**This is perfect for tasks that build on each other, but it's slow if the tasks are independent.** Next, we'll look at how to run multiple agents at the same time to speed up your workflow.

---
### Section 4
## 🛣️ Parallel Workflows - Independent Researchers

**The Problem: The Bottleneck**

The previous sequential agent is great, but it's an assembly line. Each step must wait for the previous one to finish. What if you have several tasks that are **not dependent** on each other? For example, researching three *different* topics. Running them in sequence would be slow and inefficient, creating a bottleneck where each task waits unnecessarily.

**The Solution: Concurrent Execution**

When you have independent tasks, you can run them all at the same time using a `ParallelAgent`. This agent executes all of its sub-agents concurrently, dramatically speeding up the workflow. Once all parallel tasks are complete, you can then pass their combined results to a final 'aggregator' step.

**Use Parallel when:** Tasks are independent, speed matters, and you can execute concurrently.

To learn more, check out the documentation related to [parallel agents in ADK](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/).

**Architecture: Multi-Topic Research**

<!--
```mermaid
graph TD
    A["User Request: Research 3 topics"] -- > B["Parallel Execution"]
    B -- > C["Tech Researcher"]
    B -- > D["Health Researcher"]
    B -- > E["Finance Researcher"]

    C -- > F["Aggregator"]
    D -- > F
    E -- > F
    F -- > G["Combined Report"]

    style B fill:#ffffcc
    style F fill:#ffccff
```
-->

<img width="600" src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day1/parallel-agent.png" alt="Parallel Agent" />

### 4.1 Example: Parallel Multi-Topic Research

Let's build a system with four agents:

1. **Tech Researcher** - Researches AI/ML news and trends
2. **Health Researcher** - Researches recent medical news and trends
3. **Finance Researcher** - Researches finance and fintech news and trends
4. **Aggregator Agent** - Combines all research findings into a single summary


```python
# Tech Researcher: Focuses on AI and ML trends.
tech_researcher = Agent(
    name="TechResearcher",
    model="gemini-2.5-flash-lite",
    instruction="""Research the latest AI/ML trends. Include 3 key developments,
the main companies involved, and the potential impact. Keep the report very concise (100 words).""",
    tools=[google_search],
    output_key="tech_research", # The result of this agent will be stored in the session state with this key.
)

print("✅ tech_researcher created.")
```

    ✅ tech_researcher created.



```python
# Health Researcher: Focuses on medical breakthroughs.
health_researcher = Agent(
    name="HealthResearcher",
    model="gemini-2.5-flash-lite",
    instruction="""Research recent medical breakthroughs. Include 3 significant advances,
their practical applications, and estimated timelines. Keep the report concise (100 words).""",
    tools=[google_search],
    output_key="health_research", # The result will be stored with this key.
)

print("✅ health_researcher created.")
```

    ✅ health_researcher created.



```python
# Finance Researcher: Focuses on fintech trends.
finance_researcher = Agent(
    name="FinanceResearcher",
    model="gemini-2.5-flash-lite",
    instruction="""Research current fintech trends. Include 3 key trends,
their market implications, and the future outlook. Keep the report concise (100 words).""",
    tools=[google_search],
    output_key="finance_research", # The result will be stored with this key.
)

print("✅ finance_researcher created.")
```

    ✅ finance_researcher created.



```python
# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = Agent(
    name="AggregatorAgent",
    model="gemini-2.5-flash-lite",
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research findings into a single executive summary:

    **Technology Trends:**
    {tech_research}
    
    **Health Breakthroughs:**
    {health_research}
    
    **Finance Innovations:**
    {finance_research}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
    output_key="executive_summary", # This will be the final output of the entire system.
)

print("✅ aggregator_agent created.")
```

    ✅ aggregator_agent created.


👉 **Then we bring the agents together under a parallel agent, which is itself nested inside of a sequential agent.**

This design ensures that the research agents run first in parallel, then once all of their research is complete, the aggregator agent brings together all of the research findings into a single report:


```python
# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_researcher, health_researcher, finance_researcher],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)

print("✅ Parallel and Sequential Agents created.")
```

    ✅ Parallel and Sequential Agents created.


Let's run the agent and give it a prompt to research the given topics:


```python
runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("Run the daily executive briefing on Tech, Health, and Finance")
```

    
     ### Created new session: debug_session_id
    
    User > Run the daily executive briefing on Tech, Health, and Finance
    HealthResearcher > Here's your executive briefing on recent breakthroughs:
    
    **Health:** Gene therapy is revolutionizing treatment for inherited diseases like sickle cell anemia and beta-thalassemia, with potential for blindness and other genetic conditions. CAR T-cell therapy shows promise for brain cancers, and AI is improving diagnostics for heart health and disease detection. Estimated timelines for widespread adoption vary, but clinical trials are ongoing, suggesting significant impact within 3-5 years for many applications.
    
    **Technology:** Agentic AI, capable of autonomous task execution, is a major trend, with applications in self-driving cars and process automation, expected to become more prevalent by 2028. Advancements in quantum computing are accelerating drug discovery and diagnostics. Spatial computing is merging physical and digital realms, with widespread adoption projected for the next decade.
    
    **Finance:** AI and machine learning are transforming risk assessment and user experience in finance. Blockchain is streamlining processes for lending and fraud detection, enhancing transparency. The development of Central Bank Digital Currencies (CBDCs) and tokenization are poised to redefine cross-border settlements within the next decade.
    FinanceResearcher > **Executive Briefing: Tech, Health, and Finance Trends**
    
    **Technology:** The democratization of Artificial Intelligence (AI) is the dominant trend, moving beyond large enterprises to smaller businesses and integrated across various industries like finance and healthcare. Advancements in quantum computing and the expansion of the Internet of Things (IoT) are also significant, promising more complex processing and wider connectivity.
    
    **Health:** AI is increasingly utilized in healthcare for early disease detection and personalized treatments. Mental health awareness and services continue to expand, with a growing focus on holistic wellness. Digestive health and the use of probiotics/prebiotics are mainstream priorities for consumers seeking proactive health management.
    
    **Finance:** Generative AI is a major disruptive force, reshaping customer experiences and wealth management. A notable trend is the anticipated lowering of interest rates, potentially stimulating investment and consumer spending. Sustainability and ESG considerations are also becoming critical, influencing investment decisions and corporate strategies.
    
    **Market Implications:** These trends indicate a move towards more integrated and AI-driven solutions across all sectors. Increased personalization in finance and health, alongside the growing importance of sustainable practices, will shape consumer demand and business strategies.
    
    **Future Outlook:** Expect continued rapid development in AI, leading to more sophisticated applications. Healthcare will become more accessible and personalized, while the financial sector will likely see increased digital innovation, with a stronger emphasis on responsible and sustainable practices.
    TechResearcher > **Key AI/ML Trends and Developments**
    
    1.  **Generative AI Expansion:** Beyond text, generative AI is now creating complex graphics, video, and music. Companies like Google (Muse, Imagen) and OpenAI (GPT models) are leading this evolution, significantly enhancing artistic expression and practical applications.
    2.  **Shift to Smaller, Specialized Models:** There's a growing trend towards Smaller Language Models (SLMs) and domain-specific AI models that often outperform larger, general-purpose ones. This allows for more efficient and tailored AI solutions.
    3.  **Agentic AI and Automation:** AI systems are increasingly capable of completing full tasks with minimal human input, driving automation across various sectors. This trend is transforming how businesses operate, from customer service to logistics.
    
    **Main Companies Involved:**
    *   **Google:** Developing advanced generative models (Muse, Imagen) and offering AI services via Google Cloud with its Gemini ecosystem.
    *   **OpenAI:** A key player in large language models (GPT) and a partner with Microsoft for AI development.
    *   **NVIDIA:** Providing the essential GPU hardware for AI model training and offering comprehensive AI software solutions.
    *   **Microsoft:** Heavily invested in OpenAI, leveraging its Azure platform for AI model development and offering its own AI services.
    *   **Amazon:** Investing in AI services through AWS, including its Bedrock platform for language models.
    
    **Potential Impact:**
    These trends are driving significant business growth, enhancing operational efficiency, and unlocking new opportunities across industries like healthcare and finance. The advancements also raise important considerations around AI ethics, data privacy, and the need for workforce upskilling. Businesses are rapidly increasing AI investments to remain competitive and future-proof their operations.
    AggregatorAgent > **Executive Summary: Daily Briefing on Tech, Health, and Finance Trends**
    
    Artificial Intelligence (AI) is the unifying force across technology, health, and finance, driving unprecedented innovation and market shifts. Generative AI is expanding beyond text to create complex media, while smaller, specialized models and agentic AI are enabling sophisticated automation and task completion. This AI democratization is democratizing access, moving beyond large enterprises to benefit smaller businesses and integrate across all sectors.
    
    In healthcare, AI is revolutionizing diagnostics and personalized treatments, complementing breakthroughs in gene therapy and CAR T-cell therapy, with significant impact expected within 3-5 years. Simultaneously, quantum computing is accelerating drug discovery and diagnostics, while spatial computing merges physical and digital realms.
    
    The finance sector is being reshaped by AI and machine learning for risk assessment and customer experience, alongside blockchain's role in streamlining processes and enhancing transparency. Emerging trends like Central Bank Digital Currencies (CBDCs) and tokenization are poised to redefine global settlements. Notably, sustainability and ESG considerations are increasingly influencing investment and corporate strategy.
    
    Collectively, these trends point to a future of highly integrated, personalized, and AI-driven solutions, necessitating workforce upskilling and careful consideration of ethical implications, data privacy, and responsible innovation.


🎉 Great! You've seen how parallel agents can dramatically speed up workflows by running independent tasks concurrently.

So far, all our workflows run from start to finish and then stop. **But what if you need to review and improve an output multiple times?** Next, we'll build a workflow that can loop and refine its own work.

---
### Section 5
## ➰ Loop Workflows - The Refinement Cycle

**The Problem: One-Shot Quality**

All the workflows we've seen so far run from start to finish. The `SequentialAgent` and `ParallelAgent` produce their final output and then stop. This 'one-shot' approach isn't good for tasks that require refinement and quality control. What if the first draft of our story is bad? We have no way to review it and ask for a rewrite.

**The Solution: Iterative Refinement**

When a task needs to be improved through cycles of feedback and revision, you can use a `LoopAgent`. A `LoopAgent` runs a set of sub-agents repeatedly *until a specific condition is met or a maximum number of iterations is reached.* This creates a refinement cycle, allowing the agent system to improve its own work over and over.

**Use Loop when:** Iterative improvement is needed, quality refinement matters, or you need repeated cycles.

To learn more, check out the documentation related to [loop agents in ADK](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/).

**Architecture: Story Writing & Critique Loop**

<!--
```mermaid
graph TD
    A["Initial Prompt"] -- > B["Writer Agent"]
    B -- >|story| C["Critic Agent"]
    C -- >|critique| D{"Iteration < Max<br>AND<br>Not Approved?"}
    D -- >|Yes| B
    D -- >|No| E["Final Story"]

    style B fill:#ccffcc
    style C fill:#ffcccc
    style D fill:#ffffcc
```
-->

<img width="250" src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day1/loop-agent.png" alt="Loop Agent" />

### 5.1 Example: Iterative Story Refinement

Let's build a system with two agents:

1. **Writer Agent** - Writes a draft of a short story
2. **Critic Agent** - Reviews and critiques the short story to suggest improvements


```python
# This agent runs ONCE at the beginning to create the first draft.
initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Based on the user's prompt, write the first draft of a short story (around 100-150 words).
    Output only the story text, with no introduction or explanation.""",
    output_key="current_story", # Stores the first draft in the state.
)

print("✅ initial_writer_agent created.")
```

    ✅ initial_writer_agent created.



```python
# This agent's only job is to provide feedback or the approval signal. It has no tools.
critic_agent = Agent(
    name="CriticAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a constructive story critic. Review the story provided below.
    Story: {current_story}
    
    Evaluate the story's plot, characters, and pacing.
    - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    output_key="critique", # Stores the feedback in the state.
)

print("✅ critic_agent created.")
```

    ✅ critic_agent created.


Now, we need a way for the loop to actually stop based on the critic's feedback. The `LoopAgent` itself doesn't automatically know that "APPROVED" means "stop."

We need an agent to give it an explicit signal to terminate the loop.

We do this in two parts:

1. A simple Python function that the `LoopAgent` understands as an "exit" signal.
2. An agent that can call that function when the right condition is met.

First, you'll define the `exit_loop` function:


```python
# This is the function that the RefinerAgent will call to exit the loop.
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}

print("✅ exit_loop function created.")
```

    ✅ exit_loop function created.


To let an agent call this Python function, we wrap it in a `FunctionTool`. Then, we create a `RefinerAgent` that has this tool.

👉 **Notice its instructions:** this agent is the "brain" of the loop. It reads the `{critique}` from the `CriticAgent` and decides whether to (1) call the `exit_loop` tool or (2) rewrite the story.


```python
# This agent refines the story based on critique OR calls the exit_loop function.
refiner_agent = Agent(
    name="RefinerAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a story refiner. You have a story draft and critique.
    
    Story Draft: {current_story}
    Critique: {critique}
    
    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.""",
    
    output_key="current_story", # It overwrites the story with the new, refined version.
    tools=[FunctionTool(exit_loop)], # The tool is now correctly initialized with the function reference.
)

print("✅ refiner_agent created.")
```

    ✅ refiner_agent created.


Then we bring the agents together under a loop agent, which is itself nested inside of a sequential agent.

This design ensures that the system first produces an initial story draft, then the refinement loop runs up to the specified number of `max_iterations`:


```python
# The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2, # Prevents infinite loops
)

# The root agent is a SequentialAgent that defines the overall workflow: Initial Write -> Refinement Loop.
root_agent = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer_agent, story_refinement_loop],
)

print("✅ Loop and Sequential Agents created.")
```

    ✅ Loop and Sequential Agents created.


Let's run the agent and give it a topic to write a short story about:


```python
runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("Write a short story about Unity Software stock price rocket in recent two years.")
```

    
     ### Created new session: debug_session_id
    
    User > Write a short story about Unity Software stock price rocket in recent two years.
    InitialWriterAgent > The ticker flashed U, then N, then I, then T, then Y. For two years, it had been a slow climb, a cautious ascent for Unity Software. Analysts whispered about potential, but the market remained tepid. Then, something shifted. It wasn't a single event, but a confluence of factors: a groundbreaking new engine release, a surge in indie game development, and a growing adoption of the platform for non-gaming applications like architectural visualization and automotive design.
    
    Suddenly, the whispers turned to roars. The stock price, once a steady incline, began a meteoric rise. It was a rocket launch, fueled by investor confidence and a palpable buzz. Shareholders who had patiently held watched their investments multiply, while new investors scrambled to get in on the ground floor of what was clearly becoming an industry titan. The chart, once a gentle curve, was now a vertical line, pointing straight to the moon.
    CriticAgent > This story effectively captures the narrative arc of a company's stock experiencing a dramatic rise. Here's a breakdown:
    
    **Plot:** The plot is straightforward and easy to follow. It establishes a period of slow growth and anticipation, followed by a clear turning point and subsequent rapid ascent. The "confluence of factors" provides a believable catalyst for the change. The inclusion of the "moon" imagery is a common and recognizable trope for stock market success.
    
    **Characters:** There are no explicit characters in the traditional sense. The "characters" are the market, analysts, shareholders, and investors, all acting as collective forces. This works for a story focused on a stock's performance. Unity Software itself is personified as the entity experiencing the climb.
    
    **Pacing:** The pacing is effective in building tension. The initial "slow climb" and "cautious ascent" create a sense of anticipation. The transition to "suddenly, the whispers turned to roars" is abrupt and impactful, mirroring the speed of the stock's rise. The conclusion solidifies the idea of rapid, significant growth.
    
    **Suggestions for Improvement:**
    
    1.  **Show, Don't Just Tell, the "Shift":** While the story lists the contributing factors (engine release, indie surge, non-gaming adoption), it could be more impactful by briefly *showing* one of these factors in action. For example, a sentence describing a developer's excitement about the new engine, or a brief mention of a stunning architectural visualization created with Unity, could add more texture and demonstrate the "buzz" more concretely.
    2.  **Humanize the Shareholder Experience:** The story mentions "shareholders who had patiently held watched their investments multiply." To make this more engaging, consider adding a brief, specific anecdote from a hypothetical shareholder's perspective. This could be as simple as "One such shareholder, Sarah, who had bought in years ago, found herself staring in disbelief at her brokerage account, a quiet smile spreading across her face." This would provide an emotional anchor to the financial success.
    3.  **Vary the Metaphors:** The story uses "slow climb," "cautious ascent," "meteoric rise," "rocket launch," and "vertical line, pointing straight to the moon." While effective, the repetition of similar upward trajectory metaphors can become a little one-dimensional. Perhaps introducing a metaphor that speaks to the *impact* of Unity's success, rather than just its growth rate, could add depth. For example, instead of just "industry titan," one could say "Unity was no longer just a tool; it was becoming the bedrock of a new digital landscape."
    RefinerAgent > The ticker flashed U, then N, then I, then T, then Y. For two years, it had been a slow climb, a cautious ascent for Unity Software. Analysts whispered about potential, but the market remained tepid. Then, something shifted. It wasn't a single event, but a confluence of factors: a groundbreaking new engine release that had developers buzzing with excitement over its enhanced capabilities, a surge in indie game development with a dozen groundbreaking titles emerging from small studios powered by Unity, and a growing adoption of the platform for non-gaming applications like stunning architectural visualizations and sleek automotive designs.
    
    Suddenly, the whispers turned to roars. The stock price, once a steady incline, began a meteoric rise. It was a rocket launch, fueled by investor confidence and a palpable buzz. Shareholders who had patiently held watched their investments multiply. One such shareholder, Sarah, who had bought in years ago, found herself staring in disbelief at her brokerage account, a quiet smile spreading across her face. New investors scrambled to get in on the ground floor of what was clearly becoming an industry titan. Unity was no longer just a tool; it was becoming the bedrock of a new digital landscape. The chart, once a gentle curve, was now a vertical line, pointing straight to the moon.
    CriticAgent > APPROVED


    WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.


You've now implemented a loop agent, creating a sophisticated system that can iteratively review and improve its own output. This is a key pattern for ensuring high-quality results.

You now have a complete toolkit of workflow patterns. Let's put it all together and review how to choose the right one for your use case.

--- 
### Section 6
## Summary - Choosing the Right Pattern

### Decision Tree: Which Workflow Pattern?

<!--
```mermaid
graph TD
    A{"What kind of workflow do you need?"} -- > B["Fixed Pipeline<br>(A → B → C)"];
    A -- > C["Concurrent Tasks<br>(Run A, B, C all at once)"];
    A -- > D["Iterative Refinement<br>(A ⇆ B)"];
    A -- > E["Dynamic Decisions<br>(Let the LLM decide what to do)"];

    B -- > B_S["Use <b>SequentialAgent</b>"];
    C -- > C_S["Use <b>ParallelAgent</b>"];
    D -- > D_S["Use <b>LoopAgent</b>"];
    E -- > E_S["Use <b>LLM Orchestrator</b><br>(Agent with other agents as tools)"];

    style B_S fill:#f9f,stroke:#333,stroke-width:2px
    style C_S fill:#ccf,stroke:#333,stroke-width:2px
    style D_S fill:#cff,stroke:#333,stroke-width:2px
    style E_S fill:#cfc,stroke:#333,stroke-width:2px
```
-->

<img width="1000" src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day1/agent-decision-tree.png" alt="Agent Decision Tree" />

### Quick Reference Table

| Pattern | When to Use | Example | Key Feature |
|---------|-------------|---------|-------------|
| **LLM-based (sub_agents)** | Dynamic orchestration needed | Research + Summarize | LLM decides what to call |
| **Sequential** | Order matters, linear pipeline | Outline → Write → Edit | Deterministic order |
| **Parallel** | Independent tasks, speed matters | Multi-topic research | Concurrent execution |
| **Loop** | Iterative improvement needed | Writer + Critic refinement | Repeated cycles |

---

## ✅ Congratulations! You're Now an Agent Orchestrator

In this notebook, you made the leap from a single agent to a **multi-agent system**.

You saw **why** a team of specialists is easier to build and debug than one "do-it-all" agent. Most importantly, you learned how to be the **director** of that team.

You used `SequentialAgent`, `ParallelAgent`, and `LoopAgent` to create deterministic workflows, and you even used an LLM as a 'manager' to make dynamic decisions. You also mastered the "plumbing" by using `output_key` to pass state between agents and make them collaborative.

**ℹ️ Note: No submission required!**

This notebook is for your hands-on practice and learning only. You **do not** need to submit it anywhere to complete the course.

### 📚 Learn More

Refer to the following documentation to learn more:

- [Agents in ADK](https://google.github.io/adk-docs/agents/)
- [Sequential Agents in ADK](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/)
- [Parallel Agents in ADK](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/)
- [Loop Agents in ADK](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/)
- [Custom Agents in ADK](https://google.github.io/adk-docs/agents/custom-agents/)

### 🎯 Next Steps

Ready for the next challenge? Stay tuned for Day 2 notebooks where we'll learn how to create **Custom Functions, use MCP Tools** and manage **Long-Running operations!**

---

| Authors |
| --- |
| [Kristopher Overholt](https://www.linkedin.com/in/koverholt) |
