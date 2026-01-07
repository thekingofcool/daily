---
layout: post
title: Day 3a - Agent Sessions
date: 2025-11-12
author: thekingofcool
description: ""
categories: archive
---

# 🚀 Memory Management - Part 1 - Sessions

**Welcome to Day 3 of the Kaggle 5-day Agents course!**

In this notebook, you'll learn:

- ✅ What sessions are and how to use them in your agent
- ✅ How to build *stateful* agents with sessions and events
- ✅ How to persist sessions in a database
- ✅ Context management practices such as context compaction
- ✅ Best practices for sharing session State

## ⚙️ Section 1: Setup

### 1.1: Install dependencies

The Kaggle Notebooks environment includes a pre-installed version of the [google-adk](https://google.github.io/adk-docs/) library for Python and its required dependencies, so you don't need to install additional packages in this notebook.

To install and use ADK in your own Python development environment outside of this course, you can do so by running:

```
pip install google-adk
```

### 1.2: Configure your Gemini API Key

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
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )
```

    ✅ Gemini API key setup complete.


### 1.3: Import ADK components

Now, import the specific components you'll need from the Agent Development Kit and the Generative AI library. This keeps your code organized and ensures we have access to the necessary building blocks.


```python
from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

print("✅ ADK components imported successfully.")
```

    ✅ ADK components imported successfully.


### 1.4: Helper functions

Helper function that manages a complete conversation session, handling session
creation/retrieval, query processing, and response streaming. It supports
both single queries and multiple queries in sequence.

Example:

```
>>> await run_session(runner, "What is the capital of France?", "geography-session")
>>> await run_session(runner, ["Hello!", "What's my name?"], "user-intro-session")
```


```python
# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("✅ Helper functions defined.")
```

    ✅ Helper functions defined.


### 1.5: Configure Retry Options

When working with LLMs, you may encounter transient errors like rate limits or temporary service unavailability. Retry options automatically handle these failures by retrying the request with exponential backoff.


```python
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
```

---
## 🤹 Section 2: Session Management

### 2.1 The Problem

At their core, Large Language Models are **inherently stateless**. Their awareness is confined to the information you provide in a single API call. This means an agent without proper context management will react to the current prompt without considering any previous history.

**❓ Why does this matter?** Imagine trying to have a meaningful conversation with someone who forgets everything you've said after each sentence. That's the challenge we face with raw LLMs!

In ADK, we use `Sessions` for **short term memory management** and `Memory` for **long term memory.** In the next notebook, you'll focus on `Memory`.

### 2.2 What is a Session?

#### **📦 Session**

A session is a container for conversations. It encapsulates the conversation history in a chronological manner and also records all tool interactions and responses for a **single, continuous conversation**. A session is tied to a user and agent; it is not shared with other users. Similarly, a session history for an Agent is not shared with other Agents.

In ADK, a **Session** is comprised of two key components `Events` and `State`:

**📝 Session.Events**:

> While a session is a container for conversations, `Events` are the building blocks of a conversation.
>
> Example of Events:
> - *User Input*: A message from the user (text, audio, image, etc.)
> - *Agent Response*: The agent's reply to the user
> - *Tool Call*: The agent's decision to use an external tool or API
> - *Tool Output*: The data returned from a tool call, which the agent uses to continue its reasoning
    

**{} Session.State**:

> `session.state` is the Agent's scratchpad, where it stores and updates dynamic details needed during the conversation. Think of it as a global `{key, value}` pair storage which is available to all subagents and tools.

<img src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day3/session-state-and-events.png" width="320" alt="Session state and events">

<!-- ```mermaid
graph TD
    subgraph A["Agentic Application"];
        subgraph U["User"]
            subgraph S1["Session"]
                D1["Session.Events"]
                D2["Session.State"]
            end
        end
    end
``` -->

### 2.3 How to manage sessions?

An agentic application can have multiple users and each user may have multiple sessions with the application.
To manage these sessions and events, ADK offers a **Session Manager** and **Runner**.

1. **`SessionService`**: The storage layer
   - Manages creation, storage, and retrieval of session data
   - Different implementations for different needs (memory, database, cloud)

2. **`Runner`**: The orchestration layer
   - Manages the flow of information between user and agent
   - Automatically maintains conversation history
   - Handles the Context Engineering behind the scenes

Think of it like this:

- **Session** = A notebook 📓
- **Events** = Individual entries in a single page 📝
- **SessionService** = The filing cabinet storing notebooks 🗄️
- **Runner** = The assistant managing the conversation 🤖

### 2.4 Implementing Our First Stateful Agent

Let's build our first stateful agent, that can remember and have constructive conversations. 

ADK offers different types of sessions suitable for different needs. As a start, we'll start with a simple Session Management option (`InMemorySessionService`):


```python
APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"


# Step 1: Create the LLM Agent
root_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="text_chat_bot",
    description="A text chatbot",  # Description of the agent's purpose
)

# Step 2: Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

print("✅ Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")
```

    ✅ Stateful agent initialized!
       - Application: default
       - User: default
       - Using: InMemorySessionService


### 2.5 Testing Our Stateful Agent

Now let's see the magic of sessions in action!


```python
# Run a conversation with two queries in the same session
# Notice: Both queries are part of the SAME session, so context is maintained
await run_session(
    runner,
    [
        "Hi, I am Sam! What is the capital of United States?",
        "Hello! What is my name?",  # This time, the agent should remember!
    ],
    "stateful-agentic-session",
)
```

    
     ### Session: stateful-agentic-session
    
    User > Hi, I am Sam! What is the capital of United States?
    gemini-2.5-flash-lite >  Hi Sam! The capital of the United States is Washington, D.C.
    
    User > Hello! What is my name?
    gemini-2.5-flash-lite >  Your name is Sam.


🎉 **Success!** The agent remembered your name because both queries were part of the same session. The Runner automatically maintained the conversation history.

But there's a catch: `InMemorySessionService` is temporary. **Once the application stops, all conversation history is lost.** 


### 🛑 (Optional) 2.6 Testing Agent's forgetfulness

> To verify that the agent forgets the conversation, **restart the kernel**. Then, **run ALL the previous cells in this notebook EXCEPT the `run_session` in 2.5.**
> 
> Now run the cell below. You'll see that the agent doesn't remember anything from the previous conversation.


```python
# Run this cell after restarting the kernel. All this history will be gone...
await run_session(
    runner,
    ["What did I ask you about earlier?", "And remind me, what's my name?"],
    "stateful-agentic-session",
)  # Note, we are using same session name
```

    
     ### Session: stateful-agentic-session
    
    User > What did I ask you about earlier?
    gemini-2.5-flash-lite >  I do not have access to past conversations. Therefore, I cannot tell you what you asked me about earlier.
    
    User > And remind me, what's my name?
    gemini-2.5-flash-lite >  I do not have access to past conversations. Therefore, I cannot tell you your name.


### The Problem

Session information is not persistent (i.e., meaningful conversations are lost). While this is advantageous in testing environments, **in the real world, a user should be able to refer from past and resume conversations.** To achieve this, we must persist information. 

---
## 📈 Section 3: Persistent Sessions with `DatabaseSessionService`

While `InMemorySessionService` is great for prototyping, real-world applications need conversations to survive restarts, crashes, and deployments. Let's level up to persistent storage!

### 3.1 Choosing the Right SessionService

ADK provides different SessionService implementations for different needs:

| Service | Use Case | Persistence | Best For |
|---------|----------|-------------|----------|
| **InMemorySessionService** | Development & Testing | ❌ Lost on restart | Quick prototypes |
| **DatabaseSessionService** | Self-managed apps | ✅ Survives restarts | Small to medium apps |
| **Agent Engine Sessions** | Production on GCP | ✅ Fully managed | Enterprise scale |


### 3.2 Implementing Persistent Sessions

Let's upgrade to `DatabaseSessionService` using SQLite. This gives us persistence without needing a separate database server for this demo.

Let's create a `chatbot_agent` capable of having a conversation with the user.


```python
# Step 1: Create the same agent (notice we use LlmAgent this time)
chatbot_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="text_chat_bot",
    description="A text chatbot with persistent memory",
)

# Step 2: Switch to DatabaseSessionService
# SQLite database will be created automatically
db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Step 3: Create a new runner with persistent storage
runner = Runner(agent=chatbot_agent, app_name=APP_NAME, session_service=session_service)

print("✅ Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")
```

    ✅ Upgraded to persistent sessions!
       - Database: my_agent_data.db
       - Sessions will survive restarts!


### 3.3 Test Run 1: Verifying Persistence

In this first test run, we'll start a new conversation with the session ID `test-db-session-01`. We will first introduce our name as 'Sam' and then ask a question. In the second turn, we will ask the agent for our name.

Since we are using `DatabaseSessionService`, the agent should remember the name.

After the conversation, we'll inspect the `my_agent_data.db` SQLite database directly to see how the conversation `events` (the user queries and model responses) are stored.



```python
await run_session(
    runner,
    ["Hi, I am Sam! What is the capital of the United States?", "Hello! What is my name?"],
    "test-db-session-01",
)
```

    
     ### Session: test-db-session-01
    
    User > Hi, I am Sam! What is the capital of the United States?
    gemini-2.5-flash-lite >  Hi Sam! The capital of the United States is Washington, D.C.
    
    User > Hello! What is my name?
    gemini-2.5-flash-lite >  Your name is Sam.


### 🛑 (Optional) 3.4 Test Run 2: Resuming a Conversation

> ‼️ Now, let's repeat the test again, but this time, **let's stop this Kaggle Notebook's kernel and restart it again.**
>
> 1. Run all the previous cells in the notebook, **EXCEPT** the previous Section 3.3 (`run_session` cell).
>
> 2. Now, run the below cell with the **same session ID** (`test-db-session-01`).

We will ask a new question and then ask for our name again. **Because the session is loaded from the database, the agent should still remember** that our name is 'Sam' from the first test run. This demonstrates the power of persistent sessions.



```python
await run_session(
    runner,
    ["What is the capital of India?", "Hello! What is my name?"],
    "test-db-session-01",
)
```

    
     ### Session: test-db-session-01
    
    User > What is the capital of India?
    gemini-2.5-flash-lite >  The capital of India is New Delhi.
    
    User > Hello! What is my name?
    gemini-2.5-flash-lite >  Your name is Sam.


### 3.5 Let's verify that the session data is isolated

As mentioned earlier, a session is private conversation between an Agent and a User (i.e., two sessions do not share information). Let's run our `run_session` with a different session name `test-db-session-02` to confirm this.



```python
await run_session(
    runner, ["Hello! What is my name?"], "test-db-session-02"
)  # Note, we are using new session name
```

    
     ### Session: test-db-session-02
    
    User > Hello! What is my name?
    gemini-2.5-flash-lite >  I do not have access to your personal information, including your name. As a text chatbot, I do not store any user data or personal details. Therefore, I cannot tell you what your name is.


### 3.6 How are the events stored in the Database?

Since we are using a sqlite DB to store information, let's have a quick peek to see how information is stored.


```python
import sqlite3

def check_data_in_db():
    with sqlite3.connect("my_agent_data.db") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)


check_data_in_db()
```

    ['app_name', 'session_id', 'author', 'content']
    ('default', 'test-db-session-01', 'user', '{"parts": [{"text": "Hi, I am Sam! What is the capital of the United States?"}], "role": "user"}')
    ('default', 'test-db-session-01', 'text_chat_bot', '{"parts": [{"text": "Hi Sam! The capital of the United States is Washington, D.C."}], "role": "model"}')
    ('default', 'test-db-session-01', 'user', '{"parts": [{"text": "Hello! What is my name?"}], "role": "user"}')
    ('default', 'test-db-session-01', 'text_chat_bot', '{"parts": [{"text": "Your name is Sam."}], "role": "model"}')
    ('default', 'test-db-session-01', 'user', '{"parts": [{"text": "What is the capital of India?"}], "role": "user"}')
    ('default', 'test-db-session-01', 'text_chat_bot', '{"parts": [{"text": "The capital of India is New Delhi."}], "role": "model"}')
    ('default', 'test-db-session-01', 'user', '{"parts": [{"text": "Hello! What is my name?"}], "role": "user"}')
    ('default', 'test-db-session-01', 'text_chat_bot', '{"parts": [{"text": "Your name is Sam."}], "role": "model"}')
    ('default', 'test-db-session-02', 'user', '{"parts": [{"text": "Hello! What is my name?"}], "role": "user"}')
    ('default', 'test-db-session-02', 'text_chat_bot', '{"parts": [{"text": "I do not have access to your personal information, including your name. As a text chatbot, I do not store any user data or personal details. Therefore, I cannot tell you what your name is."}], "role": "model"}')


---
## ⏳ Section 4: Context Compaction

As you can see, all the events are stored in full in the session Database, and this quickly adds up. For a long, complex task, this list of events can become very large, leading to slower performance and higher costs.

But what if we could automatically summarize the past? Let's use ADK's **Context Compaction** feature to see **how to automatically reduce the context that's stored in the Session.**

<img src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day3/context-compaction.png" width="1400" alt="Context compaction">

### 4.1 Create an App for the agent

To enable this feature, let's use the same `chatbot_agent` we created in Section 3.2. 

The first step is to create an object called `App`. We'll give it a name and pass in our chatbot_agent. 

We'll also create a new config to do the Context Compaction. This **`EventsCompactionConfig`** defines two key variables:

- **compaction_interval**: Asks the Runner to compact the history after every `n` conversations
- **overlap_size**: Defines the number of previous conversations to retain for overlap

We'll then provide this app to the Runner.



```python
# Re-define our app with Events Compaction enabled
research_app_compacting = App(
    name="research_app_compacting",
    root_agent=chatbot_agent,
    # This is the new part!
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
)

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Create a new runner for our upgraded app
research_runner_compacting = Runner(
    app=research_app_compacting, session_service=session_service
)


print("✅ Research App upgraded with Events Compaction!")
```

    ✅ Research App upgraded with Events Compaction!


    /tmp/ipykernel_131/3773147741.py:6: UserWarning: [EXPERIMENTAL] EventsCompactionConfig: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
      events_compaction_config=EventsCompactionConfig(


### 4.2 Running the Demo

Now, let's have a conversation that is long enough to trigger the compaction. When you run the cell below, the output will look like a normal conversation. However, because we configured our `App`, a compaction process will run silently in the background after the 3rd invocation.

In the next step, we'll prove that it happened.


```python
# Turn 1
await run_session(
    research_runner_compacting,
    "What is the latest news about AI in healthcare?",
    "compaction_demo",
)

# Turn 2
await run_session(
    research_runner_compacting,
    "Are there any new developments in drug discovery?",
    "compaction_demo",
)

# Turn 3 - Compaction should trigger after this turn!
await run_session(
    research_runner_compacting,
    "Tell me more about the second development you found.",
    "compaction_demo",
)

# Turn 4
await run_session(
    research_runner_compacting,
    "Who are the main companies involved in that?",
    "compaction_demo",
)
```

    
     ### Session: compaction_demo
    
    User > What is the latest news about AI in healthcare?
    gemini-2.5-flash-lite >  The field of AI in healthcare is incredibly dynamic, with new developments emerging almost daily. To give you the *latest* news, I'll need to access more up-to-date information.
    
    However, I can tell you about some of the *major ongoing trends and recent breakthroughs* that are consistently making headlines:
    
    **Key Areas of Advancement and News:**
    
    *   **Drug Discovery and Development:**
        *   **Accelerated Research:** AI is dramatically speeding up the identification of potential drug candidates, predicting their efficacy, and understanding their mechanisms of action. Companies are using AI to analyze vast datasets of biological information and chemical compounds.
        *   **Personalized Medicine:** AI is helping to tailor drug treatments to individual patients based on their genetic makeup, lifestyle, and disease characteristics.
        *   **De Novo Drug Design:** AI models are now capable of *designing* entirely new molecules with specific therapeutic properties, rather than just screening existing ones.
    
    *   **Diagnostics and Imaging:**
        *   **Radiology and Pathology:** AI algorithms are becoming increasingly sophisticated at analyzing medical images (X-rays, CT scans, MRIs, pathology slides) to detect subtle anomalies that humans might miss, aiding in early disease detection for conditions like cancer, diabetic retinopathy, and cardiovascular disease.
        *   **Predictive Diagnostics:** AI is being used to predict a patient's risk of developing certain conditions based on their electronic health records (EHRs) and other data, allowing for proactive interventions.
    
    *   **Clinical Workflow and Operational Efficiency:**
        *   **Administrative Tasks:** AI is automating tedious administrative tasks like scheduling, billing, and prior authorization requests, freeing up healthcare professionals' time.
        *   **Clinical Decision Support:** AI-powered tools are providing clinicians with real-time insights and recommendations to support their decision-making during patient care.
        *   **Patient Monitoring:** Remote patient monitoring using AI can detect early signs of deterioration and alert caregivers, improving outcomes for chronic conditions.
    
    *   **Mental Health:**
        *   **AI-powered Therapy and Support:** Chatbots and virtual assistants are providing accessible mental health support, offering cognitive behavioral therapy (CBT) exercises, and acting as initial screening tools.
        *   **Analyzing Behavioral Patterns:** AI is being used to analyze speech patterns, text messages, and social media activity to detect early signs of mental health issues.
    
    *   **Generative AI and Large Language Models (LLMs):**
        *   **Summarization and Documentation:** LLMs are being trained to summarize patient encounters, draft clinical notes, and generate reports, reducing physician burnout.
        *   **Patient Communication:** AI is helping to create more personalized and informative patient communication materials.
        *   **Medical Education and Training:** LLMs can create realistic patient scenarios for training medical students and professionals.
    
    **Recent Themes and Buzzwords You Might See:**
    
    *   **Explainable AI (XAI):** A growing focus on making AI decisions transparent and understandable to clinicians, crucial for trust and adoption.
    *   **Federated Learning:** Training AI models across multiple decentralized datasets without sharing the raw data, crucial for patient privacy.
    *   **Real-world Evidence (RWE) and Real-world Data (RWD):** AI is essential for analyzing these large, complex datasets to understand drug effectiveness and patient outcomes outside of controlled trials.
    *   **Regulatory Approval:** A significant ongoing development is the increasing number of AI-powered medical devices and software receiving regulatory approval (e.g., FDA clearance).
    
    **To get the *absolute latest* news, I would recommend checking these sources:**
    
    *   **Reputable Healthcare Technology News Outlets:** Fierce Biotech, STAT News, MobiHealthNews, Healthcare IT News, MedTech Dive.
    *   **Major AI Research Conferences:** NeurIPS, ICML, ICLR (though these can be highly technical).
    *   **Company Press Releases:** Leading AI healthcare companies (e.g., Google Health, Microsoft Healthcare, IBM Watson Health, NVIDIA, Tempus, Viz.ai, Paige.AI) often announce breakthroughs.
    *   **Medical Journals:** Publications in journals like Nature Medicine, The Lancet, JAMA, NEJM often feature AI research.
    
    If you have a specific area within AI and healthcare you're interested in, I can try to find more targeted information!
    
     ### Session: compaction_demo
    
    User > Are there any new developments in drug discovery?
    gemini-2.5-flash-lite >  Yes, there are constant and exciting new developments in drug discovery powered by AI! It's one of the most rapidly evolving areas. Here are some of the key new developments and trends:
    
    **1. Generative AI for Novel Molecule Design (De Novo Drug Design):**
    *   **What's New:** Instead of screening existing libraries of compounds, generative AI models (like GANs, VAEs, and transformer-based models) are now capable of *designing entirely new molecules* from scratch, optimized for specific target properties (e.g., binding affinity, low toxicity, good ADME properties - Absorption, Distribution, Metabolism, Excretion).
    *   **Impact:** This is a game-changer, as it vastly expands the chemical space that can be explored, potentially leading to drugs for previously "undruggable" targets.
    *   **Examples:** Companies are using these models to design novel antibiotics, cancer therapeutics, and drugs for rare diseases.
    
    **2. Advanced AI for Target Identification and Validation:**
    *   **What's New:** AI is becoming more adept at analyzing complex biological data (genomics, proteomics, transcriptomics, patient data) to identify novel disease targets and understand their role in disease pathways. This goes beyond simple gene association to predicting protein interactions and cellular mechanisms.
    *   **Impact:** Leads to more precise and effective drug development by focusing on the most relevant biological drivers of disease.
    
    **3. AI-Powered Prediction of Drug Properties and Efficacy:**
    *   **What's New:** Sophisticated AI models can now predict with higher accuracy:
        *   **Binding Affinity:** How strongly a drug candidate will bind to its target.
        *   **Toxicity:** Potential side effects and safety concerns early in the process.
        *   **Pharmacokinetics (PK) and Pharmacodynamics (PD):** How the drug will be absorbed, distributed, metabolized, and excreted by the body, and its effect over time.
        *   **Clinical Trial Success Likelihood:** By analyzing preclinical data and historical trial outcomes.
    *   **Impact:** Reduces the number of failed drug candidates, saving immense time and cost in the preclinical and clinical development phases.
    
    **4. AI for Repurposing Existing Drugs:**
    *   **What's New:** AI can analyze vast databases of drug information, disease mechanisms, and patient data to identify existing approved drugs that could be effective for new indications.
    *   **Impact:** Offers a faster and less risky pathway to new treatments, as the safety profiles of these drugs are already well-established.
    
    **5. Enhanced Virtual Screening and Lead Optimization:**
    *   **What's New:** AI algorithms can process and rank billions of potential drug compounds against a target much faster than traditional methods. They also help in optimizing the properties of promising "lead" compounds.
    *   **Impact:** Accelerates the initial stages of drug discovery, narrowing down the search space significantly.
    
    **6. AI in Clinical Trial Design and Patient Stratification:**
    *   **What's New:** While not strictly "discovery," AI is optimizing the *next steps* after discovery by helping design more efficient clinical trials and identifying patient subgroups most likely to respond to a new drug, leading to faster trial completion and higher success rates.
    *   **Impact:** Crucial for bringing new drugs to market more quickly and effectively.
    
    **Examples of Companies and Research Leading the Charge:**
    
    *   **Recursion Pharmaceuticals:** Uses AI and automation to discover novel therapeutic targets and drugs.
    *   **Exscientia:** Focuses on AI-driven drug design, having brought several AI-designed molecules into clinical trials.
    *   **Insilico Medicine:** Known for its end-to-end AI platform for drug discovery, including generative AI for molecule design.
    *   **Schrödinger:** A leader in computational chemistry, integrating AI into its physics-based modeling for drug discovery.
    *   **NVIDIA:** Provides powerful GPU hardware and AI platforms that are essential for many of these computational drug discovery efforts.
    
    **Challenges Remain:**
    
    Despite the incredible progress, challenges persist, including:
    
    *   **Data Quality and Accessibility:** High-quality, well-annotated biological and chemical data is crucial for training effective AI models.
    *   **Interpretability:** Understanding *why* an AI model makes a certain prediction is still an area of active research.
    *   **Experimental Validation:** AI predictions still need rigorous experimental validation, which remains a bottleneck.
    *   **Regulatory Hurdles:** Ensuring AI-designed drugs meet stringent regulatory standards.
    
    In summary, AI is not just incrementally improving drug discovery; it's fundamentally transforming the process, making it faster, more efficient, and capable of exploring entirely new therapeutic possibilities.
    
     ### Session: compaction_demo
    
    User > Tell me more about the second development you found.
    gemini-2.5-flash-lite >  You're likely referring to the **second development I mentioned: Advanced AI for Target Identification and Validation.**
    
    This area is incredibly significant because it addresses a fundamental bottleneck in drug discovery: **finding the *right* biological target** to address a disease. For a long time, target identification was a slow, laborious process involving extensive manual research, hypothesis generation, and experimental validation. AI is revolutionizing this.
    
    Here's a deeper dive into what this development entails and why it's important:
    
    **What "Target Identification and Validation" Means:**
    
    *   **Target Identification:** This is the process of finding a specific biological molecule (usually a protein, but it could also be a gene, RNA, or even a pathway) that plays a critical role in a disease. If you can modulate the activity of this target (e.g., inhibit an overactive enzyme, activate a deficient receptor), you might be able to treat the disease.
    *   **Target Validation:** Once a potential target is identified, it needs to be validated. This means gathering strong evidence that the target is indeed crucial for the disease and that modulating it will have a therapeutic benefit without unacceptable side effects.
    
    **How AI is Advancing This:**
    
    1.  **Analyzing Massive, Multi-Omic Datasets:**
        *   **What AI does:** AI algorithms can sift through enormous datasets that combine different biological "omics" (genomics, proteomics, transcriptomics, metabolomics) from healthy individuals and patients with specific diseases. They can also integrate data from electronic health records (EHRs), scientific literature, and clinical trial results.
        *   **Why it's powerful:** These datasets are too complex for humans to analyze manually. AI can identify subtle patterns, correlations, and causal relationships that indicate a gene or protein is dysregulated in a disease state. For instance, it might spot that a particular protein is consistently overexpressed in cancer cells but not in healthy cells across thousands of samples.
    
    2.  **Uncovering Novel Disease Pathways:**
        *   **What AI does:** AI can map out complex biological networks and pathways. It can identify how different genes, proteins, and molecules interact and how these interactions go awry in disease. This helps identify not just individual targets but also critical nodes within these dysregulated pathways.
        *   **Why it's powerful:** Diseases are rarely caused by a single faulty component. Understanding the entire pathway provides a more holistic view and can reveal "druggable" points that might have been overlooked.
    
    3.  **Predicting Target "Druggability":**
        *   **What AI does:** Beyond just identifying a target, AI models can analyze the target's structure and properties to predict whether it's likely to be "druggable" – meaning, can a molecule be designed or found that can effectively bind to and modulate its activity?
        *   **Why it's powerful:** This saves resources by prioritizing targets that have a higher probability of yielding a successful drug, rather than pursuing targets that are biologically relevant but chemically challenging.
    
    4.  **Leveraging Natural Language Processing (NLP) for Literature Mining:**
        *   **What AI does:** NLP-powered AI can rapidly scan and extract information from millions of scientific publications, patents, and conference abstracts. It can connect disparate pieces of information to form new hypotheses about potential targets.
        *   **Why it's powerful:** It helps researchers stay on top of the ever-growing body of scientific literature and discover connections that might not be obvious from reading individual papers.
    
    5.  **Integrating Real-World Data (RWD) and Real-World Evidence (RWE):**
        *   **What AI does:** AI can analyze RWD from EHRs, insurance claims, and patient registries to observe how diseases progress and how different interventions (even off-label uses of existing drugs) affect patients in real-world settings. This can provide strong validation for targets.
        *   **Why it's powerful:** This mirrors how a drug might perform in the diverse patient population encountered in clinical practice, offering a different and often complementary perspective to controlled lab experiments.
    
    **Impact and Significance:**
    
    *   **Accelerated Discovery:** AI drastically reduces the time it takes to go from understanding a disease to identifying a promising target.
    *   **Reduced Risk and Cost:** By focusing on more validated and druggable targets, AI helps de-risk the early stages of drug development, potentially saving billions of dollars and years of research.
    *   **Tackling Difficult Diseases:** AI can help identify targets for complex diseases like Alzheimer's, Parkinson's, and various autoimmune disorders, where understanding the underlying biology has been challenging.
    *   **Personalized Medicine:** AI can help identify patient subgroups that might respond to therapies targeting specific molecules, paving the way for more personalized treatments.
    
    Companies like **Deep Genomics**, **BenevolentAI**, and **Cerebras Systems** (through its AI hardware enabling large-scale biological simulations) are actively involved in advancing AI for target identification and validation. It's a foundational step that enables many of the other AI-driven innovations in drug discovery.
    
     ### Session: compaction_demo
    
    User > Who are the main companies involved in that?
    gemini-2.5-flash-lite >  You're asking about the companies at the forefront of using AI for **Target Identification and Validation**. This is a crucial area, and many companies, from large biotechs to specialized AI drug discovery firms, are involved.
    
    Here are some of the main players, categorized by their approach:
    
    **1. AI-Native Drug Discovery Platforms (often with a strong focus on Target ID):**
    
    These companies are built from the ground up with AI as their core engine for identifying and validating targets.
    
    *   **BenevolentAI:** One of the pioneers. They use their AI platform to analyze vast amounts of biomedical data (scientific literature, patents, clinical trial data, genomic data) to identify novel disease targets and potential drug candidates. They have both internal R&D programs and collaborations.
    *   **Recursion Pharmaceuticals:** While they do drug discovery across the board, their platform is heavily reliant on generating and analyzing biological data (e.g., from cellular imaging) at scale. Their AI is used to map cellular biology and identify how different genes/targets are involved in disease states.
    *   **Insilico Medicine:** Known for its end-to-end AI platform, they use AI for target discovery, generative molecule design, and even predicting clinical trial outcomes. Their target ID capabilities are integrated into their workflow.
    *   **Atomwise:** Primarily known for its AI for small molecule drug discovery (virtual screening), they also leverage their AI to understand target-drug interactions, which implicitly aids in target validation by assessing the potential for interaction.
    *   **Exscientia:** Another leader in AI-driven drug design. While they excel at *designing* molecules, their underlying AI models are trained on extensive biological data, enabling them to identify and prioritize targets that are amenable to their design approach.
    
    **2. Large Pharmaceutical Companies with Dedicated AI/Data Science Units:**
    
    Major pharma companies are heavily investing in AI, building internal capabilities, and forming partnerships.
    
    *   **Pfizer:** Has been investing significantly in AI and data science, including for target identification and early-stage research.
    *   **Novartis:** Known for its strong data science focus, they utilize AI for various stages of drug discovery, including target identification and validation.
    *   **Roche/Genentech:** Employs AI and machine learning extensively in its research and development, including exploring new biological targets for oncology, immunology, and neuroscience.
    *   **Merck (MSD):** Has various initiatives and partnerships focused on leveraging AI for drug discovery, which would include target ID.
    *   **AstraZeneca:** Actively uses AI and machine learning across its R&D pipeline, from target identification to clinical trial optimization.
    
    **3. Tech Companies Enabling Biological Data Analysis:**
    
    While not directly doing drug discovery, these companies provide the foundational tools and platforms.
    
    *   **NVIDIA:** Their GPUs are essential for training complex AI models used in biological simulations and data analysis. They also develop specialized software platforms (like Clara Discovery) that facilitate AI in drug discovery, including target ID.
    *   **Google (Google Health/DeepMind):** DeepMind's work on protein folding (AlphaFold) is revolutionary for understanding protein structures, which is critical for target identification and validation. Google's broader AI capabilities are also applied to biological data.
    *   **Microsoft:** Offers cloud computing services (Azure) and AI tools that are used by many biotech and pharma companies for their R&D, including target discovery.
    
    **4. Specialized Companies Focusing on Specific Aspects:**
    
    *   **Deep Genomics:** Uses AI to identify genetic causes of disease and then identify therapeutic targets based on that understanding.
    *   **Tempus:** While broader in scope (oncology data platform), their ability to analyze vast amounts of clinical and molecular data is invaluable for identifying and validating targets, particularly in cancer.
    
    **How They Collaborate:**
    
    It's also important to note that the landscape is characterized by:
    
    *   **Partnerships:** AI-native companies often partner with large pharma to leverage their biological expertise and clinical trial infrastructure.
    *   **Acquisitions:** Large pharma companies sometimes acquire promising AI drug discovery startups.
    *   **Consortia and Open Science Initiatives:** Efforts to share data and develop standardized AI tools also play a role.
    
    The field is very dynamic, with new companies emerging and established players expanding their AI efforts constantly. The common thread is the application of advanced computational power and machine learning algorithms to make sense of complex biological data to find and validate the right targets for new medicines.


### 4.3 Verifying Compaction in the Session History

The conversation above looks normal, but the history has been changed behind the scenes. How can we prove it?

We can inspect the `events` list from our session. The compaction process **doesn't delete old events; it replaces them with a single, new `Event` that contains the summary.** Let's find it.


```python
# Get the final session state
final_session = await session_service.get_session(
    app_name=research_runner_compacting.app_name,
    user_id=USER_ID,
    session_id="compaction_demo",
)

print("--- Searching for Compaction Summary Event ---")
found_summary = False
for event in final_session.events:
    # Compaction events have a 'compaction' attribute
    if event.actions and event.actions.compaction:
        print("\n✅ SUCCESS! Found the Compaction Event:")
        print(f"  Author: {event.author}")
        print(f"\n Compacted information: {event}")
        found_summary = True
        break

if not found_summary:
    print(
        "\n❌ No compaction event found. Try increasing the number of turns in the demo."
    )
```

    --- Searching for Compaction Summary Event ---
    
    ✅ SUCCESS! Found the Compaction Event:
      Author: user
    
     Compacted information: model_version=None content=None grounding_metadata=None partial=None turn_complete=None finish_reason=None error_code=None error_message=None interrupted=None custom_metadata=None usage_metadata=None live_session_resumption_update=None input_transcription=None output_transcription=None avg_logprobs=None logprobs_result=None cache_metadata=None citation_metadata=None invocation_id='6b6e1506-7725-45bb-b78c-e0e6a3ab307f' author='user' actions=EventActions(skip_summarization=None, state_delta={}, artifact_delta={}, transfer_to_agent=None, escalate=None, requested_auth_configs={}, requested_tool_confirmations={}, compaction={'start_timestamp': 1763092997.891181, 'end_timestamp': 1763093004.958794, 'compacted_content': {'parts': [{'function_call': None, 'code_execution_result': None, 'executable_code': None, 'file_data': None, 'function_response': None, 'inline_data': None, 'text': 'The user initiated a conversation asking for the latest news in AI in healthcare. The AI agent provided a comprehensive overview of major trends and recent breakthroughs, categorizing them into drug discovery, diagnostics/imaging, clinical workflow, mental health, and generative AI/LLMs. It also highlighted emerging themes like Explainable AI (XAI) and federated learning, and recommended sources for staying updated.\n\nThe user then specifically inquired about new developments in drug discovery. The AI agent detailed five key advancements:\n1.  **Generative AI for Novel Molecule Design:** AI designing new molecules from scratch.\n2.  **Advanced AI for Target Identification and Validation:** AI identifying and validating biological targets for drugs.\n3.  **AI-Powered Prediction of Drug Properties and Efficacy:** AI predicting binding affinity, toxicity, PK/PD, and clinical trial success.\n4.  **AI for Repurposing Existing Drugs:** AI identifying new uses for approved drugs.\n5.  **Enhanced Virtual Screening and Lead Optimization:** AI accelerating compound screening and lead compound optimization.\nIt also mentioned AI\'s role in clinical trial design and patient stratification.\n\nThe user then requested more information about the second development mentioned (AI for Target Identification and Validation). The AI agent elaborated on this, explaining:\n*   **What Target Identification and Validation entails:** Finding specific biological molecules crucial for a disease and confirming their therapeutic potential.\n*   **How AI advances this:**\n    *   Analyzing multi-omic and EHR data to find dysregulated molecules.\n    *   Uncovering complex disease pathways and critical nodes.\n    *   Predicting the "druggability" of potential targets.\n    *   Using NLP to mine scientific literature for new hypotheses.\n    *   Integrating Real-World Data (RWD) and Real-World Evidence (RWE) for validation.\n*   **The impact and significance:** Accelerated discovery, reduced risk and cost, tackling difficult diseases, and enabling personalized medicine.\n\nThe conversation concluded with the AI agent providing a deeper explanation of AI\'s role in identifying and validating drug targets.\n\n**Key Information and Decisions Made:**\n*   The user\'s initial broad query was narrowed down to AI in drug discovery.\n*   The AI agent successfully provided detailed explanations of advancements in AI for drug discovery, specifically elaborating on target identification and validation.\n\n**Unresolved Questions or Tasks:**\n*   None. The conversation followed a logical progression of the user asking for more detail on a specific topic, which the AI agent provided.', 'thought': None, 'thought_signature': None, 'video_metadata': None}], 'role': 'model'}}, end_of_agent=None, agent_state=None, rewind_before_invocation_id=None) long_running_tool_ids=set() branch=None id='7102ede5-7282-4a63-a84b-7510b381afdc' timestamp=1763093011.64706


### 4.4 What you've accomplished: Automatic Context Management

You just found the proof! The presence of that special summary `Event` in your session's history is the tangible result of the compaction process.

**Let's recap what you just witnessed:**

1.  **Silent Operation**: You ran a standard conversation, and from the outside, nothing seemed different.
2.  **Background Compaction**: Because you configured the `App` with `EventsCompactionConfig`, the ADK `Runner` automatically monitored the conversation length. Once the threshold was met, it triggered the summarization process in the background.
3.  **Verified Result**: By inspecting the session's events, you found the summary that the LLM generated. This summary now replaces the older, more verbose turns in the agent's active context.

**For all future turns in this conversation, the agent will be given this concise summary instead of the full history.** This saves costs, improves performance, and helps the agent stay focused on what's most important.


### 4.5 More Context Engineering options in ADK

#### 👉 Custom Compaction
In this example, we used ADK's default summarizer. For more advanced use cases, you can provide your own by defining a custom `SlidingWindowCompactor` and passing it to the config. This allows you to control the summarization prompt or even use a different, specialized LLM for the task. You can read more about it in the [official documentation](https://google.github.io/adk-docs/context/compaction/).

#### 👉 Context Caching
ADK also provides **Context Caching** to help reduce the token size of the static instructions that are fed to the LLM by caching the request data. Read more about it [here](https://google.github.io/adk-docs/context/caching/).

### The Problem

While we can do Context Compaction and use a database to resume a session, we face new challenges now. In some cases, **we have key information or preferences that we want to share across other sessions.** 

In these scenarios, instead of sharing the entire session history, transferring information from a few key variables can improve the session experience. Let's see how to do it!

---
## 🤝 Section 5: Working with Session State

### 5.1 Creating custom tools for Session state management

Let's explore how to manually manage session state through custom tools. In this example, we'll identify a **transferable characteristic**, like a user's name and their country, and create tools to capture and save it.

**Why This Example?**

The username is a perfect example of information that:

- Is introduced once but referenced multiple times
- Should persist throughout a conversation
- Represents a user-specific characteristic that enhances personalization

Here, for demo purposes, we'll create two tools that can store and retrieve user name and country from the Session State. **Note that all tools have access to the `ToolContext` object.** You don't have to create separate tools for each piece of information you want to share. 


```python
# Define scope levels for state keys (following best practices)
USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")


# This demonstrates how tools can write to session state using tool_context.
# The 'user:' prefix indicates this is user-specific data.
def save_userinfo(
    tool_context: ToolContext, user_name: str, country: str
) -> Dict[str, Any]:
    """
    Tool to record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country

    return {"status": "success"}


# This demonstrates how tools can read from session state.
def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """
    # Read from session state
    user_name = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")

    return {"status": "success", "user_name": user_name, "country": country}


print("✅ Tools created.")
```

    ✅ Tools created.


**Key Concepts:**
- Tools can access `tool_context.state` to read/write session state
- Use descriptive key prefixes (`user:`, `app:`, `temp:`) for organization
- State persists across conversation turns within the same session

### 5.2 Creating an Agent with Session State Tools

Now let's create a new agent that has access to our session state management tools:


```python
# Configuration
APP_NAME = "default"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

# Create an agent with session state tools
root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="text_chat_bot",
    description="""A text chatbot.
    Tools for managing user context:
    * To record username and country when provided use `save_userinfo` tool. 
    * To fetch username and country when required use `retrieve_userinfo` tool.
    """,
    tools=[save_userinfo, retrieve_userinfo],  # Provide the tools to the agent
)

# Set up session service and runner
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="default")

print("✅ Agent with session state tools initialized!")
```

    ✅ Agent with session state tools initialized!


### 5.3 Testing Session State in Action

Let's test how the agent uses session state to remember information across conversation turns:


```python
# Test conversation demonstrating session state
await run_session(
    runner,
    [
        "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know the name yet
        "My name is Sam. I'm from Poland.",  # Provide name - agent should save it
        "What is my name? Which country am I from?",  # Agent should recall from session state
    ],
    "state-demo-session",
)
```

    
     ### Session: state-demo-session
    
    User > Hi there, how are you doing today? What is my name?
    gemini-2.5-flash-lite >  Hello! I'm doing great. I'd love to tell you your name, but I don't have access to that information. I can remember it if you tell me what it is, though! I can also remember your country if you'd like to share that too.
    
    User > My name is Sam. I'm from Poland.


    WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.


    gemini-2.5-flash-lite >  It's nice to meet you, Sam! I'll be sure to remember that you're from Poland.
    gemini-2.5-flash-lite >  I've saved your name and country. Is there anything else I can help you with today?
    
    User > What is my name? Which country am I from?


    WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.


    gemini-2.5-flash-lite >  Your name is Sam and you are from Poland.


### 5.4 Inspecting Session State

Let's directly inspect the session state to see what's stored:


```python
# Retrieve the session and inspect its state
session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session"
)

print("Session State Contents:")
print(session.state)
print("\n🔍 Notice the 'user:name' and 'user:country' keys storing our data!")
```

    Session State Contents:
    {'user:name': 'Sam', 'user:country': 'Poland'}
    
    🔍 Notice the 'user:name' and 'user:country' keys storing our data!


### 5.5 Session State Isolation

As we've already seen, an important characteristic of session state is that it's isolated per session. Let's demonstrate this by starting a new session:


```python
# Start a completely new session - the agent won't know our name
await run_session(
    runner,
    ["Hi there, how are you doing today? What is my name?"],
    "new-isolated-session",
)

# Expected: The agent won't know the name because this is a different session
```

    
     ### Session: new-isolated-session
    
    User > Hi there, how are you doing today? What is my name?
    gemini-2.5-flash-lite >  Hello! I'm doing well, thank you for asking. I'd love to tell you your name, but I don't have access to that information. I'm a text-based AI and don't have memory of past conversations or personal details.


### 5.6 Cross-Session State Sharing

While sessions are isolated by default, you might notice something interesting. Let's check the state of our new session (`new-isolated-session`):


```python
# Check the state of the new session
session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id="new-isolated-session"
)

print("New Session State:")
print(session.state)

# Note: Depending on implementation, you might see shared state here.
# This is where the distinction between session-specific and user-specific state becomes important.
```

    New Session State:
    {'user:name': 'Sam', 'user:country': 'Poland'}


---

## 🧹 Cleanup


```python
# Clean up any existing database to start fresh (if Notebook is restarted)
import os

if os.path.exists("my_agent_data.db"):
    os.remove("my_agent_data.db")
print("✅ Cleaned up old database files")
```

    ✅ Cleaned up old database files


---
## 📊 Summary

🎉 Congratulations! You've learned the fundamentals of building stateful AI agents:

- ✅ **Context Engineering** - You understand how to assemble context for LLMs using Context Compaction
- ✅ **Sessions & Events** - You can maintain conversation history across multiple turns
- ✅ **Persistent Storage** - You know how to make conversations survive restarts
- ✅ **Session State** - You can track structured data during conversations
- ✅ **Manual State Management** - You've experienced both the power and limitations of manual approaches
- ✅ **Production Considerations** - You're ready to handle real-world challenges


---

## ✅ Congratulations! You did it 🎉

### 📚 Learn More

Refer to the following documentation to learn more:

- [ADK Documentation](https://google.github.io/adk-docs/)
- [ADK Sessions](https://google.github.io/adk-docs/)
- [ADK Session-State](https://medium.com/google-cloud/2-minute-adk-manage-context-efficiently-with-artifacts-6fcc6683d274)
- [ADK Session Compaction](https://google.github.io/adk-docs/context/compaction/#define-compactor)
