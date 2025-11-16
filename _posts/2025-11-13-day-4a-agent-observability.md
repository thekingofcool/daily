---
layout: post
title: Day 4a - Agent Observability
date: 2025-11-13
author: thekingofcool
description: ""
categories: thoughts
---

# 🔎 Agent Observability - Logs, Traces & Metrics

**Welcome to Day 4 of the Kaggle 5-day Agents course!**

In Day 3, you learned the **"What, Why & How" of Session and Memory management**, focusing on long-term, short-term, and shared memory (state). 

Today, you'll learn:
- How to add observability to the agent you've built and
- How to evaluate if the agents are working as expected

In this notebook, we'll focus on the first part - **Agent Observability!**

## What is Agent Observability?

**🚨 The challenge:** Unlike traditional software that fails predictably, AI agents can fail mysteriously. Example:

```
User: "Find quantum computing papers"
Agent: "I cannot help with that request."
You: 😭 WHY?? Is it the prompt? Missing tools? API error?
```

**💡 The Solution:** Agent observability gives you complete visibility into your agent's decision-making process. You'll see exactly what prompts are sent to the LLM, which tools are available, how the model responds, and where failures occur.

```
DEBUG Log: LLM Request shows "Functions: []" (no tools!)
You: 🎯 Aha! Missing google_search tool - easy fix!
```

## Foundational pillars of Agent Observability

1. **Logs:** A log is a record of a single event, telling you **what** happened at a specific moment.
2. **Traces:** A trace connects the logs into a single story, showing you **why** a final result occurred by revealing the entire sequence of steps.
3. **Metrics:** Metrics are the summary numbers (like averages and error rates) that tell you **how** well the agent is performing overall.

<center>
    <img src="https://storage.googleapis.com/github-repo/kaggle-5days-ai/day4/observability-intro.png">
</center>

**In this notebook, you'll:**

* ✅ Set up logging configuration
* ✅ Create a broken agent. Use `adk web` UI & logs to identify exactly why the agent fails
* ✅ Understand how to implement logging in production
* ✅ Learn when to use built-in logging vs custom solutions

## ⚙️ Section 1: Setup

### 1.1: Install dependencies

The Kaggle Notebooks environment includes a pre-installed version of the [google-adk](https://google.github.io/adk-docs/) library for Python and its required dependencies, so you don't need to install additional packages in this notebook.

To install and use ADK in your own Python development environment outside of this course, you can do so by running:

```
pip install google-adk
```

### 🔑 1.2: Configure your Gemini API Key

This notebook uses the [Gemini API](https://ai.google.dev/gemini-api/), which requires an API key.

**1. Get your API key**

If you don't have one already, create an [API key in Google AI Studio](https://aistudio.google.com/app/api-keys).

**2. Add the key to Kaggle Secrets**

Next, you will need to add your API key to your Kaggle Notebook as a Kaggle User Secret.

1. In the top menu bar of the notebook editor, select `Add-ons` then `Secrets`.
2. Create a new secret with the label `GOOGLE_API_KEY`.
3. Paste your API key into the "Value" field and click "Save".
4. Ensure that the checkbox next to `GOOGLE_API_KEY` is selected so that the secret is attached to the notebook.

**3. Authenticate in the notebook**

Run the cell below to access the `GOOGLE_API_KEY` you just saved and set it as an environment variable for the notebook to use:


```python
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Setup and authentication complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )
```

    ✅ Setup and authentication complete.


### ✍️ 1.3: Set up logging and cleanup old files
Let's configure logging for our debugging session. The following cell makes sure we also capture other log levels, like DEBUG.


```python
import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("✅ Logging configured")
```

    ✅ Logging configured


### 💻 1.4: Set up proxy and tunneling

We'll use a proxy to access the ADK web UI from within the Kaggle Notebooks environment. If you are running this outside the Kaggle environment, you don't need to do this.


```python
from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>⚠️ IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) ↗
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("✅ Helper functions defined.")
```

    ✅ Helper functions defined.


---
## 🐞 Section 2: Hands-On Debugging with ADK Web UI

### 2.1: Create a "Research Paper Finder" Agent


**Our goal:** Build a research paper finder agent that helps users find academic papers on any topic.

But first, let's intentionally create an incorrect version of the agent to practice debugging! We'll start by creating a new agent folder using the `adk create` CLI command.


```python
!adk create research-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
```

    [32m
    Agent created in /kaggle/working/research-agent:
    - .env
    - __init__.py
    - agent.py
    [0m


### Agent definition

Next, let's create our root agent. 
- We'll configure it as an `LlmAgent`, give it a name, model and instruction.
- The `root_agent` gets the user prompt and delegates the search to the `google_search_agent`.
- Then, the agent uses the `count_papers` tool to count the number of papers returned.

**👉 Pay attention to the root agent's instructions and the `count_papers` tool parameter!**


```python
%%writefile research-agent/agent.py

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search

from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ---- Intentionally pass incorrect datatype - `str` instead of `List[str]` ----
def count_papers(papers: List[str]):
    """
    This function counts the number of papers in a list of strings.
    Args:
      papers: A list of strings, where each string is a research paper.
    Returns:
      The number of papers in the list.
    """
    return len(papers)


# Google Search agent
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="""Use the google_search tool to find information on the given topic. Return the raw search results.
    If the user asks for a list of papers, then give them the list of research papers you found and not the summary.""",
    tools=[google_search]
)


# Root agent
root_agent = LlmAgent(
    name="research_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Your task is to find research papers and count them. 

    You MUST ALWAYS follow these steps:
    1) Find research papers on the user provided topic using the 'google_search_agent'. 
    2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
    3) Return both the list of research papers and the total number of papers.
    """,
    tools=[AgentTool(agent=google_search_agent), count_papers]
)
```

    Overwriting research-agent/agent.py


### 2.2: Run the agent

Let's now run our agent with the `adk web --log_level DEBUG` CLI command.

**📍 The key here is `--log_level DEBUG`** - this shows us:


* **Full LLM Prompts:** The complete request sent to the language model, including system instructions, history, and tools.
* Detailed API responses from services.
* Internal state transitions and variable values.

Other log levels include: INFO, ERROR and WARNING.

Get the proxied URL to access the ADK web UI in the Kaggle Notebooks environment:


```python
url_prefix = get_adk_proxy_url()
```



<div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
    <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
        <strong>⚠️ IMPORTANT: Action Required</strong>
    </div>
    <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
        The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
        <ol style="margin-top: 10px; padding-left: 20px;">
            <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
            <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
            <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
        </ol>
        <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
    </div>
    <a href='https://kkb-production.jupyter-proxy.kaggle.net/k/277607530/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..z18xG7eWl3AToyJb60TmmQ.M_V9V7e5k7lNoPOe7qPEqvuOxNLDcWQ_5-J1xc5eki2JytpiSHXEcqA292EomZJG0JRDPkxiHKOchetcBEpvLhdyUNlYfCYZDMCPcrpPEEdr5KHVjSIwSSlqjEnSannpvIodiAMHC5Pvuu-HbzDd_LfEXuVfj3Vr1Lcpqz1TAbiawsqb-WT3-NHMq8APujWLnJsMbedQs_CyPxMX75HJHBJus-rS5BkzzbKa-AZBPGz4ORmSvvzrao7IZs_0QPeA.XdQATdP52dt6hwDXtkPejA/proxy/proxy/8000' target='_blank' style="
        display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
        text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
        Open ADK Web UI (after running cell below) ↗
    </a>
</div>



Now you can start the ADK web UI with the `--log_level` parameter.

👉 **Note:** The following cell will not "complete", but will remain running and serving the ADK web UI until you manually stop the cell.


```python
!adk web --log_level DEBUG --url_prefix {url_prefix}
```

    /usr/local/lib/python3.11/dist-packages/google/adk/cli/fast_api.py:130: UserWarning: [EXPERIMENTAL] InMemoryCredentialService: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
      credential_service = InMemoryCredentialService()
    /usr/local/lib/python3.11/dist-packages/google/adk/auth/credential_service/in_memory_credential_service.py:33: UserWarning: [EXPERIMENTAL] BaseCredentialService: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.
      super().__init__()
    [32mINFO[0m:     Started server process [[36m108[0m]
    [32mINFO[0m:     Waiting for application startup.
    [32m
    +-----------------------------------------------------------------------------+
    | ADK Web Server started                                                      |
    |                                                                             |
    | For local testing, access at http://127.0.0.1:8000.                         |
    +-----------------------------------------------------------------------------+
    [0m
    [32mINFO[0m:     Application startup complete.
    [32mINFO[0m:     Uvicorn running on [1mhttp://127.0.0.1:8000[0m (Press CTRL+C to quit)
    [32mINFO[0m:     35.191.59.63:0 - "[1mGET /dev-ui/?app=research-agent&session=34ab44e7-1289-422f-8d39-4e0026bca297&userId=user HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.60:0 - "[1mGET /dev-ui/assets/config/runtime-config.json HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.63:0 - "[1mGET /list-apps?relative_path=./ HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.61:0 - "[1mGET /builder/app/research-agent?ts=1763097491889 HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.60:0 - "[1mGET /apps/research-agent/users/user/sessions/34ab44e7-1289-422f-8d39-4e0026bca297 HTTP/1.1[0m" [31m404 Not Found[0m
    [32mINFO[0m:     35.191.59.62:0 - "[1mGET /builder/app/research-agent?ts=1763097491890 HTTP/1.1[0m" [32m200 OK[0m
    INFO:google_adk.google.adk.cli.adk_web_server:New session created: 557736b1-83fe-4517-a622-6f7daa4fe479
    [32mINFO[0m:     35.191.59.62:0 - "[1mPOST /apps/research-agent/users/user/sessions HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.63:0 - "[1mGET /apps/research-agent/users/user/sessions HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.61:0 - "[1mGET /apps/research-agent/users/user/sessions/557736b1-83fe-4517-a622-6f7daa4fe479 HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.61:0 - "[1mGET /debug/trace/session/557736b1-83fe-4517-a622-6f7daa4fe479 HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.63:0 - "[1mGET /apps/research-agent/eval_results HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.62:0 - "[1mGET /apps/research-agent/eval_sets HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.60:0 - "[1mGET /apps/research-agent/users/user/sessions HTTP/1.1[0m" [32m200 OK[0m
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'appName' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'appName' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'appName' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'userId' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'userId' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'userId' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'sessionId' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'sessionId' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'sessionId' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'newMessage' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'newMessage' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'newMessage' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'streaming' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'streaming' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'streaming' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'stateDelta' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'stateDelta' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'stateDelta' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'alias' attribute with value 'invocationId' was provided to the `Field()` function, which has no effect in the context it was used. 'alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'validation_alias' attribute with value 'invocationId' was provided to the `Field()` function, which has no effect in the context it was used. 'validation_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'serialization_alias' attribute with value 'invocationId' was provided to the `Field()` function, which has no effect in the context it was used. 'serialization_alias' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
      warnings.warn(
    [32mINFO[0m:     35.191.59.60:0 - "[1mPOST /run_sse HTTP/1.1[0m" [32m200 OK[0m
    DEBUG:google_adk.google.adk.cli.utils.agent_loader:Loading agent research-agent - not in cache.
    DEBUG:google_adk.google.adk.cli.utils.agent_loader:Loading .env for agent research-agent from /kaggle/working
    DEBUG:google_adk.google.adk.cli.utils.agent_loader:Module research-agent has no root_agent. Trying next pattern.
    INFO:google_adk.google.adk.cli.utils.agent_loader:Found root_agent in research-agent.agent
    INFO:google_adk.google.adk.models.google_llm:Sending out request, model: gemini-2.5-flash-lite, backend: GoogleLLMVariant.GEMINI_API, stream: False
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Request:
    -----------------------------------------------------------
    System Instruction:
    Your task is to find research papers and count them. 
    
        You MUST ALWAYS follow these steps:
        1) Find research papers on the user provided topic using the 'google_search_agent'. 
        2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
        3) Return both the list of research papers and the total number of papers.
        
    
    You are an agent. Your internal name is "research_paper_finder_agent".
    -----------------------------------------------------------
    Config:
    {'tools': [{}]}
    -----------------------------------------------------------
    Contents:
    {"parts":[{"text":"find the latest paper about Unity"}],"role":"user"}
    -----------------------------------------------------------
    Functions:
    google_search_agent: {'request': {'type': <Type.STRING: 'STRING'>}} 
    count_papers: {'papers': {'items': {'type': <Type.STRING: 'STRING'>}, 'type': <Type.ARRAY: 'ARRAY'>}} 
    -----------------------------------------------------------
    
    INFO:google_adk.google.adk.models.google_llm:Response received from the model.
    WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Response:
    -----------------------------------------------------------
    Text:
    None
    -----------------------------------------------------------
    Function calls:
    name: google_search_agent, args: {'request': 'latest research papers about Unity'}
    -----------------------------------------------------------
    Raw response:
    {"sdk_http_response":{"headers":{"Content-Type":"application/json; charset=UTF-8","Vary":"Origin, X-Origin, Referer","Content-Encoding":"gzip","Date":"Fri, 14 Nov 2025 05:18:36 GMT","Server":"scaffolding on HTTPServer2","X-XSS-Protection":"0","X-Frame-Options":"SAMEORIGIN","X-Content-Type-Options":"nosniff","Server-Timing":"gfet4t7; dur=712","Transfer-Encoding":"chunked"}},"candidates":[{"content":{"parts":[{"function_call":{"args":{"request":"latest research papers about Unity"},"name":"google_search_agent"}}],"role":"model"},"finish_reason":"STOP","index":0}],"model_version":"gemini-2.5-flash-lite","response_id":"q7sWaYzEN9eavr0PwqfsyAM","usage_metadata":{"candidates_token_count":21,"prompt_token_count":241,"prompt_tokens_details":[{"modality":"TEXT","token_count":241}],"total_token_count":262}}
    -----------------------------------------------------------
    
    DEBUG:google_adk.google.adk.cli.adk_web_server:Generated event in agent run streaming: {"modelVersion":"gemini-2.5-flash-lite","content":{"parts":[{"functionCall":{"id":"adk-2c4a4d93-4ad2-4e0d-acd7-a02d8c221f46","args":{"request":"latest research papers about Unity"},"name":"google_search_agent"}}],"role":"model"},"finishReason":"STOP","usageMetadata":{"candidatesTokenCount":21,"promptTokenCount":241,"promptTokensDetails":[{"modality":"TEXT","tokenCount":241}],"totalTokenCount":262},"invocationId":"e-103ba79e-7e3d-423b-bb2c-6ade2311fcb6","author":"research_paper_finder_agent","actions":{"stateDelta":{},"artifactDelta":{},"requestedAuthConfigs":{},"requestedToolConfirmations":{}},"longRunningToolIds":[],"id":"0c4a01ab-8ab2-436f-911b-6c78b415b25e","timestamp":1763097515.249778}
    INFO:google_adk.google.adk.models.google_llm:Sending out request, model: gemini-2.5-flash-lite, backend: GoogleLLMVariant.GEMINI_API, stream: False
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Request:
    -----------------------------------------------------------
    System Instruction:
    Use the google_search tool to find information on the given topic. Return the raw search results.
        If the user asks for a list of papers, then give them the list of research papers you found and not the summary.
    
    You are an agent. Your internal name is "google_search_agent". The description about you is "Searches for information using Google search".
    -----------------------------------------------------------
    Config:
    {}
    -----------------------------------------------------------
    Contents:
    {"parts":[{"text":"latest research papers about Unity"}],"role":"user"}
    -----------------------------------------------------------
    Functions:
    
    -----------------------------------------------------------
    
    INFO:google_adk.google.adk.models.google_llm:Response received from the model.
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Response:
    -----------------------------------------------------------
    Text:
    Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.
    
    In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.
    
    Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.
    
    Furthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of "Unity in Diversity" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features.
    -----------------------------------------------------------
    Function calls:
    
    -----------------------------------------------------------
    Raw response:
    {"sdk_http_response":{"headers":{"Content-Type":"application/json; charset=UTF-8","Vary":"Origin, X-Origin, Referer","Content-Encoding":"gzip","Date":"Fri, 14 Nov 2025 05:18:39 GMT","Server":"scaffolding on HTTPServer2","X-XSS-Protection":"0","X-Frame-Options":"SAMEORIGIN","X-Content-Type-Options":"nosniff","Server-Timing":"gfet4t7; dur=3570","Transfer-Encoding":"chunked"}},"candidates":[{"content":{"parts":[{"text":"Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.\n\nIn the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.\n\nUnity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.\n\nFurthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features."}],"role":"model"},"finish_reason":"STOP","grounding_metadata":{"grounding_chunks":[{"web":{"title":"unity.com","uri":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEWfzfyJifbA9X6tb2thk3nUsTmKdEwCK6qNMbrloF4roVPm9aQimt1urC6-iVAFm-ZmK1-njxvOSZL339pLR71JMI6byQLd2Oz-Rq14i0GBSs9EYD2safN4w=="}},{"web":{"title":"sciencegate.app","uri":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGA4HV4XAE4Sj35RiIVhn4-ubKCuXuzrt-vKw_RQJDpQV1g1OQhpJwhkwrKuWwfCIpnJ5-HVW0gGEFfIIjcjyVRUMsrQ4bFhUcbYe_mGNEmuKt7K4EXRQ61W2k8leB70egdXFbHlg=="}},{"web":{"title":"mdpi.com","uri":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEmKj9pZQs2nYm0O9mfhjdMRGqQhts7SC7lLtGNKEP3h3dWMz-0ThfyDmxcNBEf09F3dg3lq7abhIuxagtc0wNhqlTY54N-rtO8Nf9PkNG1qeSyeY9vOW1LFDNg5XH7jCohEQ=="}},{"web":{"title":"emerald.com","uri":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHe0etwHVfXKV2LxJfwyITec7D6OTcS1DIKJtNx3b9kQNKwS50AlVgOi0KyTfVsnSeZlVEHZ_60KY9feVJbeJ0BfVvg1rVplp6Wbp1czX6kNWKC7Gf7zTCwL8S48FHoc0DmlS5OrlGMcLqOnDcy5nCzT8-ZwyGjhIVsVOeP3ybnonNB-I8rw8otYnOk2V0vwSh6U0vbAskj8UGWZP5eLr4="}},{"web":{"title":"researchgate.net","uri":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQECEfKj6QWZcmqOCdH4JvbYpqgqgxsXxBHnMqOKVLt8KdqImmhEqjV_dWBbZ2MKZyBdZIcn7-KfoFrMBbv983LEN3QSnP3ZYS2MYNIZfCu2_DYMXujyRHKwQiPNfCDCAPiXtGudWn_ogB_wq22Z0dXAal5C-4KycJ-MGd7i-gSMN1ASIlWlvgIeLbm6HV7yBUPb3SWWzd-JqirAg10MFWJOInTqCkpQh3Y="}},{"web":{"title":"researchgate.net","uri":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHqcoAml_vutNSKIyhbZZiqt57s5_o8KC1CpnASH3n4-yMMhD2BrxvoB97BMaKw94doFK-JLeUubTU3YQ1ZZ2nsXh0Xf8zoV9I-eYHOvSR5j8CSpQ7-Rfkz9U_s5av2v7Wp4m3DYlkNkSuQxt6UeSTANg85LvEt5M2I3QvySMF3JA0XfqmYF5Jn9Veoy2T6p7F5AO3T-FRIuMGSJtSc0vwPuludl7R-WpW2_9eYGZicc22U4Ut08Qb2_NiUsVU90cG-oSxobQnkVw0P4S5n5ab-qQsfSxmELRPappCDVJCpmHh-vQWvRP8hxbn9lg-oB8JuU1CcKvIJqfmMe3Hz8Sc3Ygg97uGG0oWcvpAI"}},{"web":{"title":"apjcriweb.org","uri":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHzqeRavr31NJ2gngGPQpCiNhWceFlPa7lhLe2GbFhWj7QOwxUkb--8NgORB7wuMMsuy_vxUjrtjPXG4qkycvNXbT1rvobSQajFYwM7TWFKANAsyAIHympf0TeQHrRzEQd3u_wMPWbOomQ="}}],"grounding_supports":[{"grounding_chunk_indices":[0],"segment":{"end_index":111,"text":"Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization"}},{"grounding_chunk_indices":[0],"segment":{"end_index":217,"start_index":113,"text":"Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones"}},{"grounding_chunk_indices":[0],"segment":{"end_index":333,"start_index":219,"text":"There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps"}},{"grounding_chunk_indices":[0],"segment":{"end_index":537,"start_index":336,"text":"In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications"}},{"grounding_chunk_indices":[0],"segment":{"end_index":755,"start_index":539,"text":"New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure"}},{"grounding_chunk_indices":[0],"segment":{"end_index":917,"start_index":757,"text":"For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images"}},{"grounding_chunk_indices":[1],"segment":{"end_index":1103,"start_index":1010,"text":"Studies have investigated its use in creating virtual laboratories for performance efficiency"}},{"grounding_chunk_indices":[1],"segment":{"end_index":1200,"start_index":1103,"text":", designing digital factories and augmented reality systems for production capability enhancement"}},{"grounding_chunk_indices":[1],"segment":{"end_index":1270,"start_index":1200,"text":", and developing virtual reality user experiences with IoT integration"}},{"grounding_chunk_indices":[2],"segment":{"end_index":1459,"start_index":1272,"text":"For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects"}},{"grounding_chunk_indices":[3],"segment":{"end_index":1759,"start_index":1576,"text":"Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity"}},{"grounding_chunk_indices":[4],"segment":{"end_index":1899,"start_index":1761,"text":"There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation"}},{"grounding_chunk_indices":[5],"segment":{"end_index":2156,"start_index":1901,"text":"A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering"}},{"grounding_chunk_indices":[6],"segment":{"end_index":2309,"start_index":2158,"text":"Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features"}}],"search_entry_point":{"rendered_content":"<style>\n.container {\n  align-items: center;\n  border-radius: 8px;\n  display: flex;\n  font-family: Google Sans, Roboto, sans-serif;\n  font-size: 14px;\n  line-height: 20px;\n  padding: 8px 12px;\n}\n.chip {\n  display: inline-block;\n  border: solid 1px;\n  border-radius: 16px;\n  min-width: 14px;\n  padding: 5px 16px;\n  text-align: center;\n  user-select: none;\n  margin: 0 8px;\n  -webkit-tap-highlight-color: transparent;\n}\n.carousel {\n  overflow: auto;\n  scrollbar-width: none;\n  white-space: nowrap;\n  margin-right: -12px;\n}\n.headline {\n  display: flex;\n  margin-right: 4px;\n}\n.gradient-container {\n  position: relative;\n}\n.gradient {\n  position: absolute;\n  transform: translate(3px, -9px);\n  height: 36px;\n  width: 9px;\n}\n@media (prefers-color-scheme: light) {\n  .container {\n    background-color: #fafafa;\n    box-shadow: 0 0 0 1px #0000000f;\n  }\n  .headline-label {\n    color: #1f1f1f;\n  }\n  .chip {\n    background-color: #ffffff;\n    border-color: #d2d2d2;\n    color: #5e5e5e;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #f2f2f2;\n  }\n  .chip:focus {\n    background-color: #f2f2f2;\n  }\n  .chip:active {\n    background-color: #d8d8d8;\n    border-color: #b6b6b6;\n  }\n  .logo-dark {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);\n  }\n}\n@media (prefers-color-scheme: dark) {\n  .container {\n    background-color: #1f1f1f;\n    box-shadow: 0 0 0 1px #ffffff26;\n  }\n  .headline-label {\n    color: #fff;\n  }\n  .chip {\n    background-color: #2c2c2c;\n    border-color: #3c4043;\n    color: #fff;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #353536;\n  }\n  .chip:focus {\n    background-color: #353536;\n  }\n  .chip:active {\n    background-color: #464849;\n    border-color: #53575b;\n  }\n  .logo-light {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);\n  }\n}\n</style>\n<div class=\"container\">\n  <div class=\"headline\">\n    <svg class=\"logo-light\" width=\"18\" height=\"18\" viewBox=\"9 9 35 35\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">\n      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z\" fill=\"#4285F4\"/>\n      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z\" fill=\"#34A853\"/>\n      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z\" fill=\"#FBBC05\"/>\n      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z\" fill=\"#EA4335\"/>\n    </svg>\n    <svg class=\"logo-dark\" width=\"18\" height=\"18\" viewBox=\"0 0 48 48\" xmlns=\"http://www.w3.org/2000/svg\">\n      <circle cx=\"24\" cy=\"23\" fill=\"#FFF\" r=\"22\"/>\n      <path d=\"M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z\" fill=\"#4285F4\"/>\n      <path d=\"M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z\" fill=\"#34A853\"/>\n      <path d=\"M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z\" fill=\"#FBBC05\"/>\n      <path d=\"M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z\" fill=\"#EA4335\"/>\n    </svg>\n    <div class=\"gradient-container\"><div class=\"gradient\"></div></div>\n  </div>\n  <div class=\"carousel\">\n    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGpPZGXk0OtQYSuoRFPs24oY8-83ayRK9dviyWa7ipYzMEpOIu4SyGOSnzdeP3kXJLhG_m9oG_2TotYGYIY-bPzvgutNoWbCpB9NqD1LQBDhyVfpQ3VMhH9UyCAwQTgAFbr6hAcBxYsI6PiZJFcUNeGlDrrSahmac5mGCYjc4RtiVJggh9jlgVVekP5ebviu0zBpSqSjKBZir5om7Ko1dQ=\">recent Unity research papers</a>\n    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGCiI4IVGQvvNsQHEx42JkdF69pFsE2a1FAdwrtMfIjvUkSCL6CskBPZvgfwsQrDKKtz97pbsY3Rhy4VJZrorskwfKFkSkSuzGvwMXH6P_zSyX5tNCLVIMlzGQlBPCGxvRDFnqyugnYrtourccbdUmbnlS6Ky1LEMZ-iesOpkHlO7JA_0M9SnSrEYtFefc0fvRSW-k1h5UcaaJrfdQb4BIG1_aN3L8=\">latest research papers about Unity</a>\n  </div>\n</div>\n"},"web_search_queries":["latest research papers about Unity","recent Unity research papers"]},"index":0}],"model_version":"gemini-2.5-flash-lite","response_id":"r7sWaf6FJ_ze1e8P7fncmQ8","usage_metadata":{"candidates_token_count":395,"prompt_token_count":85,"prompt_tokens_details":[{"modality":"TEXT","token_count":85}],"tool_use_prompt_token_count":107,"tool_use_prompt_tokens_details":[{"modality":"TEXT","token_count":107}],"total_token_count":587},"automatic_function_calling_history":[]}
    -----------------------------------------------------------
    
    DEBUG:google_adk.google.adk.cli.adk_web_server:Generated event in agent run streaming: {"content":{"parts":[{"functionResponse":{"id":"adk-2c4a4d93-4ad2-4e0d-acd7-a02d8c221f46","name":"google_search_agent","response":{"result":"Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.\n\nIn the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.\n\nUnity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.\n\nFurthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features."}}}],"role":"user"},"invocationId":"e-103ba79e-7e3d-423b-bb2c-6ade2311fcb6","author":"research_paper_finder_agent","actions":{"stateDelta":{},"artifactDelta":{},"requestedAuthConfigs":{},"requestedToolConfirmations":{}},"id":"df6f6de7-5d8e-4402-8a04-374b57436c79","timestamp":1763097519.772138}
    INFO:google_adk.google.adk.models.google_llm:Sending out request, model: gemini-2.5-flash-lite, backend: GoogleLLMVariant.GEMINI_API, stream: False
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Request:
    -----------------------------------------------------------
    System Instruction:
    Your task is to find research papers and count them. 
    
        You MUST ALWAYS follow these steps:
        4) Find research papers on the user provided topic using the 'google_search_agent'. 
        5) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
        6) Return both the list of research papers and the total number of papers.
        
    
    You are an agent. Your internal name is "research_paper_finder_agent".
    -----------------------------------------------------------
    Config:
    {'tools': [{}]}
    -----------------------------------------------------------
    Contents:
    {"parts":[{"text":"find the latest paper about Unity"}],"role":"user"}
    {"parts":[{"function_call":{"args":{"request":"latest research papers about Unity"},"name":"google_search_agent"}}],"role":"model"}
    {"parts":[{"function_response":{"name":"google_search_agent","response":{"result":"Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.\n\nIn the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.\n\nUnity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.\n\nFurthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features."}}}],"role":"user"}
    -----------------------------------------------------------
    Functions:
    google_search_agent: {'request': {'type': <Type.STRING: 'STRING'>}} 
    count_papers: {'papers': {'items': {'type': <Type.STRING: 'STRING'>}, 'type': <Type.ARRAY: 'ARRAY'>}} 
    -----------------------------------------------------------
    
    INFO:google_adk.google.adk.models.google_llm:Response received from the model.
    WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Response:
    -----------------------------------------------------------
    Text:
    None
    -----------------------------------------------------------
    Function calls:
    name: count_papers, args: {'papers': ["Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.", 'In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.', 'Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.', 'Furthermore, Unity\'s capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of "Unity in Diversity" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There\'s also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features.']}
    -----------------------------------------------------------
    Raw response:
    {"sdk_http_response":{"headers":{"Content-Type":"application/json; charset=UTF-8","Vary":"Origin, X-Origin, Referer","Content-Encoding":"gzip","Date":"Fri, 14 Nov 2025 05:18:41 GMT","Server":"scaffolding on HTTPServer2","X-XSS-Protection":"0","X-Frame-Options":"SAMEORIGIN","X-Content-Type-Options":"nosniff","Server-Timing":"gfet4t7; dur=1730","Transfer-Encoding":"chunked"}},"candidates":[{"content":{"parts":[{"function_call":{"args":{"papers":["Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.","In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.","Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.","Furthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features."]},"name":"count_papers"}}],"role":"model"},"finish_reason":"STOP","index":0}],"model_version":"gemini-2.5-flash-lite","response_id":"sbsWaZarGNeavr0PjafsyAM","usage_metadata":{"candidates_token_count":387,"prompt_token_count":657,"prompt_tokens_details":[{"modality":"TEXT","token_count":657}],"total_token_count":1044}}
    -----------------------------------------------------------
    
    DEBUG:google_adk.google.adk.cli.adk_web_server:Generated event in agent run streaming: {"modelVersion":"gemini-2.5-flash-lite","content":{"parts":[{"functionCall":{"id":"adk-8cfed2dd-228d-47fc-a82b-c42e04cbc012","args":{"papers":["Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.","In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.","Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.","Furthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features."]},"name":"count_papers"}}],"role":"model"},"finishReason":"STOP","usageMetadata":{"candidatesTokenCount":387,"promptTokenCount":657,"promptTokensDetails":[{"modality":"TEXT","tokenCount":657}],"totalTokenCount":1044},"invocationId":"e-103ba79e-7e3d-423b-bb2c-6ade2311fcb6","author":"research_paper_finder_agent","actions":{"stateDelta":{},"artifactDelta":{},"requestedAuthConfigs":{},"requestedToolConfirmations":{}},"longRunningToolIds":[],"id":"b10799e7-3dc2-4b7e-8393-8a055bad8156","timestamp":1763097519.773501}
    DEBUG:google_adk.google.adk.cli.adk_web_server:Generated event in agent run streaming: {"content":{"parts":[{"functionResponse":{"id":"adk-8cfed2dd-228d-47fc-a82b-c42e04cbc012","name":"count_papers","response":{"result":4}}}],"role":"user"},"invocationId":"e-103ba79e-7e3d-423b-bb2c-6ade2311fcb6","author":"research_paper_finder_agent","actions":{"stateDelta":{},"artifactDelta":{},"requestedAuthConfigs":{},"requestedToolConfirmations":{}},"id":"879103fd-8f73-48c1-b1b3-42635af17fdb","timestamp":1763097521.510302}
    INFO:google_adk.google.adk.models.google_llm:Sending out request, model: gemini-2.5-flash-lite, backend: GoogleLLMVariant.GEMINI_API, stream: False
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Request:
    -----------------------------------------------------------
    System Instruction:
    Your task is to find research papers and count them. 
    
        You MUST ALWAYS follow these steps:
        7) Find research papers on the user provided topic using the 'google_search_agent'. 
        8) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
        9) Return both the list of research papers and the total number of papers.
        
    
    You are an agent. Your internal name is "research_paper_finder_agent".
    -----------------------------------------------------------
    Config:
    {'tools': [{}]}
    -----------------------------------------------------------
    Contents:
    {"parts":[{"text":"find the latest paper about Unity"}],"role":"user"}
    {"parts":[{"function_call":{"args":{"request":"latest research papers about Unity"},"name":"google_search_agent"}}],"role":"model"}
    {"parts":[{"function_response":{"name":"google_search_agent","response":{"result":"Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.\n\nIn the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.\n\nUnity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.\n\nFurthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features."}}}],"role":"user"}
    {"parts":[{"function_call":{"args":{"papers":["Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.","In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.","Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.","Furthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features."]},"name":"count_papers"}}],"role":"model"}
    {"parts":[{"function_response":{"name":"count_papers","response":{"result":4}}}],"role":"user"}
    -----------------------------------------------------------
    Functions:
    google_search_agent: {'request': {'type': <Type.STRING: 'STRING'>}} 
    count_papers: {'papers': {'items': {'type': <Type.STRING: 'STRING'>}, 'type': <Type.ARRAY: 'ARRAY'>}} 
    -----------------------------------------------------------
    
    INFO:google_adk.google.adk.models.google_llm:Response received from the model.
    DEBUG:google_adk.google.adk.models.google_llm:
    LLM Response:
    -----------------------------------------------------------
    Text:
    Here are the latest research papers about Unity:
    
    *   Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.
    *   In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.
    *   Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.
    *   Furthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of "Unity in Diversity" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features.
    
    There are a total of 4 research papers on the topic.
    -----------------------------------------------------------
    Function calls:
    
    -----------------------------------------------------------
    Raw response:
    {"sdk_http_response":{"headers":{"Content-Type":"application/json; charset=UTF-8","Vary":"Origin, X-Origin, Referer","Content-Encoding":"gzip","Date":"Fri, 14 Nov 2025 05:18:42 GMT","Server":"scaffolding on HTTPServer2","X-XSS-Protection":"0","X-Frame-Options":"SAMEORIGIN","X-Content-Type-Options":"nosniff","Server-Timing":"gfet4t7; dur=1419","Transfer-Encoding":"chunked"}},"candidates":[{"content":{"parts":[{"text":"Here are the latest research papers about Unity:\n\n*   Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.\n*   In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.\n*   Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.\n*   Furthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features.\n\nThere are a total of 4 research papers on the topic."}],"role":"model"},"finish_reason":"STOP","index":0}],"model_version":"gemini-2.5-flash-lite","response_id":"srsWabmeLoekvr0P8LS7gQ8","usage_metadata":{"candidates_token_count":405,"prompt_token_count":1059,"prompt_tokens_details":[{"modality":"TEXT","token_count":1059}],"total_token_count":1464}}
    -----------------------------------------------------------
    
    DEBUG:google_adk.google.adk.cli.adk_web_server:Generated event in agent run streaming: {"modelVersion":"gemini-2.5-flash-lite","content":{"parts":[{"text":"Here are the latest research papers about Unity:\n\n*   Unity is a versatile game engine with ongoing research in areas like graphics, AI, and performance optimization. Recent research includes improving real-time ray tracing with enhanced texture filtering using ray cones. There's also work on a surface gradient-based bump mapping framework for layering and compositing bump/normal maps.\n*   In the realm of virtual environments and simulations, researchers are exploring data-driven paradigms for precomputed radiance transfer, aiming to simplify the process for machine learning applications. New architectures are being developed for spatio-temporal traffic forecasting, with one novel learning architecture showing competitive or superior performance without requiring prior knowledge of the graph structure. For graphics, research has focused on creating Wang tiles for procedural noise functions to enable non-periodic tiling and reduce repetition in generated images.\n*   Unity is also being utilized in academic research for various applications beyond gaming. Studies have investigated its use in creating virtual laboratories for performance efficiency, designing digital factories and augmented reality systems for production capability enhancement, and developing virtual reality user experiences with IoT integration. For augmented and virtual reality development, systems are being created to simplify the process for developers, allowing for easier interaction and behavioral configuration of 3D objects.\n*   Furthermore, Unity's capabilities are being explored in the context of broader societal and technological trends. Research has touched upon the concept of \"Unity in Diversity\" in navigating global connections through cultural exchange, exploring how to balance global unity with cultural diversity. There's also ongoing work on using Unity as a foundation for achieving a sustainable world, emphasizing global citizenship and cooperation. A technical survey of the Unity game development engine highlights its benefits, challenges, and potential solutions, noting its widespread adoption in various industries beyond gaming, such as film, architecture, automotive, construction, and engineering. Recent academic papers also offer comparative analyses of Unity against other game engines like Unreal Engine, discussing their efficiency and features.\n\nThere are a total of 4 research papers on the topic."}],"role":"model"},"finishReason":"STOP","usageMetadata":{"candidatesTokenCount":405,"promptTokenCount":1059,"promptTokensDetails":[{"modality":"TEXT","tokenCount":1059}],"totalTokenCount":1464},"invocationId":"e-103ba79e-7e3d-423b-bb2c-6ade2311fcb6","author":"research_paper_finder_agent","actions":{"stateDelta":{},"artifactDelta":{},"requestedAuthConfigs":{},"requestedToolConfirmations":{}},"id":"d7fb0f52-a106-4a67-a045-69ec9e37ba15","timestamp":1763097521.511682}
    [32mINFO[0m:     35.191.59.60:0 - "[1mGET /debug/trace/session/557736b1-83fe-4517-a622-6f7daa4fe479 HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.63:0 - "[1mGET /apps/research-agent/users/user/sessions/557736b1-83fe-4517-a622-6f7daa4fe479 HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.61:0 - "[1mGET /debug/trace/session/557736b1-83fe-4517-a622-6f7daa4fe479 HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.61:0 - "[1mGET /debug/trace/879103fd-8f73-48c1-b1b3-42635af17fdb HTTP/1.1[0m" [32m200 OK[0m
    DEBUG:google_adk.google.adk.cli.utils.agent_loader:Returning cached agent for research-agent (async)
    [32mINFO[0m:     35.191.59.62:0 - "[1mGET /apps/research-agent/users/user/sessions/557736b1-83fe-4517-a622-6f7daa4fe479/events/879103fd-8f73-48c1-b1b3-42635af17fdb/graph HTTP/1.1[0m" [32m200 OK[0m
    [32mINFO[0m:     35.191.59.63:0 - "[1mGET /debug/trace/d7fb0f52-a106-4a67-a045-69ec9e37ba15 HTTP/1.1[0m" [32m200 OK[0m
    DEBUG:google_adk.google.adk.cli.utils.agent_loader:Returning cached agent for research-agent (async)
    [32mINFO[0m:     35.191.59.60:0 - "[1mGET /apps/research-agent/users/user/sessions/557736b1-83fe-4517-a622-6f7daa4fe479/events/d7fb0f52-a106-4a67-a045-69ec9e37ba15/graph HTTP/1.1[0m" [32m200 OK[0m
    ^C
    [32mINFO[0m:     Shutting down
    [32mINFO[0m:     Waiting for application shutdown.
    [32m
    +-----------------------------------------------------------------------------+
    | ADK Web Server shutting down...                                             |
    +-----------------------------------------------------------------------------+
    [0m
    [32mINFO[0m:     Application shutdown complete.
    [32mINFO[0m:     Finished server process [[36m108[0m]
    
    Aborted!


Once the ADK web UI starts, open the proxy link using the button in the previous cell.

As you start chatting with the agent, you should see the DEBUG logs appear in the output cell below!

‼️ **IMPORTANT: DO NOT SHARE THE PROXY LINK** with anyone - treat it as sensitive data as it contains your authentication token in the URL.

### 📝 2.3: Test the agent in ADK web UI

#### **👉 Do: In the ADK web UI**

1. Select "research-agent" from the dropdown in the top-left.
2. In the chat interface, type: `Find latest quantum computing papers`
3. Send the message and observe the response. The agent should return a list of research papers and their count.

It looks like our agent works and we got a response! 🤔 **But wait, isn't the count of papers unusually large? Let's look at the logs and trace.** 

#### **👉 Do: Events tab - Traces in detail**

1. In the web UI, click the **"Events"** tab on the left sidebar
2. You'll see a chronological list of all agent actions
3. Click on any event to expand its details in the bottom panel
4. Try clicking the **"Trace"** button to see timing information for each step.
5. **Click the `execute_tool count_papers` span. You'll see that the function call to `count_papers` returns the large number as the response**.
6. Let's look at what was passed as input to this function. 
7. **Find the `call_llm` span corresponding to the `count_papers` function call**.

#### **👉 Do: Inspect the Function call in Events:**

- Click on the specific span to open the Events tab.
- Examine the `function_call`, focusing on the `papers` argument.
- Notice that `root_agent` passes the list of `papers` as a **str** instead of a **List[str]** - there's our bug! 

![Demo](https://storage.googleapis.com/github-repo/kaggle-5days-ai/day4/observability-demo.gif)

### 2.4: Your Turn - fix it! 👾 

Update the datatype of the `papers` argument in the `count_papers` tool to a `List[str]` and rerun the `adk web` command!

---

## ‼️ **Stop the ADK web UI** 🛑

**In order to run cells in the remainder of this notebook,** please stop the running cell where you started `adk web` in Section 3.1.

Otherwise that running cell will block / prevent other cells from running as long as the ADK web UI is running.

---

### 2.5: Debug through local Logs

Optionally, you can also examine the local DEBUG logs to find the root cause. Run the following cell to print the contents of the log file. Look for detailed logs like:
```
DEBUG - google_adk.models.google_llm - LLM Request: ...
DEBUG - google_adk.models.google_llm - LLM Response: ...
```


```python
# Check the DEBUG logs from the broken agent
print("🔍 Examining web server logs for debugging clues...\n")
!cat logger.log
```

    🔍 Examining web server logs for debugging clues...
    


**Other Observability questions you can now answer from logs and adk web:**
- **Efficiency**: Is the agent making optimal tool choices?
- **Reasoning Quality**: Are the prompts well-structured and context-appropriate?
- **Performance**: Look at the traces to identify which steps take the longest?
- **Failure Diagnosis**: When something goes wrong, where exactly did it fail?

**Key Learning:** Core debugging pattern: `symptom → logs → root cause → fix`.

**Debugging Victory:** You just went from "Agent mysteriously failed" to "I know exactly why and how to fix it!" This is the power of observability!


---
## 🧑‍💻 Section 3: Logging in production

**🎯 Great! You can now debug agent failures using ADK web UI and DEBUG logs.**

But what happens when you move beyond development? Real-world scenarios where you need to move beyond the web UI:

**❌ Problem 1: Production Deployment**
```
You: "Let me open the ADK web UI to check why the agent failed"
DevOps: "Um... this is a production server. No web UI access."
You: 😱 "How do I debug production issues?"
```

**❌ Problem 2: Automated Systems** 
```
You: "The agent runs 1000 times per day in our pipeline"
Boss: "Which runs are slow? What's our success rate?"
You: 😰 "I'd have to manually check the web UI 1000 times..."
```

**💡 The Solution:**

We need a way to capture observability data or in other words, **add logs to our code**. 

👉 In traditional software development, this is done by adding log statements in Python functions - **and agents are no different!** We need to add log statements to our agent and a common approach is to add log statements to **Plugins**.


### 3.1: How to add logs for production observability?

A Plugin is a custom code module that runs automatically at various stages of your agent's lifecycle. Plugins are composed of "**Callbacks**" which provide the hooks to interrupt an agent's flow. Think of it like this:

- **Your agent workflow**: User message → Agent thinks → Calls tools → Returns response
- **Plugin hooks into this**: Before agent starts → After tool runs → When LLM responds → etc.
- **Plugin contains your custom code**: Logging, monitoring, security checks, caching, etc.

![image.png](https://storage.googleapis.com/github-repo/kaggle-5days-ai/day4/plugins-callbacks.png)

#### Callbacks

Callbacks are the **atomic components inside a Plugin** - these are just Python functions that run at specific points in an agent's lifecycle! **Callbacks are grouped together to create a Plugin.**

There are different kinds of callbacks such as:
* **before/after_agent_callbacks** - runs before/after an agent is invoked
* **before/after_tool_callbacks** - runs before/after a tool is called
* **before/after_model_callbacks** - similarly, runs before/after the LLM model is called
* **on_model_error_callback** - which runs when a model error is encountered

![image.png](https://storage.googleapis.com/github-repo/kaggle-5days-ai/day4/types_of_callbacks.png)

### 3.2: To make things more concrete, what does a Plugin look like?


```python
print("----- EXAMPLE PLUGIN - DOES NOTHING ----- ")

import logging
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin


# Applies to all agent and model calls
class CountInvocationPlugin(BasePlugin):
    """A custom plugin that counts agent and tool invocations."""

    def __init__(self) -> None:
        """Initialize the plugin with counters."""
        super().__init__(name="count_invocation")
        self.agent_count: int = 0
        self.tool_count: int = 0
        self.llm_request_count: int = 0

    # Callback 1: Runs before an agent is called. You can add any custom logic here.
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Count agent runs."""
        self.agent_count += 1
        logging.info(f"[Plugin] Agent run count: {self.agent_count}")

    # Callback 2: Runs before a model is called. You can add any custom logic here.
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Count LLM requests."""
        self.llm_request_count += 1
        logging.info(f"[Plugin] LLM request count: {self.llm_request_count}")
```

    ----- EXAMPLE PLUGIN - DOES NOTHING ----- 


**Key insight**: You register a plugin **once** on your runner, and it automatically applies to **every agent, tool call, and LLM request** in your system as per your definition. Read more about Plugin hooks [here](https://google.github.io/adk-docs/plugins/#plugin-callback-hooks).

You can follow along with the numbers in the diagram below to understand the flow.

![image.png](https://storage.googleapis.com/github-repo/kaggle-5days-ai/day4/count-invocation-plugin.png)

### 3.3: ADK's built-in `LoggingPlugin`

But you don't have to define all the callbacks and plugins to capture *standard* Observability data in ADK. Instead, ADK provides a built-in **LoggingPlugin** that automatically captures all agent activity:

- 🚀 User messages and agent responses
- ⏱️ Timing data for performance analysis
- 🧠 LLM requests and responses for debugging
- 🔧 Tool calls and results
- ✅ Complete execution traces

#### Agent definition

Let's use the same agent from the previous demo - the Research paper finder!


```python
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search

from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


def count_papers(papers: List[str]):
    """
    This function counts the number of papers in a list of strings.
    Args:
      papers: A list of strings, where each string is a research paper.
    Returns:
      The number of papers in the list.
    """
    return len(papers)


# Google search agent
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="Use the google_search tool to find information on the given topic. Return the raw search results.",
    tools=[google_search],
)

# Root agent
research_agent_with_plugin = LlmAgent(
    name="research_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Your task is to find research papers and count them. 
   
   You must follow these steps:
   1) Find research papers on the user provided topic using the 'google_search_agent'. 
   2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
   3) Return both the list of research papers and the total number of papers.
   """,
    tools=[AgentTool(agent=google_search_agent), count_papers],
)

print("✅ Agent created")
```

    ✅ Agent created


### 3.4: Add LoggingPlugin to Runner

The following code creates the `InMemoryRunner`. This is used to programmatically invoke the agent.

**To use `LoggingPlugin` in the above research agent,**
1) Import the plugin
2) Add it when initializing the `InMemoryRunner`.



```python
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
from google.genai import types
import asyncio

runner = InMemoryRunner(
    agent=research_agent_with_plugin,
    plugins=[
        LoggingPlugin()
    ],  # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents
)

print("✅ Runner configured")
```

    ✅ Runner configured


Let's now run the agent using `run_debug` function.


```python
print("🚀 Running agent with LoggingPlugin...")
print("📊 Watch the comprehensive logging output below:\n")

response = await runner.run_debug("Find recent papers on quantum computing")
```

    🚀 Running agent with LoggingPlugin...
    📊 Watch the comprehensive logging output below:
    
    
     ### Created new session: debug_session_id
    
    User > Find recent papers on quantum computing
    [90m[logging_plugin] 🚀 USER MESSAGE RECEIVED[0m
    [90m[logging_plugin]    Invocation ID: e-82360c95-5996-48bb-aa07-763dd2f280a7[0m
    [90m[logging_plugin]    Session ID: debug_session_id[0m
    [90m[logging_plugin]    User ID: debug_user_id[0m
    [90m[logging_plugin]    App Name: InMemoryRunner[0m
    [90m[logging_plugin]    Root Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    User Content: text: 'Find recent papers on quantum computing'[0m
    [90m[logging_plugin] 🏃 INVOCATION STARTING[0m
    [90m[logging_plugin]    Invocation ID: e-82360c95-5996-48bb-aa07-763dd2f280a7[0m
    [90m[logging_plugin]    Starting Agent: research_paper_finder_agent[0m
    [90m[logging_plugin] 🤖 AGENT STARTING[0m
    [90m[logging_plugin]    Agent Name: research_paper_finder_agent[0m
    [90m[logging_plugin]    Invocation ID: e-82360c95-5996-48bb-aa07-763dd2f280a7[0m
    [90m[logging_plugin] 🧠 LLM REQUEST[0m
    [90m[logging_plugin]    Model: gemini-2.5-flash-lite[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    System Instruction: 'Your task is to find research papers and count them. 
       
       You must follow these steps:
       1) Find research papers on the user provided topic using the 'google_search_agent'. 
       2) Then, pass the p...'[0m
    [90m[logging_plugin]    Available Tools: ['google_search_agent', 'count_papers'][0m
    [90m[logging_plugin] 🧠 LLM RESPONSE[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: function_call: google_search_agent[0m
    [90m[logging_plugin]    Token Usage - Input: 242, Output: 21[0m
    [90m[logging_plugin] 📢 EVENT YIELDED[0m
    [90m[logging_plugin]    Event ID: 91a5d928-cadd-4b97-a1ef-b8baab6c870d[0m
    [90m[logging_plugin]    Author: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: function_call: google_search_agent[0m
    [90m[logging_plugin]    Final Response: False[0m
    [90m[logging_plugin]    Function Calls: ['google_search_agent'][0m
    [90m[logging_plugin] 🔧 TOOL STARTING[0m
    [90m[logging_plugin]    Tool Name: google_search_agent[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    Function Call ID: adk-bb5a53df-d961-4dd5-b4c4-345636d113bc[0m
    [90m[logging_plugin]    Arguments: {'request': 'recent papers on quantum computing'}[0m
    [90m[logging_plugin] 🚀 USER MESSAGE RECEIVED[0m
    [90m[logging_plugin]    Invocation ID: e-4cb5f307-ac44-4511-b077-03d0ae0ec4b0[0m
    [90m[logging_plugin]    Session ID: dffe97d3-763d-4280-aa8c-f174c990010f[0m
    [90m[logging_plugin]    User ID: debug_user_id[0m
    [90m[logging_plugin]    App Name: InMemoryRunner[0m
    [90m[logging_plugin]    Root Agent: google_search_agent[0m
    [90m[logging_plugin]    User Content: text: 'recent papers on quantum computing'[0m
    [90m[logging_plugin] 🏃 INVOCATION STARTING[0m
    [90m[logging_plugin]    Invocation ID: e-4cb5f307-ac44-4511-b077-03d0ae0ec4b0[0m
    [90m[logging_plugin]    Starting Agent: google_search_agent[0m
    [90m[logging_plugin] 🤖 AGENT STARTING[0m
    [90m[logging_plugin]    Agent Name: google_search_agent[0m
    [90m[logging_plugin]    Invocation ID: e-4cb5f307-ac44-4511-b077-03d0ae0ec4b0[0m
    [90m[logging_plugin] 🧠 LLM REQUEST[0m
    [90m[logging_plugin]    Model: gemini-2.5-flash-lite[0m
    [90m[logging_plugin]    Agent: google_search_agent[0m
    [90m[logging_plugin]    System Instruction: 'Use the google_search tool to find information on the given topic. Return the raw search results.
    
    You are an agent. Your internal name is "google_search_agent". The description about you is "Searches...'[0m
    [90m[logging_plugin] 🧠 LLM RESPONSE[0m
    [90m[logging_plugin]    Agent: google_search_agent[0m
    [90m[logging_plugin]    Content: text: 'Recent breakthroughs in quantum computing are paving the way for more powerful and practical applications. Key advancements have been made in quantum hardware, algorithms, and error correction, signal...'[0m
    [90m[logging_plugin]    Token Usage - Input: 58, Output: 412[0m
    [90m[logging_plugin] 📢 EVENT YIELDED[0m
    [90m[logging_plugin]    Event ID: fbd9410d-a558-4825-8e09-2cca07ba197b[0m
    [90m[logging_plugin]    Author: google_search_agent[0m
    [90m[logging_plugin]    Content: text: 'Recent breakthroughs in quantum computing are paving the way for more powerful and practical applications. Key advancements have been made in quantum hardware, algorithms, and error correction, signal...'[0m
    [90m[logging_plugin]    Final Response: True[0m
    [90m[logging_plugin] 🤖 AGENT COMPLETED[0m
    [90m[logging_plugin]    Agent Name: google_search_agent[0m
    [90m[logging_plugin]    Invocation ID: e-4cb5f307-ac44-4511-b077-03d0ae0ec4b0[0m
    [90m[logging_plugin] ✅ INVOCATION COMPLETED[0m
    [90m[logging_plugin]    Invocation ID: e-4cb5f307-ac44-4511-b077-03d0ae0ec4b0[0m
    [90m[logging_plugin]    Final Agent: google_search_agent[0m
    [90m[logging_plugin] 🔧 TOOL COMPLETED[0m
    [90m[logging_plugin]    Tool Name: google_search_agent[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    Function Call ID: adk-bb5a53df-d961-4dd5-b4c4-345636d113bc[0m
    [90m[logging_plugin]    Result: Recent breakthroughs in quantum computing are paving the way for more powerful and practical applications. Key advancements have been made in quantum hardware, algorithms, and error correction, signaling a transition from theoretical concepts to tangible reality.
    
    Significant progress has been seen ...}[0m
    [90m[logging_plugin] 📢 EVENT YIELDED[0m
    [90m[logging_plugin]    Event ID: e1682479-a855-4904-be6e-e44320ab7268[0m
    [90m[logging_plugin]    Author: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: function_response: google_search_agent[0m
    [90m[logging_plugin]    Final Response: False[0m
    [90m[logging_plugin]    Function Responses: ['google_search_agent'][0m
    [90m[logging_plugin] 🧠 LLM REQUEST[0m
    [90m[logging_plugin]    Model: gemini-2.5-flash-lite[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    System Instruction: 'Your task is to find research papers and count them. 
       
       You must follow these steps:
       1) Find research papers on the user provided topic using the 'google_search_agent'. 
       2) Then, pass the p...'[0m
    [90m[logging_plugin]    Available Tools: ['google_search_agent', 'count_papers'][0m
    [90m[logging_plugin] 🧠 LLM RESPONSE[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: function_call: count_papers[0m
    [90m[logging_plugin]    Token Usage - Input: 668, Output: 403[0m
    [90m[logging_plugin] 📢 EVENT YIELDED[0m
    [90m[logging_plugin]    Event ID: c73153f6-6e00-43c3-bca7-43a62deb05f5[0m
    [90m[logging_plugin]    Author: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: function_call: count_papers[0m
    [90m[logging_plugin]    Final Response: False[0m
    [90m[logging_plugin]    Function Calls: ['count_papers'][0m
    [90m[logging_plugin] 🔧 TOOL STARTING[0m
    [90m[logging_plugin]    Tool Name: count_papers[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    Function Call ID: adk-ba454ee2-ddda-4856-836b-5d33bc4b7840[0m
    [90m[logging_plugin]    Arguments: {'papers': ['Recent breakthroughs in quantum computing are paving the way for more powerful and practical applications. Key advancements have been made in quantum hardware, algorithms, and error correction, signaling a transition from theoretical concepts to tangible reality.\n\nSignificant progress...}[0m
    [90m[logging_plugin] 🔧 TOOL COMPLETED[0m
    [90m[logging_plugin]    Tool Name: count_papers[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    Function Call ID: adk-ba454ee2-ddda-4856-836b-5d33bc4b7840[0m
    [90m[logging_plugin]    Result: 1[0m
    [90m[logging_plugin] 📢 EVENT YIELDED[0m
    [90m[logging_plugin]    Event ID: c78f8a7f-27f9-4a4e-bb16-59b61f367757[0m
    [90m[logging_plugin]    Author: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: function_response: count_papers[0m
    [90m[logging_plugin]    Final Response: False[0m
    [90m[logging_plugin]    Function Responses: ['count_papers'][0m
    [90m[logging_plugin] 🧠 LLM REQUEST[0m
    [90m[logging_plugin]    Model: gemini-2.5-flash-lite[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    System Instruction: 'Your task is to find research papers and count them. 
       
       You must follow these steps:
       1) Find research papers on the user provided topic using the 'google_search_agent'. 
       2) Then, pass the p...'[0m
    [90m[logging_plugin]    Available Tools: ['google_search_agent', 'count_papers'][0m
    [90m[logging_plugin] 🧠 LLM RESPONSE[0m
    [90m[logging_plugin]    Agent: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: text: 'Here are the recent papers on quantum computing:
    
    Recent breakthroughs in quantum computing are paving the way for more powerful and practical applications. Key advancements have been made in quantum ...'[0m
    [90m[logging_plugin]    Token Usage - Input: 1086, Output: 402[0m
    [90m[logging_plugin] 📢 EVENT YIELDED[0m
    [90m[logging_plugin]    Event ID: 911faaf0-573c-4a02-9eb6-e66f97b5b408[0m
    [90m[logging_plugin]    Author: research_paper_finder_agent[0m
    [90m[logging_plugin]    Content: text: 'Here are the recent papers on quantum computing:
    
    Recent breakthroughs in quantum computing are paving the way for more powerful and practical applications. Key advancements have been made in quantum ...'[0m
    [90m[logging_plugin]    Final Response: True[0m
    research_paper_finder_agent > Here are the recent papers on quantum computing:
    
    Recent breakthroughs in quantum computing are paving the way for more powerful and practical applications. Key advancements have been made in quantum hardware, algorithms, and error correction, signaling a transition from theoretical concepts to tangible reality.
    
    Significant progress has been seen in the development of quantum processors. IBM, for instance, has unveiled its most advanced processor yet, the IBM Quantum Nighthawk, designed to deliver quantum advantage—where a quantum computer solves a problem better than any classical computer—by 2026. This processor features 120 qubits and enhanced connectivity, allowing for more complex circuits. IBM has also announced the IBM Quantum Loon processor, which incorporates essential hardware components for future fault-tolerant quantum computers. Princeton engineers have developed a new superconducting qubit that lasts three times longer than current versions, a critical step for error correction and scalability.
    
    Quantum error correction remains a central focus, with researchers demonstrating new systems and architectures to mitigate errors in quantum computations. Google Quantum AI has published research on an efficient quantum algorithm called Decoded Quantum Interferometry (DQI), which leverages quantum mechanics for optimization problems. Harvard researchers have demonstrated a fault-tolerant system using 448 atomic qubits, combining essential elements for scalable, error-corrected quantum computation.
    
    Beyond hardware and algorithms, advancements are also being made in related fields such as post-quantum cryptography, which is crucial for securing data against future quantum threats. Furthermore, the integration of quantum computing with machine learning is being explored to enhance computational power and optimize quantum algorithms. Companies like IQM Quantum Computers are launching new product lines specifically designed to accelerate quantum error correction research. Microsoft has also expanded its quantum facility to focus on topological qubit fabrication, a key step towards scalable, fault-tolerant quantum computing.
    
    The overall trend indicates a rapid acceleration in quantum computing research and development, with major technology companies and research institutions making substantial contributions towards realizing the potential of this transformative technology.
    
    There is a total of 1 paper.
    [90m[logging_plugin] 🤖 AGENT COMPLETED[0m
    [90m[logging_plugin]    Agent Name: research_paper_finder_agent[0m
    [90m[logging_plugin]    Invocation ID: e-82360c95-5996-48bb-aa07-763dd2f280a7[0m
    [90m[logging_plugin] ✅ INVOCATION COMPLETED[0m
    [90m[logging_plugin]    Invocation ID: e-82360c95-5996-48bb-aa07-763dd2f280a7[0m
    [90m[logging_plugin]    Final Agent: research_paper_finder_agent[0m


---

## 📊 Summary

**❓ When to use which type of Logging?**
1. **Development debugging?** → Use `adk web --log_level DEBUG`
2. **Common production observability?** → Use `LoggingPlugin()` 
3. **Custom requirements?** → Build Custom Callbacks and Plugins

### Try it out!

👉 Extend the agent's observability by implementing a **custom ADK plugin** that tracks and reports the total number of tool calls made during a session.

## 🎯 Congratulations!

**You now know how to:**

- ✅ Debug agent failures through DEBUG logs and the ADK web UI
- ✅ Use the core debugging pattern: symptom → logs → root cause → fix  
- ✅ Scale observability with `LoggingPlugin` for production systems
- ✅ Understand when to use the different logging types

### 📚 Resources

**Refer to the ADK documentation to learn more about observability:**

- [ADK Observability Documentation](https://google.github.io/adk-docs/observability/logging/) - Complete guide to logging in ADK
- [Custom Plugin](https://google.github.io/adk-docs/plugins/) - Build your own Plugins
- [External Integrations](https://google.github.io/adk-docs/observability/cloud-trace/) - Explore external third-party observability integrations with ADK
