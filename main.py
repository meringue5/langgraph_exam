"""
the GOAL for this code:

using langGraph, create a simple multi agent system.
the sample code to refer is 'sample_code/sample_langgraph.py'

[structure]
the app has one supervisor agent and two sub-agents.
- supervisor: 
    keep conversation and route to sub-agents. when the user says "FINISH", it will end the conversation.
    make report of impact of Trump and Vance on the asked company utilizing the sub-agents.
    when the question is irrelevant to stock market, refuse to answer.
- sub-agent1: scrape news of Donald J. Trump and J. D. Vance for last 24 hours.
- sub-agent2: get last makret price and new related to asked company.
- tools: I don't know but maybe news scraper, web search

[restriction]
don't use edges, but use langgraph.commands to control the flow of the program.
if needed, use langgraph.messages to store the conversation history.
each conversation should be stored in a list of messages, and the supervisor agent should use this list to keep track of the conversation history.
the conversation history should be logged so that later analysis can be done on the conversation flow and the agents' responses
"""