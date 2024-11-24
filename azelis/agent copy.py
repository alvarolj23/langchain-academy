import os
from dotenv import load_dotenv
import sys

load_dotenv()


from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import AnyMessage, HumanMessage
import operator
from agent_validators import *
from agent_tools import map_product_name_to_id, search_products_by_question, save_html_to_file
from datetime import datetime
from utils import get_model
import logging
import logging_config

logger = logging.getLogger(__name__)


class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


# Tools
tools = [map_product_name_to_id, search_products_by_question, save_html_to_file]
tool_node = ToolNode(tools)

# Set up model
model = get_model('azure')
model = model.bind_tools(tools = tools)


def should_continue(state: MessagesState) -> Literal["tools", "human_feedback"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "human_feedback"

#The commented part is because it breaks the UI with the input function
def should_continue_with_feedback(state: MessagesState) -> Literal["agent", "end"]:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, dict):
        if last_message.get("type","") == 'human':
            return "agent"
    if (isinstance(last_message, HumanMessage)):
        return "agent"
    return "end"


def call_model(state: MessagesState):
    messages = [SystemMessage(content=f"You are helpful assistant in Azelis, a global distributor of specialty chemicals and food ingredients in Belgium (Europe).\nAs reference, today is {datetime.now().strftime('%Y-%m-%d %H:%M, %A')}.\nKeep a friendly, professional tone.\nAvoid verbosity.\nConsiderations:\n- Don´t assume parameters in call functions that it didnt say.\n- MUST NOT force users how to write. Let them write in the way they want.\n- The conversation should be very natural like a secretary talking with a client.\n- Call only ONE tool at a time.")] + state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def read_human_feedback(state: MessagesState):
    pass


# Build graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("human_feedback", read_human_feedback)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"human_feedback": 'human_feedback',
    "tools": "tools"}
)
workflow.add_conditional_edges(
    "human_feedback",
    should_continue_with_feedback,
    {"agent": 'agent',
    "end": END}
)
workflow.add_edge("tools", 'agent')

# Memory
checkpointer = MemorySaver()


app = workflow.compile(checkpointer=checkpointer, 
                       interrupt_before=['human_feedback'])

if __name__ == '__main__':
    while True:
        question = input("Put your question: ")

        for event in app.stream(
            {"messages": [
                HumanMessage(content=question)
                ]},
            config={"configurable": {"thread_id": 42}}
            ):
            if event.get("agent","") == "":
                continue
            else:
                msg = event['agent']['messages'][-1].content
                if msg == '':
                    continue
                else:
                    print(msg)