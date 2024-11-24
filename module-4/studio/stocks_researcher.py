import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import AzureChatOpenAI

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
import os
import requests
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

### LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    temperature=0
)

### Schema

class Analyst(BaseModel):
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of stock analysis.",
    )
    description: str = Field(
        description="Description of the analyst's focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and specializations.",
    )

class GenerateAnalystsState(TypedDict):
    stock_ticker: str  # Stock ticker of interest
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analysts generated

class InterviewState(TypedDict):
    messages: list  # List of messages
    max_num_turns: int  # Number of conversation turns
    context: Annotated[list, operator.add]  # Source documents
    analyst: Analyst  # Analyst conducting the interview
    interview: str  # Interview transcript
    sections: list  # Sections for the final report
    interview_stock_ticker: str  # Renamed from 'stock_ticker' to avoid collision

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchGraphState(TypedDict):
    stock_ticker: str  # Stock ticker of interest
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analysts
    sections: Annotated[list, operator.add]  # Sections for the final report
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report

### Nodes and Edges

analyst_instructions = """You are tasked with creating a set of AI analyst personas focused on analyzing the stock of {stock_ticker}. Follow these instructions carefully:

1. Review the company:

{stock_ticker}

2. Create analysts for the following areas of analysis: Technical Analysis, Fundamental Analysis, Sentiment Analysis, Industry Analysis, Regulatory Analysis, Investment Recommendation.

3. For each area, provide an analyst persona with a unique name, role, and description of their focus.

4. Ensure that each analyst is uniquely focused on their area of expertise."""

def create_analysts(state: GenerateAnalystsState):
    """Create analysts"""

    stock_ticker = state['stock_ticker']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(stock_ticker=stock_ticker, human_analyst_feedback=human_analyst_feedback)

    # Generate analysts
    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")])

    # Write the list of analysts to state
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interrupted on"""
    pass

# Generate analyst question
question_instructions = """You are an analyst specializing in {role}.

Your focus is on analyzing the stock of {stock_ticker}.

Your goal is to gain insights related to your specialization, focusing on the future prospects of {stock_ticker} and whether it is a good investment in the coming days, weeks, or months.

Begin by introducing yourself using your name, and then ask a question that will help you delve deeper into your analysis.

Continue to ask questions to refine your understanding.

When you are satisfied with your understanding, conclude the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_question(state: InterviewState):
    """Node to generate a question"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    interview_stock_ticker = state["interview_stock_ticker"]  # Renamed

    # Generate question
    system_message = question_instructions.format(role=analyst.role, stock_ticker=interview_stock_ticker)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    messages.append(question)
    return {"messages": messages}

# Generate expert answer
answer_instructions = """You are an expert on {stock_ticker}.

The analyst specializes in {role}.

Your goal is to answer the question posed by the analyst.

Use the following context to inform your answer:

{context}

When answering questions, follow these guidelines:

1. Use only the information provided in the context.

2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. Provide clear and concise answers relevant to the analyst's specialization.

4. Include any relevant data or insights from the context.

5. For the Investment Recommendation analyst, synthesize the insights from other analyses to provide a clear investment recommendation for the coming days, weeks, or months."""

def generate_answer(state: InterviewState):
    """Node to answer a question"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]
    stock_ticker = state["interview_stock_ticker"]  # Renamed

    # Answer question
    system_message = answer_instructions.format(stock_ticker=stock_ticker, role=analyst.role, context=context[0])
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to messages
    messages.append(answer)
    return {"messages": messages}

def save_interview(state: InterviewState):
    """Save interviews"""

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"interview": interview}

def search_stock_data(state: InterviewState):
    """Retrieve stock data relevant to the analyst's question"""

    messages = state['messages']
    last_question = messages[-1].content if messages else ""

    # Prepare data retrieval based on analyst's role
    analyst = state["analyst"]
    stock_ticker = state['interview_stock_ticker'].upper()  # Renamed

    context_pieces = []

    try:
        import yfinance as yf

        stock = yf.Ticker(stock_ticker)

        # Get current price and basic info
        current_price = stock.info.get('currentPrice', 'N/A')
        market_cap = stock.info.get('marketCap', 'N/A')
        context_pieces.append(f"Current Price: ${current_price}")
        context_pieces.append(f"Market Cap: ${market_cap}")

        # Fetch analyst recommendations
        analyst_recommendations = stock.recommendations
        if analyst_recommendations is not None and not analyst_recommendations.empty:
            latest_recommendation = analyst_recommendations.iloc[-1]
            context_pieces.append(f"Latest Analyst Recommendation: {latest_recommendation['To Grade']}")
        else:
            context_pieces.append("No analyst recommendations available.")

        # Fetch historical data for technical analysis
        if 'technical' in analyst.role.lower():
            hist = stock.history(period="1mo")
            context_pieces.append("Historical Prices (last 1 month) retrieved for technical analysis.")

        # Fetch fundamental data
        if 'fundamental' in analyst.role.lower():
            financials = stock.financials
            context_pieces.append("Financial statements retrieved for fundamental analysis.")

        # Fetch sentiment data
        if 'sentiment' in analyst.role.lower():
            # Implement sentiment data retrieval (placeholder)
            context_pieces.append("Market sentiment data retrieved for analysis.")

        # Fetch industry data
        if 'industry' in analyst.role.lower():
            # Implement industry data retrieval (placeholder)
            context_pieces.append("Industry trends and competitor data retrieved for analysis.")

        # Fetch regulatory news
        if 'regulatory' in analyst.role.lower():
            # Implement regulatory news retrieval (placeholder)
            context_pieces.append("Regulatory news and updates retrieved for analysis.")

    except Exception as e:
        context_pieces.append(f'Error retrieving data: {e}')

    # Combine context
    context = "\n".join(context_pieces)

    # Append context as a message
    messages.append(HumanMessage(content=f"Retrieved data:\n{context}"))

    return {'context': [context], 'messages': messages}

def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer"""

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    # Check the number of expert answers
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2].content if len(messages) >= 2 else ""

    if "Thank you so much for your help" in last_question:
        return 'save_interview'
    return "ask_question"

# Write a summary (section of the final report) of the interview
section_writer_instructions = """You are an expert technical writer.

Your task is to create a short, easily digestible section of a report based on the interview transcript.

1. Analyze the content of the interview:

{interview}

2. Create a report structure using markdown formatting:

- Use ## for the section title (based on the analyst's role and focus)
- Use ### for sub-section headers

3. Write the report following this structure:

a. Title (## header)

b. Summary (### header)

c. Key Insights (### header)

d. Investment Considerations (### header)

e. Sources (### header)

4. Make your title engaging based on the focus area of the analyst:

{focus}

5. For the summary section:

- Provide background/context related to the focus area.

6. For Key Insights:

- Highlight the most important findings from the interview.

7. For Investment Considerations:

- Discuss potential future performance and risks based on the analysis.

8. In the Sources section:

- List any sources mentioned in the context.

9. Final review:

- Ensure the report follows the required structure.
- Include no preamble before the title of the report.
- Check that all guidelines have been followed.

10. For the Investment Recommendation analyst:

- Provide a clear recommendation on whether to invest in {stock_ticker} in the coming days, weeks, or months based on the aggregated insights.
- Highlight the rationale behind the recommendation."""

def write_section(state: InterviewState):
    """Node to write a section"""

    # Get state
    interview = state["interview"]
    analyst = state["analyst"]

    # Write section using the interview
    system_message = section_writer_instructions.format(interview=interview, focus=analyst.description, stock_ticker=state.get('interview_stock_ticker', 'UNKNOWN'))
    section = llm.invoke([SystemMessage(content=system_message)])

    # Append it to state
    sections = state.get("sections", [])
    sections.append(section.content)

    return {"sections": sections}

# Add nodes and edges
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_stock_data", search_stock_data)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_stock_data")
interview_builder.add_edge("search_stock_data", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

def initiate_all_interviews(state: ResearchGraphState):
    """Conditional edge to initiate all interviews via Send() API or return to create_analysts"""

    # Check if human feedback
    human_analyst_feedback = state.get('human_analyst_feedback', 'approve')
    if human_analyst_feedback.lower() != 'approve':
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        stock_ticker = state["stock_ticker"]
        return [Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"So you said you were analyzing {stock_ticker}?")],
            "interview_stock_ticker": stock_ticker,  # Renamed key
            "max_num_turns": 2,
            "sections": []
        }) for analyst in state["analysts"]]

# Write a report based on the interviews
report_writer_instructions = """You are a technical writer creating a report on the stock of {stock_ticker}.

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific aspect.
2. They wrote up their findings into a memo.

Your task:

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos.
4. Summarize the central points in each memo into a cohesive single narrative.
5. Focus on the future prospects of {stock_ticker} and provide an analysis of whether it may be a good investment in the coming days, weeks, or months.

To format your report:

1. Use markdown formatting.
2. Include no preamble for the report.
3. Use appropriate headings for each section.
4. Do not mention any analyst names in your report.
5. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
6. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
7. List your sources in order and do not repeat.
8. Include an **Investment Recommendation** section based on the analyses.

Here are the memos from your analysts to build your report from:

{context}"""

def write_report(state: ResearchGraphState):
    """Node to write the final report body"""

    # Validate 'stock_ticker' presence
    if 'stock_ticker' not in state:
        logging.error("The key 'stock_ticker' is missing from ResearchGraphState.")
        raise KeyError("The key 'stock_ticker' is missing from ResearchGraphState.")

    # Debugging
    logging.debug(f"write_report - stock_ticker: {state.get('stock_ticker')}")

    # Full set of sections
    sections = state["sections"]
    stock_ticker = state["stock_ticker"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join(sections)

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(stock_ticker=stock_ticker, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Write a report based upon these memos.")])
    return {"content": report.content}

# Write the introduction or conclusion
intro_conclusion_instructions = """You are a technical writer finishing a report on {stock_ticker}.

You will be given all of the sections of the report.

Your job is to write a crisp and compelling {section_type} section.

Include no preamble.

Target around 150 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing:

{formatted_str_sections}"""

def write_introduction(state: ResearchGraphState):
    """Node to write the introduction"""

    # Validate 'stock_ticker' presence
    if 'stock_ticker' not in state:
        logging.error("The key 'stock_ticker' is missing from ResearchGraphState.")
        raise KeyError("The key 'stock_ticker' is missing from ResearchGraphState.")

    # Full set of sections
    sections = state["sections"]
    stock_ticker = state["stock_ticker"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join(sections)

    # Summarize the sections into an introduction
    instructions = intro_conclusion_instructions.format(
        stock_ticker=stock_ticker,
        section_type="Introduction",
        formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke([SystemMessage(content=instructions)] + [HumanMessage(content="Write the report introduction.")])
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    """Node to write the conclusion"""

    # Validate 'stock_ticker' presence
    if 'stock_ticker' not in state:
        logging.error("The key 'stock_ticker' is missing from ResearchGraphState.")
        raise KeyError("The key 'stock_ticker' is missing from ResearchGraphState.")

    # Full set of sections
    sections = state["sections"]
    stock_ticker = state["stock_ticker"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join(sections)

    # Summarize the sections into a conclusion
    instructions = intro_conclusion_instructions.format(
        stock_ticker=stock_ticker,
        section_type="Conclusion",
        formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke([SystemMessage(content=instructions)] + [HumanMessage(content="Write the report conclusion.")])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """Combine all sections into the final report"""

    # Save full final report
    content = state["content"]
    introduction = state["introduction"]
    conclusion = state["conclusion"]

    final_report = "\n\n".join([introduction, content, conclusion])

    return {"final_report": final_report}

# Add nodes and edges
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# Logic
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
graph = builder.compile(interrupt_before=['human_feedback'])
