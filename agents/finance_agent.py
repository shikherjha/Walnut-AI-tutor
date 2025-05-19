from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain.agents import Tool, initialize_agent, AgentType
from langgraph.graph import StateGraph, END
from langchain_community.tools import BaseTool
from typing import TypedDict, Any, Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class AgentState(TypedDict):
    input: str
    intent: str
    response: str

class FinancialCalculatorTool(BaseTool):
    name: str = "financial_calculator"
    description: str = "Calculate financial metrics like ROI, EMI, break-even point, etc."
    
    def _run(self, calculation_type: str, **kwargs) -> float:
        if calculation_type.lower() == "roi":
            net_profit = float(kwargs.get("net_profit", 0))
            cost = float(kwargs.get("cost", 1))
            return (net_profit / cost) * 100
        
        elif calculation_type.lower() == "emi":
            principal = float(kwargs.get("principal", 0))
            rate = float(kwargs.get("rate", 0)) / 12 / 100  
            tenure = float(kwargs.get("tenure", 0)) * 12  
            if rate == 0:
                return principal / tenure
            return principal * rate * (1 + rate)**tenure / ((1 + rate)**tenure - 1)
        
        elif calculation_type.lower() == "compound_interest":
            principal = float(kwargs.get("principal", 0))
            rate = float(kwargs.get("rate", 0)) / 100
            time = float(kwargs.get("time", 0))
            compounds_per_year = int(kwargs.get("compounds_per_year", 1))
            return principal * (1 + rate/compounds_per_year)**(compounds_per_year * time)
        
        else:
            return f"Calculation type '{calculation_type}' not supported."

def classify_query(state: AgentState) -> AgentState:
    query = state["input"].lower()

    # Expanded keywords for more precise matching
    calculation_keywords = ["calculate", "computation", "compute", "formula", "roi", "emi", "break-even", "projection", 
                          "interest", "compound interest", "simple interest", "mortgage", "loan", "payment", 
                          "amortization", "depreciation", "appreciation", "return", "yield", "profit", "margin", 
                          "discount", "present value", "future value", "npv", "irr", "how do i calculate", "how to calculate"]
    
    content_keywords = ["what is", "explain", "teach", "describe", "learn", "difference between", "how does", 
                       "define", "meaning of", "concept of", "understand", "overview", "introduction to"]
    
    validation_keywords = ["is my", "does this", "validate", "check", "review", "verify", "assess", 
                          "evaluate", "analyze", "good idea", "bad idea", "should i", "make sense"]

    # More precise classification with priority to calculation queries
    if any(kw in query for kw in calculation_keywords):
        state['intent'] = "tools"
    elif any(kw in query for kw in content_keywords):
        state['intent'] = "content"
    elif any(kw in query for kw in validation_keywords):
        state['intent'] = "validation"
    else:
        state['intent'] = "unsure"
    return state

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,  # Reduced temperature for more consistent outputs
    api_key=GROQ_API_KEY,
    max_tokens=1024,  # Limit token generation for faster responses
    request_timeout=10.0  # Set timeout to prevent long waits
)

financial_calculator_tool = FinancialCalculatorTool()

tools = [
    Tool.from_function(
        func=financial_calculator_tool.run,
        name="financial_calculator",
        description=financial_calculator_tool.description
    )
]

if TAVILY_API_KEY:
    try:
        tavily_search = TavilySearch(api_key=TAVILY_API_KEY)
        tools.append(Tool.from_function(
            func=tavily_search.run,
            name="tavily_search",
            description="Search the web for financial information and trends"
        ))
    except Exception:
        pass

# Enhanced prompt with more financial knowledge embedded
content_teaching_prompt = ChatPromptTemplate.from_template('''
You are a Finance and Entrepreneurship professor with comprehensive knowledge of financial concepts.

Query: {input}

Provide a comprehensive explanation including:
1. Clear definition with technical accuracy
2. Practical applications in business and personal finance
3. Real-world examples from well-known companies or scenarios
4. Best practices and implementation strategies
5. Common pitfalls to avoid

Your explanation should cover basic principles, intermediate concepts, and advanced applications when relevant.
Use simple language suitable for beginners but don't shy away from important terminology.

Include numerical examples where appropriate to illustrate the concept in practice.
''')

# Enhanced calculation prompt
tool_prompt = ChatPromptTemplate.from_template('''
You are a finance expert with deep knowledge of financial calculations and formulas.

Query: {input}

First, identify exactly what needs to be calculated:
1. Determine the specific financial metric or value requested
2. Identify all required input variables 
3. Provide the formula that should be applied
4. Perform the calculation step-by-step
5. Interpret the result in practical business terms

For scenarios where external data would be required in real applications, use reasonable assumptions and clearly state them.

If you encounter complex calculations beyond basic arithmetic, break them down into manageable steps.
Include alternative approaches when multiple methods exist.
''')

# Enhanced validation prompt
validation_prompt = ChatPromptTemplate.from_template('''
You are a seasoned business consultant with expertise in evaluating business models and financial strategies.

Query: {input}

Analyze this business idea or strategy by addressing:

1. Market viability and competitive landscape
2. Financial feasibility (startup costs, ROI potential, breakeven timeline)
3. Operational considerations and scalability
4. Key risks and mitigation strategies
5. Strategic recommendations for implementation or improvement

Base your analysis on established business principles and known market trends.
Include both qualitative assessment and quantitative considerations where relevant.
Provide actionable recommendations that are specific and practical.
''')

unsure_prompt = ChatPromptTemplate.from_template('''
Query: {input}

I'd like to help you with your finance or entrepreneurship question. To provide the most accurate and helpful response, could you please clarify:

1. Are you looking for a concept explanation or definition?
2. Do you need specific financial calculations or analyses?
3. Are you seeking validation or feedback on a business idea or strategy?

While waiting for your clarification, here's a relevant insight about finance or entrepreneurship:
[Insert a brief, general tip related to keywords in their query]

I can provide more specific assistance once you clarify your needs.
''')

def tool_node(state: AgentState) -> AgentState:
    query = state["input"]
    
    # Direct handling of common calculation types without using agent framework
    query_lower = query.lower()
    
    # Check for compound interest calculation specifically
    if "compound interest" in query_lower and ("calculate" in query_lower or "how" in query_lower):
        response = """
To calculate compound interest:

Formula: A = P(1 + r/n)^(nt)

Where:
- A = Final amount including interest
- P = Principal (initial investment)
- r = Annual interest rate (as a decimal)
- n = Number of times interest is compounded per year
- t = Time in years

Example: For $1,000 invested for 5 years at 5% interest compounded quarterly:
- P = $1,000
- r = 0.05
- n = 4 (quarterly)
- t = 5 years

A = $1,000(1 + 0.05/4)^(4×5)
A = $1,000(1 + 0.0125)^20
A = $1,000(1.0125)^20
A = $1,000 × 1.2824
A = $1,282.40

The compound interest earned would be $1,282.40 - $1,000 = $282.40

You can adjust these values based on your specific scenario.
"""
        state['response'] = response
        return state
    
    # Handle other common financial calculations directly
    elif "roi" in query_lower and ("calculate" in query_lower or "how" in query_lower):
        response = """
To calculate Return on Investment (ROI):

Formula: ROI = (Net Profit / Cost of Investment) × 100%

Example: If you invested $1,000 and earned $1,250:
- Net Profit = $1,250 - $1,000 = $250
- Cost of Investment = $1,000
- ROI = ($250 / $1,000) × 100% = 25%

This means your investment generated a 25% return.
"""
        state['response'] = response
        return state
    
    # Try using the specialized tools if no direct match
    try:
        # Use financial calculator tool directly if possible
        if financial_calculator_tool and any(term in query_lower for term in ["roi", "emi", "compound interest"]):
            calculation_type = ""
            params = {}
            
            if "roi" in query_lower:
                calculation_type = "roi"
                # Extract parameters if present in query (simplified extraction)
                # In real code, would need more sophisticated parameter extraction
                params = {"net_profit": 250, "cost": 1000}  # Default example values
                
            elif "emi" in query_lower:
                calculation_type = "emi"
                params = {"principal": 100000, "rate": 5, "tenure": 5}  # Default example values
                
            elif "compound interest" in query_lower:
                calculation_type = "compound_interest"
                params = {"principal": 1000, "rate": 5, "time": 5, "compounds_per_year": 1}  # Default example values
            
            if calculation_type:
                try:
                    result = financial_calculator_tool._run(calculation_type, **params)
                    response = f"For a basic {calculation_type} calculation with standard values: {result}\n\n"
                    response += f"To calculate your specific scenario, use the formula and replace with your actual values."
                    state['response'] = response
                    return state
                except Exception:
                    pass  # Fall through to next approach if direct calculation fails
        
        # Fall back to agent if direct calculation doesn't work
        agent = initialize_agent(
            llm=llm,
            tools=tools,
            agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
        response = agent.run(query)
        state['response'] = response
    except Exception as e:
        # Final fallback to template-based response
        chain = tool_prompt | llm | StrOutputParser()
        response = chain.invoke({"input": query})
        state['response'] = response
    
    return state

def content_node(state: AgentState) -> AgentState:
    query = state["input"]
    try:
        chain = content_teaching_prompt | llm | StrOutputParser()
        response = chain.invoke({"input": query})
        state['response'] = response
    except Exception as e:
        state['response'] = f"I'm having trouble with the detailed explanation. Here's what I can tell you: {query} is an important financial concept that impacts both business operations and investment strategies. It involves analyzing monetary flows and making informed decisions based on financial data."
    return state

def validation_node(state: AgentState) -> AgentState:
    query = state["input"]
    try:
        chain = validation_prompt | llm | StrOutputParser()
        response = chain.invoke({"input": query})
        state['response'] = response
    except Exception as e:
        state['response'] = f"To properly evaluate your business idea, I need to consider market potential, financial requirements, operational challenges, and strategic fit. Based on limited information, I'd suggest focusing on validating market demand and creating a detailed financial projection before proceeding."
    return state

def unsure_node(state: AgentState) -> AgentState:
    query = state["input"]
    try:
        chain = unsure_prompt | llm | StrOutputParser()
        response = chain.invoke({"input": query})
        state['response'] = response
    except Exception as e:
        state['response'] = "Could you please rephrase your question with more details? I'd like to help with your finance or entrepreneurship inquiry but need more context about what specific information or assistance you're seeking."
    return state

builder = StateGraph(AgentState)

builder.add_node("classify", classify_query)
builder.add_node("tool", tool_node)
builder.add_node("content", content_node)
builder.add_node("validation", validation_node)
builder.add_node("unsure", unsure_node)

builder.set_entry_point("classify")

builder.add_conditional_edges("classify", lambda state: state['intent'], {
    "tools": "tool",
    "content": "content", 
    "validation": "validation",
    "unsure": "unsure"
})

builder.add_edge("tool", END)
builder.add_edge("content", END)
builder.add_edge("validation", END)
builder.add_edge("unsure", END)

graph = builder.compile()

class FinanceEntrEduAgent:
    def __init__(self, graph: Any):
        self.graph = graph
        # Cache for common financial queries
        self.response_cache = {
            "how do i calculate compound interest": """
The compound interest formula is A = P(1 + r/n)^(nt), where:
- A is the future value of the investment
- P is the principal amount
- r is the annual interest rate (as a decimal)
- n is the number of times interest is compounded per year
- t is the time in years

To calculate compound interest:
1. Identify your principal amount (P)
2. Convert your annual interest rate to decimal (r)
3. Determine compounding frequency per year (n)
4. Specify investment time period in years (t)
5. Use the formula A = P(1 + r/n)^(nt)
6. Subtract P from A to get just the interest earned

Example: $1,000 invested for 5 years at 5% compounded annually
A = $1,000(1 + 0.05/1)^(1×5)
A = $1,000(1.05)^5
A = $1,000 × 1.276
A = $1,276

The compound interest earned would be $1,276 - $1,000 = $276
""",
            "what is compound interest": """
Compound interest is interest calculated on the initial principal and also on the accumulated interest from previous periods. It's essentially "interest on interest."

Key characteristics:
1. Growth accelerates over time (exponential growth)
2. Affected by compounding frequency (more frequent compounding yields higher returns)
3. Best for long-term investments due to the snowball effect

Formula: A = P(1 + r/n)^(nt)

Compound interest is fundamental to wealth building, retirement planning, and understanding the true cost of loans. It's what makes investments grow significantly over time and why starting to invest early is so advantageous.

Example: $10,000 invested at 5% annual interest:
- Simple interest after 10 years: $5,000 ($500 per year)
- Compound interest after 10 years: $6,289 (additional $1,289 from compounding)
"""
        }

    def handle_query(self, query: str) -> Tuple[str, Any]:
        # Check cache first for instant responses
        normalized_query = query.lower().strip()
        if normalized_query in self.response_cache:
            return self.response_cache[normalized_query], {"input": query, "from_cache": True}
        
        # Common calculation patterns with direct responses
        if "compound interest" in normalized_query and ("calculate" in normalized_query or "how" in normalized_query):
            return self.response_cache["how do i calculate compound interest"], {"input": query, "pattern_match": True}
            
        # Continue with normal flow for non-cached queries
        initial_state = {
            "input": query,
            "intent": "",
            "response": ""
        }
        try:
            result = self.graph.invoke(initial_state)
            return result['response'], result
        except Exception as e:
            fallback = f"I can help with your question about '{query}'. Finance and entrepreneurship involve balancing risk and reward, managing resources effectively, and making strategic decisions based on market conditions and data analysis."
            return fallback, {"error": str(e), "input": query}

def get_finance_agent() -> FinanceEntrEduAgent:
    return FinanceEntrEduAgent(graph)