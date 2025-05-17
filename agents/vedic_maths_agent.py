from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import WolframAlphaQueryRun
from langchain_community.tools import WolframAlphaAPIWrapper
from langchain_community.tools import BaseTool
from langchain_tavily import TavilySearch 
from typing import Union, Any
import os
from dotenv import load_dotenv
load_dotenv()

WOLFRAM_API=os.environ("WOLFRAM_API_KEY")
TAVILY_API_KEY=os.environ("TAVILY_API_KEY")
GROQ_API_KEY=os.environ("GROQ_API_KEY")

### Initialising the tools
tavily_search = TavilySearch(api_key=TAVILY_API_KEY)
wolfram = WolframAlphaQueryRun(
    api_wrapper = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_API)
)
class VerifyCalcTool(BaseTool):
    name="verify_calculation"
    description=("Verify the correctness  of a basic arithemetic expression(provided as a string)"
               "using vedic mathematics techniques. return True if the claimed result matches, else false"  
    )             

    def _run(self,expression:str,claimed_result:str) -> float:
        computed =  eval(expression)

        return computed==claimed_result

    async def _arun(self, expression:str,claimed_result:str) -> float:
        return self._run(expression,claimed_result)

verify_tool = VerifyCalcTool()        
    
## Initializing LLM for the agent
llm = ChatGroq(
    model_name="qwen-qwq-32b",
    temperature=0.7,
    api_key=GROQ_API_KEY
)   

## Chat prompt 
prompt = ChatPromptTemplate.from_template('''
  You are a Vedic mathematics expert and educator.

For any user query:
1. If it’s a question about Vedic math fundamentals, provide a detailed, step‑by‑step explanation in clear, approachable language.
2. If it’s an arithmetic expression:
   a. Solve it step‑by‑step using Vedic techniques.
   b. At the end, invoke the tool `verify_vedic_calc` with the original expression and your final numeric answer.
   c. Report both your worked solution and the tool’s True/False verification result.
3. If the query is outside of Vedic mathematics or general math:
   a. Reply: “Sorry, I can’t help you with that.”
   b. Then share an exciting trivia or concept about Vedic mathematics.
   c. Finally say: “You can learn these cool things and many more on Walnut Excellence Education—enrol now!”

Always explain every step in detail and use easy‑to‑understand language.
Expression : {input}                       
''')

## Creating the chain
chain = llm.with_prompt(prompt).with_output_parser(StrOutputParser())

## tools 
tools = [
    Tool.from_function(
        func=wolfram.run,
        name="wolfram_alpha",
        description="Use for advanced symbolic or numeric computation"
    ),
    Tool.from_function(
        func=tavily_search.run,
        name="tavily_search",
        description="Use to get up-to-date web content or tutorials"
    ),
    Tool.from_function(
        func=verify_tool.run,
        name="verify_vedic_calc",
        description=verify_tool.description
    )

]

## Initializing the agent
agent = initialize_agent(
        tools=tools,
        llm=chain,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True
)

## Agent class
class VedicMathsAgent:
    def __init__(self,agent_executor:Any):
        self.agent = agent_executor
    def handle_query(self, query:str)->str:
        response,trace = self.agent.run_and_trace(query)
        return response,trace
    
def get_vedic_agent()->VedicMathsAgent:
    return VedicMathsAgent(agent)    











