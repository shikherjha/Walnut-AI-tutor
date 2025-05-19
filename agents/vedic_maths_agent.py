from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import BaseTool
from langchain_tavily import TavilySearch
from typing import Union, Any, Type
from pydantic import BaseModel, Field
import os
import re
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

### Initializing the search tool - retaining only Tavily for efficiency
tavily_search = TavilySearch(api_key=TAVILY_API_KEY)

class VerifyCalcInput(BaseModel):
    query: str = Field(description="Format: 'expression=result' e.g., '25*43=1075'")

class VerifyCalcTool(BaseTool):
    name: str = "verify_calculation"
    description: str = (
        "Verify the correctness of a basic arithmetic expression. "
        "Input format should be 'expression=claimed_result', for example: '25*43=1075' "
        "Returns True if the claimed result matches the actual calculation, else False"
    )
    args_schema: Type[BaseModel] = VerifyCalcInput

    def _run(self, query: str) -> bool:
        try:
            # Parse the input string to extract expression and claimed result
            if '=' not in query:
                return False
            
            expression, claimed_result_str = query.split('=', 1)
            expression = expression.strip()
            claimed_result_str = claimed_result_str.strip()
            
            # Safely evaluate the expression
            computed = eval(expression)
            
            # Convert claimed_result to the same type for comparison
            try:
                if isinstance(computed, int):
                    claimed = int(float(claimed_result_str))
                else:
                    claimed = float(claimed_result_str)
                return computed == claimed
            except ValueError:
                return False
                
        except Exception as e:
            return False

    async def _arun(self, query: str) -> bool:
        return self._run(query)

verify_tool = VerifyCalcTool()

# Enhanced Vedic technique reference tool with more examples and clearer explanations
class VedicTechniqueInput(BaseModel):
    technique: str = Field(description="Name of the Vedic math technique to get information about")

class VedicTechniqueTool(BaseTool):
    name: str = "vedic_technique_reference"
    description: str = (
        "Get detailed information about a specific Vedic mathematics technique. "
        "Input should be the name of the technique, like 'Nikhilam', 'Urdhva Tiryagbhyam', etc."
    )
    args_schema: Type[BaseModel] = VedicTechniqueInput

    def _run(self, technique: str) -> str:
        # Dictionary of key Vedic techniques with detailed explanations
        techniques = {
            "nikhilam": """
                Nikhilam Navatashcaramam Dashatah (All from 9 and the last from 10):
                This technique is particularly useful for multiplication when numbers are close to power bases (like 10, 100, 1000).
                
                Method:
                1. Find the deviation of each number from the nearest power base
                2. Cross-add/subtract to get the first part of the answer
                3. Multiply the deviations to get the second part
                
                Example for 98 × 97:
                Base = 100
                Deviations: 98 = -2, 97 = -3
                First part: 98-3 = 95 (or 97-2 = 95)
                Second part: (-2)×(-3) = 6
                Answer: 9506
                
                Example for 108 × 107:
                Base = 100
                Deviations: 108 = +8, 107 = +7
                First part: 108+7 = 115 (or 107+8 = 115)
                Second part: (+8)×(+7) = 56
                Answer: 11556
                
                Example for 993 × 997:
                Base = 1000
                Deviations: 993 = -7, 997 = -3
                First part: 993-3 = 990 (or 997-7 = 990)
                Second part: (-7)×(-3) = 21
                Answer: 990021
            """,
            
            "urdhva tiryagbhyam": """
                Urdhva Tiryagbhyam (Vertically and Crosswise):
                A general multiplication technique that divides the calculation into smaller parts.
                
                Method:
                1. Arrange numbers vertically
                2. Multiply vertically and crosswise, starting from the rightmost digits
                3. Add the products in each step, carrying over as needed
                
                Example for 34 × 12:
                Step 1: 4×2 = 8
                Step 2: (3×2)+(4×1) = 6+4 = 10, write 0, carry 1
                Step 3: 3×1+1(carry) = 4
                Answer: 408
                
                Example for 85 × 96:
                Step 1: 5×6 = 30, write 0, carry 3
                Step 2: (8×6)+(5×9)+3(carry) = 48+45+3 = 96, write 6, carry 9
                Step 3: 8×9+9(carry) = 72+9 = 81
                Answer: 8160
                
                Example for 123 × 456:
                Let's break it down step by step:
                Step 1: 3×6 = 18, write 8, carry 1
                Step 2: (2×6)+(3×5)+1(carry) = 12+15+1 = 28, write 8, carry 2
                Step 3: (1×6)+(2×5)+(3×4)+2(carry) = 6+10+12+2 = 30, write 0, carry 3
                Step 4: (1×5)+(2×4)+3(carry) = 5+8+3 = 16, write 6, carry 1
                Step 5: 1×4+1(carry) = 5
                Answer: 56088
            """,
            
            "ekadhikena purvena": """
                Ekadhikena Purvena (One More than the Previous):
                A technique for finding squares of numbers ending in 5 and for special divisions.
                
                For squares of numbers ending in 5:
                1. Take the number without the last digit
                2. Multiply it by one more than itself
                3. Append 25 to the result
                
                Example for 35²:
                First part: 3×4 = 12
                Final result: 1225
                
                Example for 75²:
                First part: 7×8 = 56
                Final result: 5625
                
                Example for 125²:
                First part: 12×13 = 156
                Final result: 15625
                
                For division problems where divisor is close to a power base:
                1. Find complement of divisor from power base
                2. Multiply dividend repeatedly by this complement
                3. Adjust place values as needed
                
                Example for 4256 ÷ 98:
                Divisor 98 is close to 100 with complement 2
                4256 × 2 = 8512, but adjustment needed for place value
                Take 8512 ÷ 100 = 85.12, carry 12 remainder
                Continue: 12 × 2 = 24, add to new dividend: 24
                24 ÷ 100 = 0.24, remainder 24
                Final result: 43 + 0.42 + 0.0024 = 43.4224 (actually 43.43 when rounded)
            """,
            
            "antyayordasake'pi": """
                Antyayordasake'pi (Last digits sum to 10):
                For multiplying numbers whose last digits sum to 10 and other digits are the same.
                
                Method:
                1. Multiply the first digits by one more than themselves
                2. Multiply the last digits
                
                Example for 43 × 47:
                First part: 4×5 = 20
                Second part: 3×7 = 21
                Answer: 2021
                
                Example for 76 × 74:
                First part: 7×8 = 56
                Second part: 6×4 = 24
                Answer: 5624
                
                Example for 85 × 85:
                This is a square, so use: 8×9 = 72, 5×5 = 25
                Answer: 7225
            """,
            
            "anurupyena": """
                Anurupyena (Proportionality):
                A technique for solving proportions and ratios quickly.
                
                Method:
                1. Identify the proportional relationship
                2. Apply cross-multiplication principles
                3. Simplify using Vedic math techniques
                
                Example 1:
                If 4 books cost $60, how much do 7 books cost?
                Using proportionality: 4 : 60 :: 7 : x
                x = (60 × 7) ÷ 4 = 420 ÷ 4 = 105
                Result: $105
                
                Example 2:
                If 8 workers can complete a job in 12 days, how many workers would be needed to complete it in 6 days?
                Using inverse proportion: 8 × 12 = x × 6
                x = (8 × 12) ÷ 6 = 96 ÷ 6 = 16
                Result: 16 workers
            """,
            
            "vinculum": """
                Vinculum:
                A concept of representing numbers close to power bases using negative numbers.
                
                Example 1:
                98 can be written as 100-2 or 10²-2
                This simplifies calculations involving numbers close to power bases.
                
                Example 2:
                For calculating 998 × 997:
                Express as 1000-2 and 1000-3
                Apply Nikhilam: (1000-2)(1000-3) = 1000²-1000×5+6 = 1000000-5000+6 = 995006
                
                Example 3:
                For division like 1996 ÷ 96:
                Express as 1996 ÷ (100-4)
                Apply Paravartya Yojayet to simplify the division process
            """,
            
            "ekanyunena purvena": """
                Ekanyunena Purvena (One Less than the Previous):
                Useful for finding certain types of quotients and products.
                
                Method:
                1. Identify if the number is one less than a power base
                2. Apply special formulas based on the context
                
                Example 1:
                For 9999 ÷ 9:
                Since 9999 = 10000-1, and 9 = 10-1
                Result is 1111 (using pattern recognition)
                
                Example 2:
                For 999 × 999:
                Express as (1000-1)(1000-1) = 1000²-2000+1 = 998001
            """,
            
            "paravartya yojayet": """
                Paravartya Yojayet (Transpose and Apply):
                A powerful technique for division that transforms the problem into simpler calculations.
                
                Method:
                1. Express the divisor as a number close to a power base (10, 100, etc.)
                2. Find its deviation from that base
                3. Divide the first digit(s) of the dividend by the first digit(s) of the divisor
                4. Apply a special formula involving the deviation to adjust the result
                5. Continue the process for remaining digits
                
                Example 1 - Basic Division: 825 ÷ 11
                Step 1: Express 11 as (10+1), with deviation +1 from base 10
                Step 2: First digit of dividend is 8, so initial quotient digit is 8÷1 = 8
                Step 3: New dividend: 8 - (8×1) = 0, bring down 2, so new dividend is 2
                Step 4: 2÷1 = 2, new dividend: 2 - (2×1) = 0, bring down 5, so new dividend is 5
                Step 5: 5÷1 = 5, check: 5 - (5×1) = 0, no remainder
                Result: 75
                
                Example 2 - More Complex: 4961 ÷ 19
                Step 1: Express 19 as (20-1), with deviation -1 from base 20
                Step 2: First two digits of dividend are 49, so initial quotient digit is 49÷2 = 24.5, but we take 24
                Step 3: New dividend: 49 - (24×2) = 1, adjust for deviation: 1 - (24×(-1)) = 1+24 = 25
                Step 4: Bring down 6, so new dividend is 256
                Step 5: 256÷2 = 128, but we take 12 (next quotient digit)
                Step 6: New dividend: 256 - (12×20) = 16, adjust for deviation: 16 - (12×(-1)) = 16+12 = 28
                Step 7: Bring down 1, so new dividend is 281
                Step 8: 281÷2 = 140.5, but we take 14 as the next quotient digit
                Step 9: Check: 281 - (14×20) = 1, adjust for deviation: 1 - (14×(-1)) = 1+14 = 15
                Final result: 261 with remainder 1
                
                Example 3 - Division by 9: 6372 ÷ 9
                Step 1: Express 9 as (10-1), with deviation -1 from base 10
                Step 2: First digit of dividend is 6, so initial quotient digit is 6÷1 = 6
                Step 3: New dividend: 6 - (6×1) = 0, adjust for deviation: 0 - (6×(-1)) = 0+6 = 6
                Step 4: Bring down 3, so new dividend is 63
                Step 5: 63÷1 = 63, but take 7 as the next quotient digit
                Step 6: New dividend: 63 - (7×1) = 56, adjust for deviation: 56 - (7×(-1)) = 56+7 = 63
                Step 7: Bring down 7, so new dividend is 637
                Step 8: 637÷1 = 637, but take 7 as the next quotient digit
                Step 9: New dividend: 637 - (7×1) = 630, adjust for deviation: 630 - (7×(-1)) = 630+7 = 637
                Step 10: Bring down 2, so new dividend is 6372
                Step 11: 6372÷1 = 6372, but take 8 as the next quotient digit
                Step 12: Check: 6372 - (8×1) = 6364, adjust for deviation: 6364 - (8×(-1)) = 6364+8 = 6372
                Result: 708 with no remainder
                
                This technique is extremely powerful once mastered, as it transforms complex division into a series of simple multiplications and subtractions.
            """,
            
            "yavadunam": """
                Yavadunam (Whatever the Extent of its Deficiency):
                Used for quick squaring of numbers close to power bases.
                
                Method:
                1. Find the deviation from the nearest power base
                2. Subtract the deviation from the number to get first part
                3. Square the deviation for the second part
                
                Example for 98²:
                Deviation from 100: -2
                First part: 98-2 = 96
                Second part: (-2)² = 4
                Answer: 9604
                
                Example for 106²:
                Deviation from 100: +6
                First part: 106+6 = 112
                Second part: (+6)² = 36
                Answer: 11236
                
                Example for 992²:
                Deviation from 1000: -8
                First part: 992-8 = 984
                Second part: (-8)² = 64
                Answer: 984064
            """,
            
            "digital roots": """
                Digital Root Method:
                Used for quick verification of results.
                
                Method:
                1. Find digital root by adding all digits until a single digit remains
                2. Apply operation on digital roots
                3. Compare with digital root of result
                
                Example: 
                For 57 × 83 = 4731
                Digital root of 57: 5+7=12, 1+2=3
                Digital root of 83: 8+3=11, 1+1=2
                3×2=6
                Digital root of 4731: 4+7+3+1=15, 1+5=6
                The match confirms the result may be correct.
                
                Example:
                For 1343 + 2857 = 4200
                Digital root of 1343: 1+3+4+3=11, 1+1=2
                Digital root of 2857: 2+8+5+7=22, 2+2=4
                2+4=6
                Digital root of 4200: 4+2+0+0=6
                The match confirms the result may be correct.
                
                Example:
                For 625 ÷ 25 = 26
                Digital root of 625: 6+2+5=13, 1+3=4
                Digital root of 25: 2+5=7
                Digital root of 26: 2+6=8
                To verify: if 7×8=56, 5+6=11, 1+1=2, then doubled (for division) = 4
                Match confirmed.
            """,
            
            "dwandwa yoga": """
                Dwandwa Yoga (Duplex combination):
                A method for squaring numbers, especially effective for certain types of numbers.
                
                Method:
                1. Separate the number into parts
                2. Apply special formulas based on the duplex relationship
                3. Combine results appropriately
                
                Example for two-digit numbers:
                For 46²:
                Duplex of 46 = 4×6 = 24
                46² = (40+6)² = 40² + 2(40×6) + 6² = 1600 + 480 + 36 = 2116
                
                Using Dwandwa: 
                - First part: 4² = 16 (put in hundreds place)
                - Middle part: Duplex × 2 = 24 × 2 = 48 (put in tens place)
                - Last part: 6² = 36
                Result: 2116
                
                Example for 52²:
                Duplex of 52 = 5×2 = 10
                - First part: 5² = 25 (put in hundreds place)
                - Middle part: Duplex × 2 = 10 × 2 = 20 (put in tens place)
                - Last part: 2² = 4
                Result: 2704
            """,
            
            "shunya": """
                Shunya (Zero):
                Special techniques for handling calculations with zeros.
                
                Includes methods for:
                1. Multiplication with zeros in specific positions
                2. Division when zeros are present
                3. Simplifying expressions with zeros
                
                Example 1 - Multiplication with trailing zeros:
                For 400 × 500:
                Ignore zeros first: 4 × 5 = 20
                Count total zeros: 2 from 400 and 2 from 500 = 4 zeros
                Result: 200000 (20 followed by 4 zeros)
                
                Example 2 - Division with zeros:
                For 84000 ÷ 400:
                Eliminate common zeros: 840 ÷ 4 = 210
                
                Example 3 - Quick multiplication with internal zeros:
                For 102 × 103:
                First digits: 1×1 = 1
                Last digits: 2×3 = 6
                Middle: (1+2)×(1+3) - 1×1 - 2×3 = 3×4 - 1 - 6 = 12 - 7 = 5
                Result: 10506
            """,
            
            "vilokanam": """
                Vilokanam (Observation):
                A general approach to solve problems by observation and pattern recognition.
                
                Method:
                1. Observe the special properties or patterns in the numbers
                2. Apply shortcuts based on these observations
                3. Verify the result
                
                Example 1 - Multiplication with pattern recognition:
                For 11 × 11 = 121
                For 111 × 111 = 12321
                For 1111 × 1111 = 1234321
                Pattern: Each digit appears according to the Pascal's triangle pattern
                
                Example 2 - Quick division by observation:
                For 1089 ÷ 11:
                Observe the pattern of remainders when divided by 11
                Quick solution: 99
                
                Example 3 - Finding special square roots:
                For √2025:
                Observe that 2025 = 45² because the first two digits (20) is close to 4×5=20, and last two digits form a perfect square (25)
                Result: 45
            """,
            
            "anurupya sutra": """
                Anurupya Sutra (Specific proportionality):
                Used for solving specific types of proportions quickly.
                
                Method:
                1. Identify the proportional relationship
                2. Apply the specific formula for that relationship
                3. Calculate the result directly
                
                Example 1 - Simple proportion:
                If 8 items cost $24, how much do 12 items cost?
                Solution: (24 × 12) ÷ 8 = 36
                
                Example 2 - Compound proportion:
                If 8 machines can produce 96 items in 12 hours, how many items can 10 machines produce in 15 hours?
                Solution: 96 × (10÷8) × (15÷12) = 96 × 1.25 × 1.25 = 150
                
                Example 3 - Investment proportion:
                If $800 invested for 3 years gives $96 interest, what will $1200 give in 4 years at the same rate?
                Solution: 96 × (1200÷800) × (4÷3) = 96 × 1.5 × 1.33 = 192
            """,
            
            "division": """
                Vedic Division Techniques:
                Division in Vedic mathematics uses multiple approaches depending on the divisor.
                
                Technique 1: Paravartya Yojayet (Transpose and Apply)
                Best for divisors close to power bases.
                
                Example: 4532 ÷ 19
                Express 19 as (20-1)
                Step 1: 45÷2 = 22.5, take 22 as quotient digit
                Step 2: Remainder: 45-(22×2) = 1, adjust: 1-(22×(-1)) = 23
                Step 3: Bring down 3: 233
                Step 4: 233÷2 = 116.5, take 11 as next quotient digit
                Step 5: Remainder: 233-(11×20) = 13, adjust: 13-(11×(-1)) = 24
                Step 6: Bring down 2: 242
                Step 7: 242÷2 = 121, take 12 as final quotient digit
                Step 8: Remainder: 242-(12×20) = 2, adjust: 2-(12×(-1)) = 14
                Solution: 238 with remainder 14
                
                Technique 2: Division by Number Close to 10
                Example: 6843 ÷ 9
                Step 1: Express 9 as (10-1)
                Step 2: Divide first digit by first part of divisor: 6÷1 = 6
                Step 3: Subtract: 6-(6×1) = 0, adjust: 0-(6×(-1)) = 6 
                Step 4: Bring down 8: new dividend 68
                Step 5: 68÷1 = 68, take 7 (we're using an estimation technique)
                Step 6: 68-(7×1) = 61, adjust: 61-(7×(-1)) = 68
                Step 7: Bring down 4: new dividend 684
                Step 8: 684÷1 = 684, take 6
                Step 9: 684-(6×1) = 678, adjust: 678-(6×(-1)) = 684 
                Step 10: Bring down 3: new dividend 6843
                Step 11: 6843÷1 = 6843, take 9
                Step 12: 6843-(9×1) = 6834, adjust: 6834-(9×(-1)) = 6843
                Result: 760.33...
                
                Technique 3: Simple Division by Base Minus Small Number
                Example: 759 ÷ 9
                Since 9 = 10-1, we can use a straightforward method:
                Step 1: Take each digit and add the previous quotient digit
                Step 2: 7÷1 = 7 (first quotient digit)
                Step 3: (5+7)÷1 = 12, take 2, carry 1
                Step 4: (9+1+2)÷1 = 12, take 2, carry 1
                Step 5: Since there are no more digits, the quotient is 84 with remainder 3
                Check: 84×9 = 756, 759-756 = 3
                
                Technique 4: Division by Factors
                Example: 936 ÷ 24
                Break 24 into 8×3
                Step 1: 936÷8 = 117
                Step 2: 117÷3 = 39
                Result: 39
                
                These techniques drastically reduce complex division to manageable, mental calculations.
            """
        }
        
        # Normalize input by converting to lowercase and removing spaces
        norm_technique = technique.lower().strip()
        
        # Special case for division (many users ask about this)
        if "division" in norm_technique:
            return techniques["division"]
        
        # Try to find exact match first
        if norm_technique in techniques:
            return techniques[norm_technique]
        
        # Try to find partial match
        for key in techniques:
            if norm_technique in key or key in norm_technique:
                return techniques[key]
        
        return "Technique not found in the reference. Please check the spelling or try a different technique name."

    async def _arun(self, technique: str) -> str:
        return self._run(technique)

vedic_technique_tool = VedicTechniqueTool()
    
## Initializing LLM for the agent
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5,  # Reduced temperature for more focused responses
    api_key=GROQ_API_KEY
)   

## Enhanced Chat prompt with detailed instructions for better teaching
prompt = ChatPromptTemplate.from_template('''
You are a Vedic mathematics expert and passionate teacher with decades of experience. Your goal is to make Vedic math accessible, exciting, and practical for learners of all levels.

When responding to queries:

1. For questions about Vedic math principles or techniques:
   a. Begin with a brief, clear definition of the concept
   b. Explain the underlying principle in simple language
   c. Provide a step-by-step breakdown of how to apply the technique
   d. Include 2-3 progressively challenging examples with complete solutions
   e. Highlight common mistakes students make and how to avoid them
   f. End with practical applications or tips for practice

2. For calculation problems:
   a. First identify which Vedic technique(s) would be most efficient
   b. Explain your reasoning for choosing this approach
   c. Walk through the solution step-by-step with clear explanations for each step
   d. Show the traditional method briefly for comparison if relevant
   e. Use the verify_calculation tool with format 'expression=your_answer' at the end
   f. If verification fails, double-check your work and explain any corrections

3. For the 16 Vedic Sutras or specific techniques:
   a. Provide the Sanskrit name and translation
   b. Explain the core principle in simple terms
   c. Give specific examples of when and how to apply it
   d. Use visual explanations with alignment of numbers when helpful
   e. Connect it to related mathematical concepts students might already know

TEACHING STYLE GUIDELINES:
- Use clear, engaging language appropriate for the apparent level of the student
- Break down complex concepts into digestible chunks
- Always validate calculations using appropriate tools or cross-checking
- Explain WHY a technique works, not just HOW to use it
- Use analogies and real-world applications where possible
- Be enthusiastic about the elegance and efficiency of Vedic techniques
- Format your responses with appropriate spacing and formatting for clarity

At the end of your response, include a brief note about another related Vedic math concept the student might find interesting.

Expression: {input}
''')

chain = prompt | llm | StrOutputParser()

## Updated tools - removed Wolfram Alpha for faster response
tools = [
    Tool.from_function(
        func=tavily_search.run,
        name="tavily_search",
        description="Use to get up-to-date web content or tutorials on Vedic mathematics"
    ),
    verify_tool,
    vedic_technique_tool
]

## Initializing the agent with enhanced system prompt
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    early_stopping_method="generate",
    max_iterations=3  # Limit iterations for faster response
)



## Agent class with improved error handling
class VedicMathsAgent:
    def __init__(self, agent_executor: Any):
        self.agent = agent_executor
        self.vedic_techniques = {
            "ekadhikena purvena": "One more than the previous one",
            "nikhilam": "All from 9 and the last from 10",
            "urdhva tiryagbhyam": "Vertically and crosswise",
            "paravartya yojayet": "Transpose and adjust",
            "shunyam samyasamuccaye": "When sum is same that sum is zero",
            "anurupyena": "By proportion",
            "sankalana vyavakalanabhyam": "By addition and subtraction",
            "puranapuranabhyam": "By completion or non-completion",
            "chalana kalanabyham": "By using operations",
            "yaavadunam": "Whatever the extent of its deficiency",
            "vyashtisamanstih": "Part and whole",
            "shesanyankena charamena": "The remainders by the last digit",
            "sopaantyadvayamantyam": "The ultimate and twice the penultimate",
            "ekanyunena purvena": "One less than the previous",
            "gunitasamuchyah": "The product of sum is equal to sum of product",
            "gunakasamuchyah": "Factors of sum is equal to sum of factors"
        }
        
    def handle_query(self, query: str) -> tuple[str, Any]:
        # Check if query is about one of the 16 sutras by simple keyword matching
        if "16 sutras" in query.lower() or "16 vedic sutras" in query.lower() or "sutras" in query.lower():
            sutras_response = self._generate_sutras_response()
            return sutras_response, []
            
        # Check if it's specifically asking about a technique
        for technique, meaning in self.vedic_techniques.items():
            if technique.lower() in query.lower():
                # Use the vedic_technique_tool directly
                technique_info = vedic_technique_tool._run(technique)
                # Add a formatted response
                formatted_response = f"""
# {technique.title()} ({meaning})

{technique_info}

I hope this explanation helps! Let me know if you'd like more examples or have questions about applying this technique.

**Related Concept:** You might also be interested in exploring the relationship between {technique} and {self._get_related_technique(technique)}.
"""
                return formatted_response, []
                
        # Regular agent processing for other queries
        try:
            result = self.agent.invoke({"input": query})
            response = result.get("output", "")
            trace = result.get("intermediate_steps", [])
            
            # If response is too brief, enrich it
            if len(response.split()) < 50:
                response = self._enrich_response(query, response)
                
            return response, trace
            
        except Exception as e:
            # Fallback response using direct chain instead of agent
            try:
                response = chain.invoke({"input": query})
                return response, []
            except:
                return f"I apologize for the technical difficulty. Let me answer your question about {query} directly:\n\n{self._generate_fallback_response(query)}", []
    
    def _enrich_response(self, query, original_response):
        """Add more educational content to brief responses"""
        if "calculation" in query.lower() or any(op in query for op in ['+', '-', '*', '/', '×', '÷']):
            return original_response + "\n\nRemember that in Vedic mathematics, we focus on mental calculation techniques that reduce complexity. Try practicing this technique with similar problems to build your speed and confidence!"
        else:
            return original_response
    
    def _generate_sutras_response(self):
        """Generate a comprehensive response about the 16 Vedic sutras"""
        response = """
# The 16 Vedic Mathematics Sutras

Vedic Mathematics is based on 16 Sutras (aphorisms or formulas) and 13 sub-sutras. Here they are with their meanings and key applications:

## Main Sutras:

1. **Ekadhikena Purvena** - "By one more than the previous one"
   - Used for: Squaring numbers ending in 5, finding squares near bases, special divisions
   - Example: To find 85², take 8 × 9 = 72 and append 25. Result: 7225

2. **Nikhilam Navatashcaramam Dashatah** - "All from 9 and the last from 10"
   - Used for: Subtraction, multiplication of numbers close to bases (10, 100, etc.)
   - Example: 998 × 997 using base 1000: (-2)(-3) = 6, 998-3 = 995. Result: 995,006

3. **Urdhva Tiryagbhyam** - "Vertically and crosswise"
   - Used for: General multiplication, polynomial multiplication
   - Example: 34 × 12 computed as: (3×1)(3×2+4×1)(4×2) = 408

4. **Paravartya Yojayet** - "Transpose and adjust"
   - Used for: Division, solving linear equations
   - Example: Efficient division when divisors have complicated forms

5. **Shunyam Samyasamuccaye** - "When sum is same that sum is zero"
   - Used for: Solving equations, simplifying expressions
   - Example: Solving systems where coefficients follow specific patterns

6. **Anurupyena** - "By proportion"
   - Used for: Proportions, ratios, percentages
   - Example: Quick calculations of percentages and proportional values

7. **Sankalana Vyavakalanabhyam** - "By addition and subtraction"
   - Used for: Solving equations, simplifying complex computations
   - Example: Breaking down complex problems into simpler addition/subtraction

8. **Puranapuranabhyam** - "By completion or non-completion"
   - Used for: Completing patterns, solving equations
   - Example: Finding missing values in sequences or patterns

9. **Chalana Kalanabyham** - "Differences and similarities"
   - Used for: Calculus-like operations, finding patterns
   - Example: Finding derivatives and integrals through pattern recognition

10. **Yaavadunam** - "Whatever the extent of its deficiency"
    - Used for: Computing near bases, squaring numbers near round numbers
    - Example: 96² can be computed as 96-4=92, append 16 (4²): 9216

11. **Vyashtisamanstih** - "Part and whole"
    - Used for: Breaking problems into parts, solving by components
    - Example: Breaking complex multiplication into simpler components

12. **Shesanyankena Charamena** - "The remainders by the last digit"
    - Used for: Finding remainders, divisibility tests
    - Example: Quick divisibility rules and remainder calculations

13. **Sopaantyadvayamantyam** - "The ultimate and twice the penultimate"
    - Used for: Special series, pattern recognition
    - Example: Finding values in particular series with special relationships

14. **Ekanyunena Purvena** - "One less than the previous"
    - Used for: Finding values related to previous computations
    - Example: Finding values in descending patterns

15. **Gunitasamuchyah** - "The product of sum is equal to sum of products"
    - Used for: Factoring, algebraic identities
    - Example: Simplifying algebraic expressions

16. **Gunakasamuchyah** - "Factors of sum is equal to sum of factors"
    - Used for: Factor relationships, algebraic manipulation
    - Example: Working with factors in algebraic expressions

Each of these sutras represents a specific pattern or mathematical principle that can be applied across various calculations to achieve remarkable speed and efficiency. The beauty of Vedic mathematics lies in how these principles can be combined and applied flexibly to solve complex problems mentally.

Would you like me to explain any particular sutra in more detail with specific examples?
"""
        return response
        
    def _get_related_technique(self, technique):
        """Return a related Vedic technique for the given one"""
        relations = {
            "ekadhikena purvena": "ekanyunena purvena",
            "nikhilam": "yaavadunam",
            "urdhva tiryagbhyam": "paravartya yojayet",
            "paravartya yojayet": "nikhilam",
            "shunyam samyasamuccaye": "anurupyena",
            "anurupyena": "sankalana vyavakalanabhyam",
            "sankalana vyavakalanabhyam": "puranapuranabhyam",
            "puranapuranabhyam": "chalana kalanabyham",
            "chalana kalanabyham": "yaavadunam",
            "yaavadunam": "nikhilam",
            "vyashtisamanstih": "shesanyankena charamena",
            "shesanyankena charamena": "sopaantyadvayamantyam",
            "sopaantyadvayamantyam": "ekanyunena purvena",
            "ekanyunena purvena": "ekadhikena purvena",
            "gunitasamuchyah": "gunakasamuchyah",
            "gunakasamuchyah": "gunitasamuchyah"
        }
        
        technique_lower = technique.lower()
        for key, value in relations.items():
            if technique_lower in key:
                return value
                
        # Default to a common one if no match
        return "urdhva tiryagbhyam"
    
    def _generate_fallback_response(self, query):
        """Generate a detailed fallback response based on query keywords"""
        query_lower = query.lower()
        
        if "division" in query_lower:
            return """
# Vedic Division Techniques

Vedic mathematics offers several powerful techniques for division that make even complex calculations manageable mentally:

## 1. Paravartya Yojayet (Transpose and Adjust)
This technique is exceptionally powerful for divisions where the divisor is close to a power base (like 10, 100, etc.).

**Example: 4532 ÷ 19**
Since 19 is close to 20, we express it as (20-1) and apply the technique:
- 45÷2 = 22.5, take 22 as the first quotient digit
- Remainder: 45-(22×2) = 1, adjust: 1-(22×(-1)) = 23
- Bring down 3: 233
- 233÷2 = 116.5, take 11 as the next quotient digit
- Remainder: 233-(11×20) = 13, adjust: 13-(11×(-1)) = 24
- Bring down 2: 242
- 242÷2 = 121, take 12 as the final quotient digit
- Remainder: 242-(12×20) = 2, adjust: 2-(12×(-1)) = 14

Final answer: 238 with remainder 14 (or 238.736...)

## 2. Nikhilam Technique for Division by 9, 99, 999, etc.
When dividing by numbers like 9, 99, 999, we can use a pattern-based approach.

The beauty of Vedic division lies in transforming complex calculations into simple, step-by-step processes that can be performed mentally. Would you like me to demonstrate with specific numbers from your problem?
"""
            
        if "multiplication" in query_lower:
            return """
# Vedic Multiplication Techniques

Vedic mathematics offers several elegant approaches to multiplication that dramatically reduce calculation time:

## 1. Urdhva Tiryagbhyam (Vertically and Crosswise)
This is the most versatile technique, applicable to all numbers regardless of size.

**Example: 86 × 97**
Step 1: Multiply units digits: 6×7 = 42, write 2, carry 4
Step 2: Multiply crosswise and add: (8×7)+(9×6) = 56+54 = 110, add carry: 110+4 = 114, write 4, carry 11
Step 3: Multiply tens digits: 8×9 = 72, add carry: 72+11 = 83
Result: 8342

## 2. Nikhilam (All from 9 and last from 10)
Excellent for numbers close to bases like 10, 100, 1000.

**Example: 94 × 97** (Both close to 100)
Step 1: Find deviations: 94 = -6, 97 = -3
Step 2: Cross-add: 94-3 = 91 (or 97-6 = 91) for first part
Step 3: Multiply deviations: (-6)×(-3) = 18 for second part
Result: 9118

## 3. Special Case: Numbers Ending in 5
For multiplying numbers that end in 5, we can use a quick technique.

**Example: 35 × 45**
Step 1: Multiply first digits: 3×4 = 12
Step 2: Add one to either first digit and multiply: 3×5 = 15 (or 4×4 = 16)
Step 3: Append 25 (5×5): 1575

These techniques transform what seems like complex multiplication into a series of simpler steps that can be performed mentally. I'd be happy to demonstrate with specific examples from your problem!
"""
            
        if "square" in query_lower:
            return """
# Vedic Techniques for Squaring Numbers

Vedic mathematics provides remarkable shortcuts for squaring numbers mentally:

## 1. Numbers Ending in 5 (Ekadhikena Purvena)
This technique allows instant squaring of any number ending in 5.

**Method:**
1. Take the digit(s) before 5
2. Multiply by one more than itself
3. Append 25 to the result

**Examples:**
- 25² = 2×3 = 6, append 25: 625
- 75² = 7×8 = 56, append 25: 5625
- 135² = 13×14 = 182, append 25: 18225

## 2. Numbers Near Bases (Yaavadunam)
For numbers close to power bases (10, 100, etc.), this technique is extremely fast.

**Method:**
1. Find the deviation from the nearest base
2. Add/subtract the deviation from the original number
3. Append the square of the deviation

**Examples:**
- 98² (base 100, deviation -2):
  98-2 = 96, append (-2)² = 4: 9604
- 104² (base 100, deviation +4):
  104+4 = 108, append (+4)² = 16: 10816
- 996² (base 1000, deviation -4):
  996-4 = 992, append (-4)² = 16: 992016

## 3. General Squaring Using Duplex Method
For any two-digit number, the duplex method offers a systematic approach.

**Example: 67²**
Step 1: Square first digit: 6² = 36 (thousands place)
Step 2: Calculate duplex: 2×6×7 = 84 (tens place)
Step 3: Square second digit: 7² = 49 (units place)
Result: 4489

These techniques make squaring even large numbers feasible mentally with practice. I'd be happy to demonstrate with your specific numbers!
"""
            
        # General fallback
        return """
# Vedic Mathematics: An Ancient System for Modern Calculation

Vedic mathematics is a system of mental calculation techniques based on 16 sutras (formulas) and 13 sub-sutras derived from ancient Indian texts. These powerful methods can reduce complex calculations to simple mental processes, often cutting solution time dramatically.

## Key Advantages of Vedic Mathematics:

1. **Speed** - Most calculations can be done in a single line, much faster than conventional methods
2. **Efficiency** - Reduces the burden on memory and minimizes calculation steps
3. **Flexibility** - Multiple approaches to solve the same problem based on number patterns
4. **Creativity** - Encourages understanding number relationships rather than rote procedures
5. **Mental Calculation** - All techniques are designed to be performed mentally

## Core Principles:

The system builds on pattern recognition and number relationships. For example:
- Numbers near bases (like 10, 100, 1000) have special properties
- Numbers with specific digit patterns can be calculated using shortcuts
- Complements (like 9 from 10, 99 from 100) create powerful calculation paths

## Applications:

Vedic mathematics excels in:
- Multiplication of any size numbers
- Division with complex divisors
- Squaring and finding square roots
- Cubic calculations
- Solving algebraic equations
- Calculating trigonometric values
- Working with fractions and decimals

I'd be happy to explain any specific technique or demonstrate how to solve particular problems using these elegant Vedic approaches!
"""
    
def get_vedic_agent() -> VedicMathsAgent:
    return VedicMathsAgent(agent)