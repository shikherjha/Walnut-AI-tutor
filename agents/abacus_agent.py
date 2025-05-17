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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import base64
import json
from typing import List, Dict, Tuple, Optional
load_dotenv()

WOLFRAM_API=os.environ["WOLFRAM_API_KEY"]
TAVILY_API_KEY=os.environ["TAVILY_API_KEY"]
GROQ_API_KEY=os.environ["GROQ_API_KEY"]

### Initialising the tools
tavily_search = TavilySearch(api_key=TAVILY_API_KEY)
wolfram = WolframAlphaQueryRun(
    api_wrapper = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_API)
)

class VerifyCalcTool(BaseTool):
    name="verify_calculation"
    description=("Verify the correctness of a basic arithmetic expression(provided as a string)"
               "return True if the claimed result matches, else false"  
    )             

    def _run(self, expression:str, claimed_result:str) -> float:
        computed = eval(expression)
        return computed==claimed_result

    async def _arun(self, expression:str, claimed_result:str) -> float:
        return self._run(expression, claimed_result)

verify_tool = VerifyCalcTool()

class AbacusSimulationTool(BaseTool):
    name = "generate_abacus_simulation"
    description = (
        "Generates a visualization of abacus bead movements for a calculation. "
        "Input should be a JSON string with these keys: "
        "'operation' (e.g., '123+456'), "
        "'steps' (list of abacus states as digits), "
        "'title' (optional, for the visualization). "
        "This tool returns a base64 encoded image that can be displayed in Streamlit."
    )
    
    def __init__(self):
        super().__init__()
        # Indian abacus (similar to Japanese soroban) typically has:
        # - 1 bead in upper deck (value 5)
        # - 4 beads in lower deck (each value 1)
        self.rows = 7  # Number of decimal places to show
        self.upper_beads = 1
        self.lower_beads = 4
    
    def _run(self, input_str: str) -> str:
        """
        Generate a series of abacus visualizations showing calculation steps
        
        Args:
            input_str: JSON string containing:
                - operation: The calculation being performed (e.g., "123+456")
                - steps: List of abacus states, where each state is a list of digits
                - title: Optional title for the visualization
        
        Returns:
            Base64 encoded image data that can be displayed in Streamlit with st.image
        """
        try:
            if isinstance(input_str, str):
                input_data = json.loads(input_str)
            else:
                input_data = input_str
                
            operation = input_data.get("operation", "")
            steps = input_data.get("steps", [])
            title = input_data.get("title", f"Abacus Calculation: {operation}")
            
            # Create figure with subplots for each step
            num_steps = len(steps)
            if num_steps == 0:
                return "No steps provided for visualization"
            
            fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 6))
            if num_steps == 1:
                axes = [axes]  # Make it iterable if there's only one subplot
            
            # Draw each step
            for i, (step_state, ax) in enumerate(zip(steps, axes)):
                self._draw_abacus(ax, step_state)
                step_title = f"Step {i+1}"
                if i == 0:
                    step_title = "Initial"
                elif i == num_steps - 1:
                    step_title = "Final"
                ax.set_title(step_title)
            
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            return f"Error generating abacus simulation: {str(e)}"
    
    def _draw_abacus(self, ax, state: List[int]):
        """Draw an abacus with the given state"""
        # Determine number of columns (digits) to display
        cols = len(state)
        
        # Set up the plot
        ax.set_xlim(0, cols + 1)
        ax.set_ylim(0, 7)
        ax.axis('off')
        
        # Draw the horizontal beam
        ax.add_patch(patches.Rectangle((0.5, 3), cols, 0.2, color='brown'))
        
        # Draw vertical rods
        for i in range(cols):
            rod_x = i + 1
            ax.plot([rod_x, rod_x], [0.5, 5.5], color='gray', linewidth=1.5)
            
            # Add place value labels
            place_value = cols - i - 1
            if place_value == 0:
                label = "1's"
            elif place_value == 1:
                label = "10's"
            elif place_value == 2:
                label = "100's"
            elif place_value == 3:
                label = "1000's"
            else:
                label = f"10^{place_value}"
            ax.text(rod_x, 0.2, label, ha='center', va='center', fontsize=8)
        
        # Draw beads according to state
        for i, value in enumerate(state):
            x_pos = cols - i
            
            # Convert the digit to abacus bead positions
            upper_active = 1 if value >= 5 else 0
            lower_active = value % 5
            
            # Draw upper beads
            for j in range(self.upper_beads):
                is_active = j < upper_active
                y_pos = 4.5 - j * 0.6
                color = 'darkred' if is_active else 'lightcoral'
                ax.add_patch(patches.Ellipse((x_pos, y_pos), 0.6, 0.3, color=color))
            
            # Draw lower beads
            for j in range(self.lower_beads):
                is_active = j < lower_active
                y_pos = 2.5 - j * 0.6
                color = 'darkblue' if is_active else 'lightblue'
                ax.add_patch(patches.Ellipse((x_pos, y_pos), 0.6, 0.3, color=color))
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)

class AbacusStepsGeneratorTool(BaseTool):
    name = "generate_abacus_steps"
    description = (
        "Generates step-by-step abacus states for a calculation. "
        "Input should be a calculation string like '123+456'. "
        "Returns JSON with operation and steps that can be fed to the visualization tool."
    )
    
    def _run(self, calculation: str) -> str:
        """
        Parse a calculation string and generate abacus steps.
        
        Args:
            calculation: String calculation (e.g., "123+456")
        
        Returns:
            JSON string with operation and steps for visualization
        """
        try:
            # Parse the calculation
            import re
            
            calculation = calculation.strip()
            match = re.match(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', calculation)
            if not match:
                return json.dumps({"operation": calculation, "steps": []})
            
            num1, op, num2 = match.groups()
            num1, num2 = int(num1), int(num2)
            
            # Generate abacus states
            steps = []
            
            # Get maximum number of digits needed
            result = 0
            if op == '+':
                result = num1 + num2
            elif op == '-':
                result = num1 - num2
            elif op == '*':
                result = num1 * num2
            elif op == '/':
                result = num1 // num2
            
            max_digits = max(len(str(num1)), len(str(num2)), len(str(result)))
            
            # Initial state - set up num1
            init_state = [0] * max_digits
            for i, digit in enumerate(reversed(str(num1))):
                init_state[i] = int(digit)
            steps.append(list(init_state))
            
            # Final state - after operation
            final_state = [0] * max_digits
            for i, digit in enumerate(reversed(str(result))):
                final_state[i] = int(digit)
            
            # Add intermediate steps (simplified)
            if op == '+':
                # If addition, show adding each digit of num2
                current_state = list(init_state)
                for i, digit in enumerate(reversed(str(num2))):
                    new_state = list(current_state)
                    digit_val = int(digit)
                    
                    # Add to appropriate position with carrying
                    pos = i
                    while digit_val > 0 and pos < max_digits:
                        new_val = new_state[pos] + digit_val
                        new_state[pos] = new_val % 10
                        digit_val = new_val // 10
                        pos += 1
                        
                    steps.append(new_state)
                    current_state = new_state
            elif op == '-':
                # For subtraction
                current_state = list(init_state)
                for i, digit in enumerate(reversed(str(num2))):
                    new_state = list(current_state)
                    digit_val = int(digit)
                    
                    # Subtract with borrowing
                    pos = i
                    while digit_val > 0 and pos < max_digits:
                        if new_state[pos] >= digit_val:
                            new_state[pos] -= digit_val
                            digit_val = 0
                        else:
                            # Need to borrow
                            borrow_pos = pos + 1
                            while borrow_pos < max_digits and new_state[borrow_pos] == 0:
                                borrow_pos += 1
                            
                            if borrow_pos < max_digits:
                                # Found a position to borrow from
                                new_state[borrow_pos] -= 1
                                for j in range(borrow_pos-1, pos, -1):
                                    new_state[j] += 9
                                new_state[pos] += 10 - digit_val
                                digit_val = 0
                    
                    steps.append(new_state)
                    current_state = new_state
            
            # For other operations or if no intermediate steps added, just show final state
            if steps[-1] != final_state:
                steps.append(final_state)
            
            return json.dumps({
                "operation": calculation,
                "steps": steps,
                "title": f"Abacus: {calculation} = {result}"
            })
            
        except Exception as e:
            return json.dumps({"error": str(e), "operation": calculation, "steps": []})
    
    async def _arun(self, calculation: str) -> str:
        return self._run(calculation)

# Initialize the custom abacus tools
abacus_sim_tool = AbacusSimulationTool()
abacus_steps_tool = AbacusStepsGeneratorTool()

## Initializing LLM for the agent
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7,
    api_key=GROQ_API_KEY
)   

## Chat prompt 
prompt = ChatPromptTemplate.from_template('''
You are an expert Abacus teacher, specializing in the **Indian Abacus** method. 
You teach students in an interactive and easy-to-understand way using visual simulation tools.

Your core abilities are:

1. **Teaching Topics or Techniques**:
   - When the user asks about an **abacus-related topic** (e.g., addition, subtraction, complements, place values, visualizing large numbers),
     respond with a simple, step-by-step explanation using clear language.
   - Use analogies where appropriate, and if relevant, invoke `generate_abacus_steps` followed by `generate_abacus_simulation` for visual support.

2. **Solving Doubts or Expressions**:
   - When given a math expression (like `123 + 456`), solve it using the Indian abacus technique.
   - Break it into digit-wise steps (1's, 10's, etc.) and explain **bead movement logic** clearly.
   - Then, use `generate_abacus_steps` to show the bead positions and `generate_abacus_simulation` to visually represent the solution.

3. **Verification**:
   - If needed, or when instructed, verify answers using additional tools like `verify_vedic_calc` or `wolfram` for correctness.

4. **Out-of-Domain Queries**:
   - If the question is **not related to abacus or basic arithmetic**, say:
     > "Sorry, I can’t help you with that as I specialize in Abacus learning."
   - Then follow up with:
     > "Did you know? *[Insert random Indian abacus fact or trivia]*. 
     > Want to explore such cool concepts? Join **Walnut Excellence Education** today — Enroll now!"

Use easy, friendly language for young learners, and keep responses encouraging and engaging.

---
User Query:  
{input}
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
        name="verify_abacus_calc",
        description=verify_tool.description
    ),
    Tool.from_function(
        func=abacus_steps_tool.run,
        name="generate_abacus_steps",
        description=abacus_steps_tool.description
    ),
    Tool.from_function(
        func=abacus_sim_tool.run,
        name="generate_abacus_simulation",
        description=abacus_sim_tool.description
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
class AbacusAgent:
    def __init__(self, agent_executor: Any):
        self.agent = agent_executor
    
    def handle_query(self, query: str) -> Tuple[str, Any]:
        response, trace = self.agent.run_and_trace(query)
        return response, trace
    
def get_abacus_agent() -> AbacusAgent:
    return AbacusAgent(agent)