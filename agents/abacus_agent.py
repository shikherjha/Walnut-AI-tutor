from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import BaseTool
from typing import Union, Any, Dict, Tuple
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
import json
from typing import List, Dict, Optional, Any
from pydantic import Field

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

class AbacusSimulationTool(BaseTool):
    name: str = "generate_abacus_simulation"
    description: str = (
        "Generates a visualization of abacus bead movements for a calculation. "
        "Input should be a JSON string with these keys: "
        "'operation' (e.g., '123+456'), "
        "'steps' (list of abacus states as digits), "
        "'title' (optional, for the visualization). "
        "This tool returns a base64 encoded image that can be displayed in Streamlit."
    )
    rows: int = Field(default=7, description="Number of decimal places to show")
    upper_beads: int = Field(default=1, description="Number of beads in upper deck (value 5)")
    lower_beads: int = Field(default=4, description="Number of beads in lower deck (each value 1)")
    
    def _run(self, input_str: str) -> str:
        try:
            if isinstance(input_str, str):
                input_data = json.loads(input_str)
            else:
                input_data = input_str
                
            operation = input_data.get("operation", "")
            steps = input_data.get("steps", [])
            title = input_data.get("title", f"Abacus Calculation: {operation}")
            
            num_steps = len(steps)
            if num_steps == 0:
                return "No steps provided for visualization"
            
            fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 6))
            if num_steps == 1:
                axes = [axes]
            
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
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            return f"Error generating abacus simulation: {str(e)}"
    
    def _draw_abacus(self, ax, state: List[int]):
        cols = len(state)
        
        ax.set_xlim(0, cols + 1)
        ax.set_ylim(0, 7)
        ax.axis('off')
        
        ax.add_patch(patches.Rectangle((0.5, 3), cols, 0.2, color='brown'))
        
        for i in range(cols):
            rod_x = i + 1
            ax.plot([rod_x, rod_x], [0.5, 5.5], color='gray', linewidth=1.5)
            
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
        
        for i, value in enumerate(state):
            x_pos = cols - i
            
            upper_active = 1 if value >= 5 else 0
            lower_active = value % 5
            
            for j in range(self.upper_beads):
                is_active = j < upper_active
                y_pos = 4.5 - j * 0.6
                color = 'darkred' if is_active else 'lightcoral'
                ax.add_patch(patches.Ellipse((x_pos, y_pos), 0.6, 0.3, color=color))
            
            for j in range(self.lower_beads):
                is_active = j < lower_active
                y_pos = 2.5 - j * 0.6
                color = 'darkblue' if is_active else 'lightblue'
                ax.add_patch(patches.Ellipse((x_pos, y_pos), 0.6, 0.3, color=color))
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)

class AbacusStepsGeneratorTool(BaseTool):
    name: str = "generate_abacus_steps"
    description: str = (
        "Generates step-by-step abacus states for a calculation. "
        "Input should be a calculation string like '123+456'. "
        "Returns JSON with operation and steps that can be fed to the visualization tool."
    )
    
    def _run(self, calculation: str) -> str:
        try:
            import re
            
            calculation = calculation.strip()
            match = re.match(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', calculation)
            if not match:
                return json.dumps({"operation": calculation, "steps": []})
            
            num1, op, num2 = match.groups()
            num1, num2 = int(num1), int(num2)
            
            steps = []
            
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
            
            init_state = [0] * max_digits
            for i, digit in enumerate(reversed(str(num1))):
                init_state[i] = int(digit)
            steps.append(list(init_state))
            
            final_state = [0] * max_digits
            for i, digit in enumerate(reversed(str(result))):
                final_state[i] = int(digit)
            
            if op == '+':
                current_state = list(init_state)
                for i, digit in enumerate(reversed(str(num2))):
                    new_state = list(current_state)
                    digit_val = int(digit)
                    
                    pos = i
                    while digit_val > 0 and pos < max_digits:
                        new_val = new_state[pos] + digit_val
                        new_state[pos] = new_val % 10
                        digit_val = new_val // 10
                        pos += 1
                        
                    steps.append(new_state)
                    current_state = new_state
            elif op == '-':
                current_state = list(init_state)
                for i, digit in enumerate(reversed(str(num2))):
                    new_state = list(current_state)
                    digit_val = int(digit)
                    
                    pos = i
                    while digit_val > 0 and pos < max_digits:
                        if new_state[pos] >= digit_val:
                            new_state[pos] -= digit_val
                            digit_val = 0
                        else:
                            borrow_pos = pos + 1
                            while borrow_pos < max_digits and new_state[borrow_pos] == 0:
                                borrow_pos += 1
                            
                            if borrow_pos < max_digits:
                                new_state[borrow_pos] -= 1
                                for j in range(borrow_pos-1, pos, -1):
                                    new_state[j] += 9
                                new_state[pos] += 10 - digit_val
                                digit_val = 0
                    
                    steps.append(new_state)
                    current_state = new_state
            elif op == '*':
                # Simple step generation for multiplication
                # Just show start and result for now
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

abacus_sim_tool = AbacusSimulationTool()
abacus_steps_tool = AbacusStepsGeneratorTool()

llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,  # Reduced temperature for more consistent responses
    api_key=GROQ_API_KEY,
    max_tokens=2048,  # Limited token output for faster responses
)   

# Simplified prompt with better structure and explicit directives
prompt = ChatPromptTemplate.from_template('''
You are an expert Abacus teacher, specializing in the Indian Abacus method.
Your answers should be structured, concise, and always include step-by-step explanations.

For ANY abacus question:
1. First provide a brief introduction to the concept
2. Then give step-by-step instructions (never skip steps)
3. Include precise bead movement explanations
4. ALWAYS explain visually using the abacus tools

For multiplication specifically:
- Explain the column-by-column approach
- Show how to record partial products
- Demonstrate the final carrying and consolidation steps

When teaching parts of an abacus:
- Always mention both the upper deck (heaven beads: worth 5 each) and lower deck (earth beads: worth 1 each)
- Explain how the beads represent values (moved toward the bar)
- Describe the horizontal beam and vertical rods

Use friendly, simple language with clear, numbered steps.

User Query:  
{input}
''')

chain = prompt | llm | StrOutputParser()

class AbacusAgent:
    def __init__(self, llm, chain, abacus_steps_tool, abacus_sim_tool):
        self.llm = llm
        self.chain = chain
        self.abacus_steps_tool = abacus_steps_tool
        self.abacus_sim_tool = abacus_sim_tool
    
    def handle_query(self, query: str) -> Tuple[str, Dict]:
        """Handle a query and return the response and trace information"""
        try:
            # Simplified approach: Always use the direct chain first
            response_text = self.chain.invoke({"input": query})
            
            # Check if this is a calculation that would benefit from visualization
            import re
            calc_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', query)
            
            if calc_match:
                try:
                    # Get the matched calculation
                    calculation = calc_match.group(0)
                    
                    # Generate abacus steps and visualization
                    steps_json = self.abacus_steps_tool.run(calculation)
                    img_base64 = self.abacus_sim_tool.run(steps_json)
                    
                    # Add visualization to response
                    response_text += f"\n\n![Abacus Visualization](data:image/png;base64,{img_base64})"
                except Exception as viz_error:
                    # Silently continue without visualization if there's an error
                    pass
            
            # Adding a standard footer for consistent branding
            response_text += "\n\n---\n*Visualized with the Indian Abacus technique.*"
            
            return response_text, {"direct_chain": True}
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            return error_msg, {"error": str(e)}
    
def get_abacus_agent() -> AbacusAgent:
    return AbacusAgent(llm, chain, abacus_steps_tool, abacus_sim_tool)