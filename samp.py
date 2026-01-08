

import typer
from rich.console import Console
from rich.text import Text
from rich_gradient import Gradient
from rich_gradient import Panel 
import pyfiglet
from main import inference

chat = typer.Typer()
console = Console()
text = pyfiglet.figlet_format("Onyx",font='larry3d')
model_info = """
Model:          Onyx-70M (MoE)
Architecture:   Mixture-of-Experts (MoE) Transformer
Size:           70 Million Parameters (Active params < 20M/token)
Training:       
    - Pretraining: FineWebEdu (General English understanding)
    - Fine-tuning: Manim Dataset (Python math animation syntax)
Capabilities:   Translates natural language prompts into executable Manim scripts.
Limitation:     Optimized for short, single-file scripts. May require minor syntax corrections.
"""

@chat.command()
def chat():
    console.print(Gradient(text,colors=['#00ffff','#ccffff','#0099ff','#33cc33','#ff0000'],justify='center'),style='magenta')
    console.print(
        Panel(
        f"[reset][white]{model_info}[/white]",
        title="[#ff0000]model_info",
        colors=['#00ffff','#ccffff','#0099ff',"#171a17"],
        subtitle="[#33cc33]ONyx",
        style='magenta',
        border_style="bold #ccffff",
        justify= 'center'
    ))
    while True:
        prompt = console.input("[#b3b3b3]You:").strip()
        
        full_response = ""
        for resp in inference.gen(prompt):
            console.print(resp, style='dim italic #262626',end="")
            
    


        if prompt.lower() in ("bye", "quit", "exit"):
            break
        

if __name__ == "__main__":
    chat() 







































