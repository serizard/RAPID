from rich.console import Console
from rich.panel import Panel
import yaml
import json
from openai import OpenAI
from model import RAPIDModel

APHASIA_TYPES = {
    0: "Control",
    1: "Fluent",
    2: "Non_Comprehensive", 
    3: "Non_Fluent"
}

def get_analysis(api_key, prediction):
    client = OpenAI(api_key=api_key)
    
    prompt = f"""As a neurologist specializing in aphasia, provide a comprehensive analysis of the following diagnosis results:

Diagnosis Values:
- Control (Normal): {1 if prediction == 0 else 0}
- Fluent Aphasia: {1 if prediction == 1 else 0}
- Non-comprehensive Aphasia: {1 if prediction == 2 else 0}
- Non-fluent aphasia: {1 if prediction == 3 else 0}

Please provide the analysis in the following JSON format:
{{
    "currentStatus": {{
        "description": "Concise summary of current condition",
        "details": "Detailed explanation of the aphasia type and its manifestations",
        "mainAffectedAreas": ["list", "of", "affected", "areas"]
    }},
    "riskLevel": {{
        "percentage": "number between 0-100",
        "description": "Detailed explanation of the risk level and its implications",
        "recommendations": ["list", "of", "recommended", "actions"],
        "immediateActions": ["list", "of", "immediate", "steps", "needed"]
    }}
}}"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert neurologist with extensive experience in aphasia diagnosis and treatment."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content)

def main():
    console = Console()
    
    with open('/workspace/RAPID/demo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model = RAPIDModel(config)
    pred, _, _ = model.predict(config['video_path'])
    
    analysis = get_analysis(config['api_key'], pred)
    
    console.print("\n[bold cyan]Aphasia Diagnosis Report[/bold cyan]\n")
    
    console.print(Panel(
        f"Diagnosis: [bold]{APHASIA_TYPES[pred]}[/bold]\n\n{analysis['currentStatus']['description']}",
        title="Primary Diagnosis",
        border_style="blue"
    ))
    
    console.print(Panel(
        f"{analysis['currentStatus']['details']}\n\nMain Affected Areas:\n" + 
        "\n".join(f"• {area}" for area in analysis['currentStatus']['mainAffectedAreas']),
        title="Detailed Analysis",
        border_style="yellow"
    ))
    
    console.print(Panel(
        f"Risk Level: {analysis['riskLevel']['percentage']}%\n\n{analysis['riskLevel']['description']}",
        title="Risk Assessment",
        border_style="red"
    ))
    
    console.print(Panel(
        "Recommendations:\n" + 
        "\n".join(f"• {rec}" for rec in analysis['riskLevel']['recommendations']) +
        "\n\nImmediate Actions:\n" +
        "\n".join(f"• {action}" for action in analysis['riskLevel']['immediateActions']),
        title="Action Plan",
        border_style="green"
    ))

if __name__ == '__main__':
    main()