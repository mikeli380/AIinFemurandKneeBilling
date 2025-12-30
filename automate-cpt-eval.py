import yaml
import random
import pandas as pd
import os
import json
import ollama
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.program import LLMTextCompletionProgram
from enum import Enum
#Set up prompt CPT Description: "{cpt_description}"
prompt_template_str = """
Given the following orthopedic operational note, the associated CPT code, please evaluate whether the CPT code accurately reflects the procedures described in the note. If the code is suitable, confirm that the code aligns with the documented procedure. If it is not suitable, suggest an alternative CPT code that better matches the procedural details provided in the operational note.
Type of orthopedic surgery: "{type_of_orthopedic_surgery}"
Operational Note: "{operational_note}"
CPT Code: "{cpt_code}"
CPT Description: "{cpt_description}"
Please provide the output in a one word "Yes" or "No" format with a Yes for a correct CPT code, and a No for an incorrect CPT code.  
In your evaluation, consider factors such as the specific procedures performed, any anatomical details, the surgical approach, and any modifiers that may apply based on the documentation. 
DO NOT MAKE UP ANY INFORMATION.
"""
# Load the config file
with open('Pipeline\config.yaml', 'r') as f:
   config = yaml.safe_load(f)

#Load config variables
pkl_file_path = config['pkl_file_path']
ollama_model = config['ollama_model']
output_directory = config['output_directory']
variables = config['variables']
cpt_database = config['sample_cpt_database']
sample_type = config['sample_type']
sample_number = config['sample_number']
temperature = config['temperature']

type_of_orthopedic_surgery = variables.get('type_of_orthopedic_surgery', 'Orthopedic Surgery')
length_of_overall_explanation = variables.get('length_of_overall_explanation', 3)

# Load the .pkl file
df = pd.read_pickle(pkl_file_path)

# Ensure output directory exists
if not os.path.exists(output_directory):
   os.makedirs(output_directory)

# # Define the data model
# class CPTCodeSuitability(BaseModel):
#    b_documentation_validation: str
# # d_documentation_validation: str
# # Set up the LLM
# llm = Ollama(model=ollama_model, request_timeout=9000.0)
# # Set up the program
# program = LLMTextCompletionProgram.from_defaults(
#    llm=llm,
#    output_cls=CPTCodeSuitability,
#    prompt_template_str=prompt_template_str,
#    verbose=True,
# )

#Produce Random Code
def get_random_cpt_description(file_path):
   with open(file_path, 'r') as file:
       data = json.load(file)
   random_code = str(random.choice(list(data.keys())))
   cpt_description = str(data[random_code].get("CPT Code Description", "No description available"))
   return random_code, cpt_description

# Process each row
for index, row in df.head(sample_number).iterrows():
    ENCOUNTER_ID = row['ENCOUNTER_ID']
    PROC_NOTE_TEXT = row['PROC_NOTE_TEXT']
    ORTHO_PROC_CPT = row['ORTHO_PROC_CPT']  # Should be a list of CPT codes
    CPT_GUIDELINE = row['CPT_GUIDELINE']  # List of dictionaries
    for idx, cpt_code in enumerate(ORTHO_PROC_CPT):
        # Match CPT code and guideline by index
        if idx < len(CPT_GUIDELINE):
            cpt_guideline = CPT_GUIDELINE[idx]
            #cpt_description = cpt_guideline.get('Description', '')
            cpt_description = str(cpt_guideline)
        else:
            cpt_description = 'MISSING DESCRIPTION'
    if sample_type.lower() == 'positive':
        # Prepare input variables for the prompt
        input_variables = {
        'operational_note': PROC_NOTE_TEXT,
        'cpt_code': cpt_code,
        'cpt_description': cpt_description,
        'type_of_orthopedic_surgery': type_of_orthopedic_surgery,
        'length_of_overall_explanation': length_of_overall_explanation
         }
    else:
        sampled_cpt, sampled_cpt_d = get_random_cpt_description(cpt_database)
        ORTHO_PROC_CPT = sampled_cpt  # Sampled CPT
        CPT_GUIDELINE = sampled_cpt_d  # Sampled description
         # Prepare input variables for the prompt
        input_variables = {
            'operational_note': PROC_NOTE_TEXT,
            'cpt_code': sampled_cpt,
            'cpt_description': sampled_cpt_d,
            'type_of_orthopedic_surgery': type_of_orthopedic_surgery,
            'length_of_overall_explanation': length_of_overall_explanation
        }
      
    formatted_prompt = prompt_template_str.format(**input_variables)

    # Run the LLM
    ChatResponse = ollama.chat(
        model= ollama_model,
        messages=[
            {
                'role': 'user',
                'content': formatted_prompt,           
            },
            ],
        options={"temperature":temperature}
    )
       # Convert the output to dict
    response = ChatResponse['message']['content']

    # Prepare output data
    if sample_type.lower() == 'positive':
        output_data = {
            'ENCOUNTER_ID': ENCOUNTER_ID,
            'PROC_NOTE_TEXT': PROC_NOTE_TEXT,
            'sample_type': 'Positive',
            'cpt_code': cpt_code,
            'cpt_description': cpt_description,
            'response': response
        }
    else:
        output_data = {
            'ENCOUNTER_ID': ENCOUNTER_ID,
            'PROC_NOTE_TEXT': PROC_NOTE_TEXT,
            'sample_type': 'Negative',
            'cpt_code': cpt_code,
            'cpt_description': cpt_description,
            'response': response,
            'false_code': sampled_cpt,
            'false_code_description': sampled_cpt_d
        }

    # Save to JSON file
    filename = f"{ENCOUNTER_ID}_{cpt_code}.json"
    filepath = os.path.join(output_directory, filename)
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    with open(output_directory + "\output.txt", "a", encoding='utf-8') as file:
        file.write(f"{ENCOUNTER_ID}, {cpt_code}, "+ response + "\n")

