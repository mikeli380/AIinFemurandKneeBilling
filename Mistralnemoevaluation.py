import json
import ollama
import yaml
import random
import pandas as pd
import os
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.program import LLMCompletionProgram
from enum import Enum
#Set up prompt CPT Description: "{cpt_description}" 

#Binary Response Trial Template
prompt_template_str = """
Given the following orthopedic operative note, the associated CPT code, and its description, please evaluate whether the CPT code accurately reflects the operativeedures described in the note. If the code is suitable, confirm that the code aligns with the documented operativeedure.
Type of orthopedic surgery: "{type_of_orthopedic_surgery}"
Operational note: "{operational_note}"
CPT Code: "{cpt_code}"
CPT Description: "{cpt_description}"
Please provide the output in a one word "Yes" or "No" format with a Yes for a correct CPT code, and a No for an incorrect CPT code. Do not include anything else.
In your evaluation, consider factors such as the specific operativeedures performed, any anatomical details, the surgical approach, and any modifiers that may apply based on the documentation. DO NOT MAKE UP ANY INFORMATION.
"""

# Confidece-Response Trials
# prompt_template_str = """
# Given the following orthopedic operative note, the associated CPT code, and its description, please evaluate whether the CPT code accurately reflects the operativeedures described in the note. If the code is suitable, confirm that the code aligns with the documented operativeedure.
# Type of orthopedic surgery: "{type_of_orthopedic_surgery}"
# Operational note: "{operational_note}"
# CPT Code: "{cpt_code}"
# CPT Description: "{cpt_description}"
# Please provide the output in a numeric format from a 0 to a 100 scale with a 100 representing complete confidence for a correct CPT code, and a 0 representing complete confidence if it is an incorrect CPT code. Do not include anything else.
# In your evaluation, consider factors such as the specific operativeedures performed, any anatomical details, the surgical approach, and any modifiers that may apply based on the documentation. DO NOT MAKE UP ANY INFORMATION.
# """
# Load the config file
with open('setup.yaml', 'r') as f:
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

type_of_orthopedic_surgery = variables.get('type_of_orthopedic_surgery', 'orthopedic Surgery')

# Load the .pkl file
df = pd.read_pickle(pkl_file_path)

# Ensure output directory exists
if not os.path.exists(output_directory):
   os.makedirs(output_directory)

#Produce Random CPT Code
def get_random_cpt_description(file_path):
   with open(file_path, 'r') as file:
       data = json.load(file)
   random_code = str(random.choice(list(data.keys())))
   cpt_description = str(data[random_code].get("CPT Code Description", "No description available"))
   return random_code, cpt_description

# Prepare each note for trial
for index, row in df.head(sample_number).iterrows():
    operative_note_ = row['operative_note_']
    ortho_operative_CPT = row['ortho_operative_CPT']  # Should be a list of CPT codes
    CPT_GUIDELINE = row['CPT_GUIDELINE']  # List of dictionaries
    for idx, cpt_code in enumerate(ortho_operative_CPT):
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
        'operational_note': operative_note_,
        'cpt_code': cpt_code,
        'cpt_description': cpt_description,
        'type_of_orthopedic_surgery': type_of_orthopedic_surgery,
        'length_of_overall_explanation': length_of_overall_explanation
         }
    else:
        sampled_cpt, sampled_cpt_d = get_random_cpt_description(cpt_database)
        ortho_operative_CPT = sampled_cpt  # Sampled CPT
        CPT_GUIDELINE = sampled_cpt_d  # Sampled description
         # Prepare input variables for the prompt
        input_variables = {
            'operational_note': operative_note_,
            'cpt_code': sampled_cpt,
            'cpt_description': sampled_cpt_d,
            'type_of_orthopedic_surgery': type_of_orthopedic_surgery,
            'length_of_overall_explanation': length_of_overall_explanation
        }
      
    formatted_prompt = prompt_template_str.format(**input_variables)

    # Prompt the language model
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
       #Format the output
    response = ChatResponse['message']['content']

    # Prepare output data
    if sample_type.lower() == 'positive':
        output_data = {
            'sample_type': 'Positive',
            'cpt_code': cpt_code,
            'cpt_description': cpt_description,
            'response': response
        }
    else:
        output_data = {
            'sample_type': 'Negative',
            'cpt_code': cpt_code,
            'cpt_description': cpt_description,
            'response': response,
            'false_code': sampled_cpt,
            'false_code_description': sampled_cpt_d
        }

# Save output_data dictionary to CSV
csv_filepath = os.path.join(output_directory, csv_filename)

with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=output_data.keys())
    writer.writeheader()
    writer.writerow(output_data)

# Append to master CSV
master_csv = os.path.join(output_directory, "output.csv")
file_exists = os.path.isfile(master_csv)

with open(master_csv, 'a', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(f, fieldnames=output_data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(output_data)

