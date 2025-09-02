from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fixed_description_embed(names, embed_dict):
    description_embed_list = []
    for input_name in names:
        # print(input_name)
        type_mapping = {'D': 'Dress', 'T': 'Tops', 'P': 'Pants', 'S': 'Skirt'}
        property_mapping = {'S': 'Short', 'L': 'Long', 'H': 'Hooded', 'C': 'Collar', 'N': 'No-Collar'}
        sleeve_mapping = {'G': 'Gallus', 'T': 'Tube', 'L': 'Long-Sleeve', 'S': 'Short-Sleeve', 'N':'No-Sleeve'}
        extra_mapping = {'S': ' ', 'C': 'FrontClose', 'O': 'FrontOpen'}

        # Define the folding methods based on clothing types
        sleeveless_folding = ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']
        short_sleeve_folding = ['DLSS', 'DSSS', 'TCSC', 'TNSC']
        long_sleeve_folding = ['DLLS', 'DSLS', 'TCLC', 'TCLO', 'THLC', 'THLO', 'TNLC']
        pants_folding = ['PL', 'PS']

        # Extract information from the input string
        parts = input_name.split('_')
        cloth_code = parts[0]
        # cloth_type_name = parts[1]
        action = int(parts[-1].replace('action', ''))

        # Determine the type of clothing
        cloth_type = type_mapping.get(cloth_code[0], 'Unknown')
        description_parts = [cloth_type]

        is_mirror = False
        if parts[1] == 'L':
            is_mirror = False
        elif parts[1] == 'R':
            is_mirror = True
        else:
            assert 0, "assumpt the second letter indicate L/R"

        # Determine the properties of the clothing
        if len(cloth_code) > 1:
            description_parts.append(property_mapping.get(cloth_code[1], ''))

        if len(cloth_code) > 2:
            description_parts.append(sleeve_mapping.get(cloth_code[2], ''))

        if len(cloth_code) > 3:
            description_parts.append(extra_mapping.get(cloth_code[3], ''))

        if cloth_code in sleeveless_folding:
            description = "Fold the no-sleeve cloth bottom-up."
        elif cloth_code in short_sleeve_folding:
            if cloth_type == 'Dress':
                if action == 0:
                    description = "Fold the short-sleeve cloth from the left."
                elif action == 1:
                    description = "Fold the short-sleeve cloth from the right."
                elif action == 2:
                    description = "Fold the short-sleeve cloth bottom-up."
            elif cloth_type == 'Tops':
                if action == 0:
                    description = "Fold the short-sleeve cloth from the left."
                elif action == 1:
                    description = "Fold the short-sleeve cloth from the right."
                elif action == 2:
                    description = "Fold the short-sleeve cloth bottom-up."
        elif cloth_code in long_sleeve_folding:
            if action == 0:
                description = "Fold the long-sleeve cloth from the left."
            elif action == 1:
                description = "Fold the long-sleeve cloth from the right."
            elif action == 2:
                description = "Fold the long-sleeve cloth bottom-up."
        elif cloth_code in pants_folding:
            if action == 0:
                description = "Fold the pants from the left."
            elif action == 1:
                description = "Fold the pants bottom-up."
        else:
            description = "unknown folding method"


        select_description = description

        if is_mirror:
            select_description = select_description.replace("left", "temp").replace("right", "left").replace("temp", "right")

        if 'left' in select_description:
            select_description = select_description.replace("left", "right")
        elif 'right' in select_description:
            select_description = select_description.replace("right", "left")
        description_embed = embed_dict[select_description]
        description_embed_list.append(description_embed)
    description_embed_batch = torch.stack(description_embed_list, dim=0)
    # print(description_embed_batch)
    # print(description_embed_batch.shape)
    # assert 0
    return torch.tensor(description_embed_batch)



def fixed_description_list(names, embed_dict, mode=None):
    description_list = []
    for input_name in names:
        # print(input_name)
        type_mapping = {'D': 'Dress', 'T': 'Tops', 'P': 'Pants', 'S': 'Skirt'}
        property_mapping = {'S': 'Short', 'L': 'Long', 'H': 'Hooded', 'C': 'Collar', 'N': 'No-Collar'}
        sleeve_mapping = {'G': 'Gallus', 'T': 'Tube', 'L': 'Long-Sleeve', 'S': 'Short-Sleeve', 'N':'No-Sleeve'}
        extra_mapping = {'S': ' ', 'C': 'FrontClose', 'O': 'FrontOpen'}

        # Define the folding methods based on clothing types
        sleeveless_folding = ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']
        short_sleeve_folding = ['DLSS', 'DSSS', 'TCSC', 'TNSC']
        long_sleeve_folding = ['DLLS', 'DSLS', 'TCLC', 'TCLO', 'THLC', 'THLO', 'TNLC']
        pants_folding = ['PL', 'PS']

        # Extract information from the input string
        parts = input_name.split('_')
        cloth_code = parts[0]
        # cloth_type_name = parts[1]
        action = int(parts[-1].replace('action', ''))

        # Determine the type of clothing
        cloth_type = type_mapping.get(cloth_code[0], 'Unknown')
        description_parts = [cloth_type]

        is_mirror = False
        if parts[1] == 'L':
            is_mirror = False
        elif parts[1] == 'R':
            is_mirror = True
        else:
            assert 0, "assumpt the second letter indicate L/R"

        # Determine the properties of the clothing
        if len(cloth_code) > 1:
            description_parts.append(property_mapping.get(cloth_code[1], ''))

        if len(cloth_code) > 2:
            description_parts.append(sleeve_mapping.get(cloth_code[2], ''))

        if len(cloth_code) > 3:
            description_parts.append(extra_mapping.get(cloth_code[3], ''))

        if cloth_code in sleeveless_folding:
            description = "Fold the no-sleeve cloth bottom-up."
        elif cloth_code in short_sleeve_folding:
            if cloth_type == 'Dress':
                if action == 0:
                    description = "First, fold the short-sleeve cloth from the left."
                elif action == 1:
                    description = "Second, fold the short-sleeve cloth from the right."
                elif action == 2:
                    description = "Fold the short-sleeve cloth bottom-up."
            elif cloth_type == 'Tops':
                if action == 0:
                    description = "First, fold the short-sleeve cloth from the left."
                elif action == 1:
                    description = "Second, fold the short-sleeve cloth from the right."
                elif action == 2:
                    description = "Fold the short-sleeve cloth bottom-up."
        elif cloth_code in long_sleeve_folding:
            if action == 0:
                description = "First, fold the long-sleeve cloth from the left."
            elif action == 1:
                description = "Second, fold the long-sleeve cloth from the right."
            elif action == 2:
                description = "Fold the long-sleeve cloth bottom-up."
        elif cloth_code in pants_folding:
            if action == 0:
                description = "Fold the pants from the left."
            elif action == 1:
                description = "Fold the pants bottom-up."
        else:
            description = "unknown folding method"

        select_description = description

        if is_mirror:
            select_description = select_description.replace("left", "temp").replace("right", "left").replace("temp", "right")

        # if 'left' in select_description:
        #     select_description = select_description.replace("left", "right")
        # elif 'right' in select_description:
        #     select_description = select_description.replace("right", "left")

        if mode=='releft':
            select_description = select_description.replace("right", "left")
            select_description = select_description.replace("bottom-up", "from the left")
        elif mode =='reright':
            select_description = select_description.replace("left", "right")
            select_description = select_description.replace("bottom-up", "from the right")
        elif mode == 'rebottom-up':
            select_description = select_description.replace("from the right", "bottom-up")
            select_description = select_description.replace("from the left", "bottom-up")

        description_list.append(select_description)

    return description_list


def gen_descriptions(prefixes, clothes, methods, suffixes):
    """
    prefixes = ['Please fold the ', 'Fold the ']
    clothes = ['dress', 'long dress']
    method = ['bottom-up', 'from the bottom', 'from bottom to top']
    suffixes = ['', '.', ]
    """
    descriptions = []
    for prefix in prefixes:
        for cloth in clothes:
            for method in methods:
                for suffix in suffixes:
                    if cloth == '':
                        descriptions.append(f"{prefix}{cloth}{method}{suffix}")
                    else:
                        descriptions.append(f"{prefix}{cloth} {method}{suffix}")
    return descriptions



def randomize_description_embed(names, embed_dict):
    description_embed_list = []
    for input_name in names:
        # print(input_name)
        type_mapping = {'D': 'Dress', 'T': 'Tops', 'P': 'Pants', 'S': 'Skirt'}
        property_mapping = {'S': 'Short', 'L': 'Long', 'H': 'Hooded', 'C': 'Collar', 'N': 'No-Collar'}
        sleeve_mapping = {'G': 'Gallus', 'T': 'Tube', 'L': 'Long-Sleeve', 'S': 'Short-Sleeve', 'N':'No-Sleeve'}
        extra_mapping = {'S': ' ', 'C': 'FrontClose', 'O': 'FrontOpen'}

        # Define the folding methods based on clothing types
        sleeveless_folding = ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']
        short_sleeve_folding = ['DLSS', 'DSSS', 'TCSC', 'TNSC']
        long_sleeve_folding = ['DLLS', 'DSLS', 'TCLC', 'TCLO', 'THLC', 'THLO', 'TNLC']
        pants_folding = ['PL', 'PS']

        # Extract information from the input string
        parts = input_name.split('_')
        cloth_code = parts[0]
        # cloth_type_name = parts[1]
        action = int(parts[-1].replace('action', ''))

        # Determine the type of clothing
        cloth_type = type_mapping.get(cloth_code[0], 'Unknown')
        description_parts = [cloth_type]

        is_mirror = False
        if parts[1] == 'L':
            is_mirror = False
        elif parts[1] == 'R':
            is_mirror = True
        else:
            assert 0, "assumpt the second letter indicate L/R"

        # Determine the properties of the clothing
        if len(cloth_code) > 1:
            description_parts.append(property_mapping.get(cloth_code[1], ''))

        if len(cloth_code) > 2:
            description_parts.append(sleeve_mapping.get(cloth_code[2], ''))

        if len(cloth_code) > 3:
            description_parts.append(extra_mapping.get(cloth_code[3], ''))

        description = ', '.join(filter(None, description_parts))

        # print(cloth_type)

        # Determine folding method based on the code
        if cloth_code in sleeveless_folding:

            possible_description = gen_descriptions(
                prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                clothes=['the cloth', 'the garment', 'the no-sleeve cloth', 'the sleeveless cloth', ''],
                methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                suffixes=['', '.', ]
            )
            if cloth_type == "Tops":
                tops_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the top', 'the shirt', 'the no-sleeve top', 'the no-sleeve shirt', 'the sleeveless top', 'the sleeveless shirt'],
                    methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                    suffixes=['', '.', ]
                )
                possible_description.extend(tops_description)
            elif cloth_type == "Dress":
                dress_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the dress', 'the no-sleeve dress', 'the sleeveless dress'],
                    methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                    suffixes=['', '.', ]
                )
                possible_description.extend(dress_description)
            elif cloth_type == "Skirt":
                skirt_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the skirt'],
                    methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                    suffixes=['', '.', ]
                )
                possible_description.extend(skirt_description)
        elif cloth_code in short_sleeve_folding:
            if action == 0:
                possible_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the cloth', 'the garment', 'the short-sleeve cloth', 'the short-sleeve garment', ''],
                    methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
                    suffixes=['', '.', ]
                )
                if cloth_type == "Tops":
                    tops_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the top', 'the shirt', 'the short-sleeve top', 'the short-sleeve shirt', 'the T-shirt'],
                        methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(tops_description)
                elif cloth_type == "Dress":
                    dress_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the dress', 'the short-sleeve dress'],
                        methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(dress_description)
                
            elif action == 1:
                possible_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the cloth', 'the garment', 'the short-sleeve cloth', 'the short-sleeve garment', ''],
                    methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
                    suffixes=['', '.', ]
                )
                if cloth_type == "Tops":
                    tops_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the top', 'the shirt', 'the short-sleeve top', 'the short-sleeve shirt', 'the T-shirt'],
                        methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(tops_description)
                elif cloth_type == "Dress":
                    dress_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the dress', 'the short-sleeve dress'],
                        methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(dress_description)
            
            elif action == 2:
                possible_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the cloth', 'the garment', 'the short-sleeve cloth', 'the short-sleeve garment', ''],
                    methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                    suffixes=['', '.', ]
                )
                if cloth_type == "Tops":
                    tops_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the top', 'the shirt', 'the short-sleeve top', 'the short-sleeve shirt', 'the T-shirt'],
                        methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(tops_description)
                elif cloth_type == "Dress":
                    dress_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the dress', 'the short-sleeve dress'],
                        methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(dress_description)
        elif cloth_code in long_sleeve_folding:
            if action == 0:
                possible_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the cloth', 'the garment', 'the long-sleeve cloth', 'the long-sleeve garment', ''],
                    methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
                    suffixes=['', '.', ]
                )
                if cloth_type == "Tops":
                    tops_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the top', 'the shirt', 'the long-sleeve top', 'the long-sleeve shirt', 'the T-shirt'],
                        methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(tops_description)
                elif cloth_type == "Dress":
                    dress_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the dress', 'the long-sleeve dress'],
                        methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(dress_description)
                
            elif action == 1:
                possible_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the cloth', 'the garment', 'the long-sleeve cloth', 'the long-sleeve garment', ''],
                    methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
                    suffixes=['', '.', ]
                )
                if cloth_type == "Tops":
                    tops_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the top', 'the shirt', 'the long-sleeve top', 'the long-sleeve shirt', 'the T-shirt'],
                        methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(tops_description)
                elif cloth_type == "Dress":
                    dress_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the dress', 'the long-sleeve dress'],
                        methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(dress_description)
            
            elif action == 2:
                possible_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the cloth', 'the garment', 'the long-sleeve cloth', 'the long-sleeve garment', ''],
                    methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                    suffixes=['', '.', ]
                )
                if cloth_type == "Tops":
                    tops_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the top', 'the shirt', 'the long-sleeve top', 'the long-sleeve shirt', 'the T-shirt'],
                        methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(tops_description)
                elif cloth_type == "Dress":
                    dress_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the dress', 'the long-sleeve dress'],
                        methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                        suffixes=['', '.', ]
                    )
                    possible_description.extend(dress_description)
        elif cloth_code in pants_folding:
            if action == 0:
                possible_description = gen_descriptions(
                    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                    clothes=['the pants', 'the trousers', ''],
                    methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the right side', 'from the left leg'],
                    suffixes=['', '.', ]
                )
            elif action == 1:
                possible_description = gen_descriptions(
                        prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
                        clothes=['the pants', 'the trousers', ''],
                        methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
                        suffixes=['', '.', ]
                    )
        else:
            possible_description = "unknown folding method"

        # Combine description with folding method
        # final_description = f"{description}, {fold_description}"
        select_description = random.choice(possible_description)
        # print(select_description, ': ', embed_dict[select_description])
        
        if is_mirror:
            select_description = select_description.replace("left", "temp").replace("right", "left").replace("temp", "right")

        # select_description = select_description.replace("left", "temp").replace("right", "left").replace("temp", "right")
        print(select_description, ': ', embed_dict[select_description])
        print('matching: ', description_matching(select_description))
        description_embed = embed_dict[select_description]
        description_embed_list.append(description_embed)
        
    description_embed_batch = torch.stack(description_embed_list, dim=0)
    # print(description_embed_batch)
    # print(description_embed_batch.shape)
    return torch.tensor(description_embed_batch)


def description_matching(description, embed_dict):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    texts = [
"You are a language classifier. I will provide you with some classification targets, \
and you need to help me read the user's description and match it to the classification target that has the closest meaning. \
Note: You should focus more on cloth type and folding directions.\
The '\n' and '\"' are not included in the classification target. You need to output every character of the classification target precisely.\
Output: You can only output the closest classification target and no other text.\
The classification targets are as follows (separated by commas): \
\"Fold the no-sleeve cloth bottom-up.\",\
\"Fold the short-sleeve dress from the left.\",\
\"Fold the short-sleeve dress from the right.\",\
\"Fold the short-sleeve dress bottom-up.\",\
\"Fold the short-sleeve top from the left.\",\
\"Fold the short-sleeve top from the right.\",\
\"Fold the short-sleeve top bottom-up.\",\
\"Fold the long-sleeve cloth from the left.\",\
\"Fold the long-sleeve cloth from the right.\",\
\"Fold the long-sleeve cloth bottom-up.\",\
\"Fold the pants from the left.\",\
\"Fold the pants bottom-up.\"\
",
    # "Please fold the trousers from the bottom to the top."     
    # "Tops, Long-Sleeves, fold the right sleeve."  
    # "Please fold the cloth from the bottom to the top."
    # "Please fold the cloth from the right side."
    # "Please fold the short-sleeve cloth from the right side."
    # "Please fold the T-shirt from the left."
    description
    ]
    # for input_text in texts:
    messages = [
        {"role": "system", "content": texts[0]},
        {"role": "system", "content": texts[1]},
        # {"role": "user", "content": "Please explain llama3 for me."},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )

    matching_description = outputs[0]["generated_text"][-1]['content']
    print(f'Origin: \"{description}\",  Matching: \"{matching_description}\"')

    embedding = embed_dict[matching_description].unsqueeze(0)

    return matching_description, embedding


def fixed_description_embed_new(names, embed_dict, mode = None):
    description_embed_list = []
    for input_name in names:
        # print(input_name)
        type_mapping = {'D': 'Dress', 'T': 'Tops', 'P': 'Pants', 'S': 'Skirt'}
        property_mapping = {'S': 'Short', 'L': 'Long', 'H': 'Hooded', 'C': 'Collar', 'N': 'No-Collar'}
        sleeve_mapping = {'G': 'Gallus', 'T': 'Tube', 'L': 'Long-Sleeve', 'S': 'Short-Sleeve', 'N':'No-Sleeve'}
        extra_mapping = {'S': ' ', 'C': 'FrontClose', 'O': 'FrontOpen'}

        # Define the folding methods based on clothing types
        sleeveless_folding = ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']
        short_sleeve_folding = ['DLSS', 'DSSS', 'TCSC', 'TNSC']
        long_sleeve_folding = ['DLLS', 'DSLS', 'TCLC', 'TCLO', 'THLC', 'THLO', 'TNLC']
        pants_folding = ['PL', 'PS']

        # Extract information from the input string
        parts = input_name.split('_')
        cloth_code = parts[0]
        # cloth_type_name = parts[1]
        action = int(parts[-1].replace('action', ''))

        # Determine the type of clothing
        cloth_type = type_mapping.get(cloth_code[0], 'Unknown')
        description_parts = [cloth_type]

        is_mirror = False
        if parts[1] == 'L':
            is_mirror = False
        elif parts[1] == 'R':
            is_mirror = True
        else:
            assert 0, "assumpt the second letter indicate L/R"

        # Determine the properties of the clothing
        if len(cloth_code) > 1:
            description_parts.append(property_mapping.get(cloth_code[1], ''))

        if len(cloth_code) > 2:
            description_parts.append(sleeve_mapping.get(cloth_code[2], ''))

        if len(cloth_code) > 3:
            description_parts.append(extra_mapping.get(cloth_code[3], ''))

        if cloth_code in sleeveless_folding:
            description = "First,no-sleeve,bottom-up"
        elif cloth_code in short_sleeve_folding:
            if cloth_type == 'Dress':
                if action == 0:
                    description = "First,short-sleeve,left"
                elif action == 1:
                    description = "Second,short-sleeve,right"
                elif action == 2:
                    description = "Third,short-sleeve,bottom-up"
            elif cloth_type == 'Tops':
                if action == 0:
                    description = "First,short-sleeve,left"
                elif action == 1:
                    description = "Second,short-sleeve,right"
                elif action == 2:
                    description = "Third,short-sleeve,bottom-up"
        elif cloth_code in long_sleeve_folding:
            if action == 0:
                description = "First,long-sleeve,left"
            elif action == 1:
                description = "Second,long-sleeve,right"
            elif action == 2:
                description = "Third,long-sleeve,bottom-up"
        elif cloth_code in pants_folding:
            if action == 0:
                description = "First,pants,left"
            elif action == 1:
                description = "Second,pants,bottom-up"
        else:
            description = "unknown folding method"


        select_description = description

        if is_mirror:
            select_description = select_description.replace("left", "temp").replace("right", "left").replace("temp", "right")

        if mode=='releft':
            select_description = select_description.replace("right", "left")
            select_description = select_description.replace("bottom-up", "left")
        elif mode =='reright':
            select_description = select_description.replace("left", "right")
            select_description = select_description.replace("bottom-up", "right")
        elif mode == 'rebottom-up':
            select_description = select_description.replace("right", "bottom-up")
            select_description = select_description.replace("left", "bottom-up")
                
        description_embed = embed_dict[select_description]
        description_embed_list.append(description_embed)
    description_embed_batch = torch.stack(description_embed_list, dim=0)
    # print(description_embed_batch)
    # print(description_embed_batch.shape)
    # assert 0
    return torch.tensor(description_embed_batch)