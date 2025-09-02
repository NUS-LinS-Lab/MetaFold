from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import description_encoding, llm_embedding

llm_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(llm_model_id, use_fast=True) 
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


llm_model = LlamaForCausalLM.from_pretrained(llm_model_id, output_hidden_states=True)
llm_model.resize_token_embeddings(len(tokenizer))

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

possible_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the cloth', 'the garment', 'the no-sleeve cloth', 'the sleeveless cloth', ''],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
)

tops_description = gen_descriptions(
    prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
    clothes=['the top', 'the shirt', 'the no-sleeve top', 'the no-sleeve shirt', 'the sleeveless top', 'the sleeveless shirt'],
    methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
    suffixes=['', '.', ]
)
possible_description.extend(tops_description)

dress_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the dress', 'the no-sleeve dress', 'the sleeveless dress'],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
)
possible_description.extend(dress_description)

skirt_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the skirt'],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
)
possible_description.extend(skirt_description)

possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the cloth', 'the garment', 'the short-sleeve cloth', 'the short-sleeve garment', ''],
methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
suffixes=['', '.', ]
))

tops_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the top', 'the shirt', 'the short-sleeve top', 'the short-sleeve shirt', 'the T-shirt'],
methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(tops_description)

dress_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the dress', 'the short-sleeve dress'],
methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(dress_description)


possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the cloth', 'the garment', 'the short-sleeve cloth', 'the short-sleeve garment', ''],
methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
suffixes=['', '.', ]
))

tops_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the top', 'the shirt', 'the short-sleeve top', 'the short-sleeve shirt', 'the T-shirt'],
methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(tops_description)

dress_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the dress', 'the short-sleeve dress'],
methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(dress_description)


possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the cloth', 'the garment', 'the short-sleeve cloth', 'the short-sleeve garment', ''],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
))

tops_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the top', 'the shirt', 'the short-sleeve top', 'the short-sleeve shirt', 'the T-shirt'],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
)
possible_description.extend(tops_description)

dress_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the dress', 'the short-sleeve dress'],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
)
possible_description.extend(dress_description)


possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the cloth', 'the garment', 'the long-sleeve cloth', 'the long-sleeve garment', ''],
methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
suffixes=['', '.', ]
))

tops_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the top', 'the shirt', 'the long-sleeve top', 'the long-sleeve shirt', 'the T-shirt'],
methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(tops_description)

dress_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the dress', 'the long-sleeve dress'],
methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(dress_description)


possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the cloth', 'the garment', 'the long-sleeve cloth', 'the long-sleeve garment', ''],
methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
suffixes=['', '.', ]
))

tops_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the top', 'the shirt', 'the long-sleeve top', 'the long-sleeve shirt', 'the T-shirt'],
methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(tops_description)

dress_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the dress', 'the long-sleeve dress'],
methods=['from the right', 'from right to left', 'from the right side', 'from the right side to the middle', 'from the right side to the left side', 'from the right sleeve'],
suffixes=['', '.', ]
)
possible_description.extend(dress_description)


possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the cloth', 'the garment', 'the long-sleeve cloth', 'the long-sleeve garment', ''],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
))

tops_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the top', 'the shirt', 'the long-sleeve top', 'the long-sleeve shirt', 'the T-shirt'],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
)
possible_description.extend(tops_description)

dress_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the dress', 'the long-sleeve dress'],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
)
possible_description.extend(dress_description)

possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the pants', 'the trousers', ''],
methods=['from the left', 'from left to right', 'from the left side', 'from the left side to the middle', 'from the left side to the right side', 'from the left leg'],
suffixes=['', '.', ]
))

possible_description.extend(gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the pants', 'the trousers', ''],
methods=['bottom-up', 'from the bottom', 'from the bottom to the top', 'upward from the bottom'],
suffixes=['', '.', ]
))

print(possible_description)

print(len(possible_description))
total_len = len(possible_description)



loaded_description_dict = torch.load('description_embeddings.pt')

for i, description in enumerate(possible_description):                  # add some descriptions
    print(f'Now processing {i}/{total_len} :', description)
    description = description.replace("left", "temp").replace("right", "left").replace("temp", "right")
    if description not in loaded_description_dict:
        print('Description: ', description, ' not in dict.')
        description_embed = llm_embedding(llm_model, tokenizer, description)
        loaded_description_dict[description] = description_embed


delete_description = gen_descriptions(
prefixes=['Please fold ', 'please fold ', 'Fold ', 'fold '],
clothes=['the pants', 'the trousers'],
methods=['from the left side to the middle'],
suffixes=['', '.', ]
)

for i, description in enumerate(delete_description):                  # delete some descriptions
    print(f'Now delete {i}/{len(delete_description)} :', description)
    if description in loaded_description_dict:
        del loaded_description_dict[description]
    description = description.replace("left", "temp").replace("right", "left").replace("temp", "right")
    if description in loaded_description_dict:
        del loaded_description_dict[description]

torch.save(loaded_description_dict, 'description_embeddings_mirrored.pt')
print('Successful saved.')
