import random
import collections
import numpy as np
import re 
import json
from llava.visual_prompt_generator import image_blending, color_pool, words_shape




def build_prompt(question, options):
    """
    Build a prompt string based on the given question and options.

    Parameters:
        question (str): The question to be asked.
        options (list): List of options for the question.

    Returns:
        str: The formatted prompt string.
    """
    if len(options) != 4:
        return "Error: Exactly 4 options are required."
    
    # Create the options string
    options_str = '\n'.join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
    
    # Build the prompt
    prompt = f"""{question}
{options_str}
Answer with the option's letter from the given choices directly."""
    
    return prompt



def add_period_and_autocorrect(annotation):
    # List of common abbreviations that should not be split
    abbreviations = ['Dr.', 'Mrs.', 'Mr.', 'Ms.', 'e.g.', 'i.e.', 'U.S.A.']

    # Replace abbreviations with placeholders
    for i, abbr in enumerate(abbreviations):
        annotation = annotation.replace(abbr, f"__ABBREVIATION{i}__")
    
    annotation = annotation.strip()
    annotation = annotation[0].upper() + annotation[1:]

    
    if not annotation[-1] in ['.', '!', '?']:
        annotation += '.'
    annotation = re.sub(r'\s*,\s*', ', ', annotation)
    for i, abbr in enumerate(abbreviations):
        annotation = annotation.replace(f"__ABBREVIATION{i}__", abbr)
    
    return annotation





WHY_QUESTIONS = [
    'why?',
    'why',
    "What's the rationale for your decision?",
    'What led you to that conclusion?',
    "What's the reasoning behind your opinion?",
    'Why do you believe that to be true?',
    'Can you explain the basis for your thinking?',
    'What factors influenced your perspective?',
    'How did you arrive at that perspective?',
    'What evidence supports your viewpoint?',
    'What makes you think that way?',
    "What's the logic behind your argument?",
    'Can you provide some context for your opinion?',
    "What's the basis for your assertion?",
    'Why do you hold that belief?',
    'What experiences have shaped your perspective?',
    'What assumptions underlie your reasoning?',
    "What's the foundation of your assertion?",
    "What's the source of your reasoning?",
    "What's the motivation behind your decision?",
    "What's the impetus for your belief?",
    "What's the driving force behind your conclusion?",
    'Why do you think that?',
    "What's your reasoning?",
    'What makes you say that?',
    'Why do you feel that way?',
    "What's the story behind that?",
    "What's your thought process?",
    "What's the deal with that?",
    "What's the logic behind it?",
    'Why do you believe that?',
    "What's the real deal here?",
    "What's the reason behind it?",
    "What's the thought process behind your decision?",
    "What's the rationale for your opinion?",
    'Why do you have that impression?',
    "What's the background to that?",
    "What's the evidence that supports your view?",
    "What's the explanation for that?"
]



answer_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}
    
def get_adjective():
    return random.choice(['The correct', 'The most accurate', 'The best', 'The ultimate', 'The final', 'The only', 'The ideal', 'The optimal', 'The most fitting', 'The definitive'])

def get_punctuation():
    return random.choice([':', '->', '→', '::', '—', ';', '|', '⇒'])

def get_answer(choice, content, use_multichoice_why):
    choice =answer_map[choice]
    choice_upper = choice.upper()
    if use_multichoice_why:
        # Added different styles and structures
        content = content[0].lower() + content[1:] if content else content
        content = random.choice([
            f'({choice_upper})',
            f'({choice_upper})',
            f'{get_adjective()} answer is ({choice_upper})',
            f'{get_adjective()} answer is ({choice_upper})',
            f'({choice_upper}){get_punctuation()} {content}',
            f'({choice_upper}){get_punctuation()} {content}',
            f'{get_adjective()} answer is ({choice_upper}) — {content}',
            f'{get_adjective()} answer is ({choice_upper}) — {content}',
            f'({choice_upper}) — {get_adjective()} because {content}',
            f'({choice_upper}) — {get_adjective()} because {content}',
            f'Answer ({choice_upper}): {content}',
            f'Answer ({choice_upper}): {content}',
            f'Opt for ({choice_upper}) if {content}',
            f'Opt for ({choice_upper}) if {content}'
        ])
        return content.replace("\u2014", "-")
    else:
        return content


question_prefixes = [
    'Based on the provided source image, please answer this question: ',
    'In the context of the source image, can you answer: ',
    'With reference to the source image, please respond to the following query: ',
    "Considering the source image, what's your answer to: ",
    'Please provide an answer for the subsequent question, keeping the source image in mind: ',
    'Taking into account the source image, please answer: ',
    'After observing the source image, could you please answer the following: ',
    'Upon examining the source image, what would your answer be to: ',
    'Using the source image as a reference, please respond to: ',
    'In light of the source image, could you please answer: '
]


options_prefixes = [
    'Available choices are as follows: ',
    'Select from the options below: ',
    'You may choose from the following: ',
    'Your choices include: ',
    'Here are your options: ',
    'Please pick one from the given possibilities: ',
    'The following options are available: ',
    'You have the following selections: ',
    'Which among these would you choose: ',
    'You can select from these alternatives: '
]



questions = {
    "semantic": [
        "Please describe the image with the object referred to by the visual prompts; please do not mention the actual visual prompt.",
        "Describe the provided image using the semantic object referred to by the visual prompts. Please produce a sentence in natural language, and do not mention the actual visual prompts."
    ],
    "visual_prompt": [
        "Please describe the image with the object referred to by the visual prompts; please just mention the actual visual prompt and do not mention the semantic category.",
        "Please describe the image with the object referred to by the visual prompts; please just mention the actual visual prompt, such as a red box, and do not mention the semantic category, such as a dog."
    ],
    "semantic_visual_prompt": [
        "Please describe the image with the object referred to by the visual prompts; make sure to mention both the actual visual prompt and the semantic category.",
        "Please describe the image with the object referred to by the visual prompts; make sure to mention both the actual visual prompt, such as a red box, and the semantic category, such as a dog."
    ]
}





def generate_conversation(convs):
    conv = []
    for (human_conv, gpt_conv) in convs:
        conv.extend([
            {"from": "human", "value": human_conv},
            {"from": "gpt", "value": gpt_conv},
        ]
        )
        

def vip_conv_generator(source, sampled_shapes, dataset_type, sub_type = ''):
    convs_source = []
    if dataset_type == 'refcocog':
        if sub_type == 'gpt4v':
            color_name, _, shape= sampled_shapes[0]
            word1, word2 = words_shape[shape]
            color_string = f' {color_name}' if color_name!= None else ''
            text = f'{word1} the{color_string} {word2}'
            for i in range(len(source['conversations'])):
                source['conversations'][i]['value'] = source['conversations'][i]['value'].replace('<bbox>', text)
            source['conversations'][0]['value'] = '<image>\n' + source['conversations'][0]['value']
            return source['conversations']
        else:
            if random.random()<0.25:
                prompt = random.choice([f'Describe the object with the visual prompt.', f'Describe the pointed region.']) 
            else:
                color_name, _, shape= sampled_shapes[0]
                word1, word2 = words_shape[shape]
                color_string = f' {color_name}' if color_name!= None else ''
                prompt = f'Describe the object .'
            prompt += ' Please provide a short phrase.'
            convs_source.append([prompt, source['answer']])
    elif dataset_type == 'vg_rel':
        if sub_type == 'gpt4v':
            for bbox_index, ( color_name, _ , predefined_shape) in enumerate( sampled_shapes ):
                word1, word2 = words_shape[predefined_shape]
                text = word1 +' '
                if random.random()<0.5:
                    text += 'the '
                if color_name is not None:
                    text += color_name + ' '
                text += word2
                for i in range(len(source['conversations'])):
                    source['conversations'][i]['value'] = source['conversations'][i]['value'].replace(f'<bbox{bbox_index}>', text)
            return source['conversations']
        else:
            prompts = []
            for  color_name, _ , shape in sampled_shapes:
                word1, word2 = words_shape[shape]
                color_string = f' {color_name}' if color_name!= None else ''
                prompts.append(f'{word1} the{color_string} {word2}')
            prompt = f"Please describe the relationship between the subject {prompts[0]} and the object {prompts[1]}. Provide a short triplet (subject, relationship, object) to represent this. Here, the subject and object are noun phrases, and the relationship can be verbs or prepositions."
            convs_source.append([prompt, source['answer']])
    conv = [] 
    for (human_conv, gpt_conv) in convs_source:
        conv.extend([
            {"from": "human", "value": human_conv},
            {"from": "gpt", "value": gpt_conv},
        ]
        )
        
    return conv



def get_all_instances(all_corpus):
    all_instance_index= []
    for corpus in all_corpus:
        for instance in corpus:
            if type(instance) == list:
                 all_instance_index.extend(instance)
    all_instance_index = list(set(all_instance_index))
    return all_instance_index


def get_color_shape(all_instance_index, shape_choices, color_list):
    # Step 1: Randomly sample shapes
    shapes = random.choices(shape_choices, k=len(all_instance_index))
    shape_counts = collections.Counter(shapes)
    non_unique_shapes = {shape for shape, count in shape_counts.items() if count > 1}

    # Step 2: Initialize the list to store the results
    results = {}
    
    # Create a dictionary to keep track of shapes and their corresponding colors
    shape_color_dict = {}
    
    for i, instance in enumerate(all_instance_index):
        shape = shapes[i]
        
        # Initialize shape in shape_color_dict if it doesn't exist
        if shape not in shape_color_dict:
            shape_color_dict[shape] = []
        
        # Check if this shape is already chosen or not unique
        if shape in shape_color_dict and shape_color_dict[shape] or shape in non_unique_shapes:
            available_colors = [color for color in color_list if color[0] not in shape_color_dict[shape]]
            if available_colors:
                color_name, color_rgb = random.choice(available_colors)
                shape_color_dict[shape].append(color_name)
            else:
                color_name = None
                color_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            if random.choice([True, False]):
                color_name, color_rgb = random.choice(color_list)
            else:
                color_name = None
                color_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            if color_name:
                shape_color_dict[shape].append(color_name)
        
        results[instance] = [color_name, color_rgb, shape]
    
    return results

def get_all_qa(all_corpus, shape_color_info, class_names, answer_type = ''):
    all_text= []
    shape_color_info_visual_prompt_image = []
    for corpus in all_corpus:
        text = ''
        for instance_index, instance in enumerate(corpus):
            if type(instance) == list:
                for obejct_index in range(len(instance)):
                    shape_color = shape_color_info[instance[obejct_index]]
                    if instance_index == 0 and obejct_index == 0:
                        text += 'The '
                    else:
                        text += ' the '
                    if class_names==None:
                        text += 'object'
                    elif random.random()<0.5 and answer_type != 'direct':
                        text += random.choice(['object', 'instance']) 
                    else:
                        text +=  class_names[instance[obejct_index]]
                    word1, word2 = words_shape[shape_color[2]]
                    text += ' '+word1 +' '
                    if random.random()<0.5:
                        text += 'the '
                    if shape_color[0] is not None:
                        text += shape_color[0] + ' '
                    text += word2

                    if obejct_index!=len(instance)-1:
                        text += ' and'
                    shape_color_info_visual_prompt_image.append(instance[obejct_index])
            elif type(instance) == str:
                text +=  instance
            else:
                breakpoint()
            if instance_index!=len(corpus)-1 and type(corpus[instance_index+1])==str:
                if corpus[instance_index+1] not in {'.', ',', '?', '!', ':', ';'}:
                    text += ' '
                    
                    
        all_text.append(text)
            

    return all_text, shape_color_info_visual_prompt_image




def get_question(question, all_choices, use_multichoice_question, why_question = False, no_image=False ): 
    # question is a string like "What is the color of object?"
    # all_choices is a list of strings like ["red", "green", "blue"]
    # Mapping for shuffled choices
    if why_question:
        question_prompt = random.choice(WHY_QUESTIONS)
    else:
        image_str = '' if no_image else '<image>\n'
        question_prompt = image_str + random.choice(question_prefixes) + question
    if use_multichoice_question:
        all_options = ''
        for choice_index, choice in enumerate(all_choices):
            choice = '(' + answer_map[choice_index] + ') ' + choice
            all_options += choice
            if choice_index != len(all_choices) - 1:
                all_options += ' '
            else:
                all_options += ''
        question_prompt +=  " " + random.choice(options_prefixes) + all_options
    # Using random.choice instead of random.sample for single items
    return question_prompt



def create_question_direct_qa(line, shape_choices, color_list):
    question =[ line['question']]
    answer = line['answer_choices']
    all_corpus = question + answer
    all_instance_index =  get_all_instances(all_corpus)
    shape_color_info = get_color_shape(all_instance_index, shape_choices, color_list)
    class_names = line['class_names']
    shape_color_info_visual_prompt_image_all = []
    question, shape_color_info_visual_prompt_image = get_all_qa(question, shape_color_info, class_names, answer_type = 'direct')
    question = question[0]
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
    answer, shape_color_info_visual_prompt_image  = get_all_qa(answer, shape_color_info, class_names, answer_type = 'direct')
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
    
    question_prompt ='<image>\n' + build_prompt(question, answer)
    question_answer_prompt = answer_map[line['answer_label']]  
    
    
    conversations=  [
            {
                "from": "human",
                "value":question_prompt
            },
            {
                "from": "gpt",
                "value": question_answer_prompt
            }, 
            ]   
    shape_color_info =  [shape_color_info[instance_index] for instance_index in all_instance_index]
    return shape_color_info,all_instance_index, conversations           
              
def create_question_direct_qar(line, shape_choices, color_list):
    question =[ line['question']]
    org_answer = [ line['answer_choices'] [line['answer_label']] ]
    why_answer = line['rationale_choices']
    all_corpus = question + org_answer + why_answer
    all_instance_index =  get_all_instances(all_corpus)
    
    shape_color_info = get_color_shape(all_instance_index, shape_choices, color_list)
    class_names = line['class_names']
    shape_color_info_visual_prompt_image_all = []
    question, shape_color_info_visual_prompt_image = get_all_qa(question, shape_color_info, class_names, answer_type = 'direct')
    question = question[0]
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
    
    org_answer, shape_color_info_visual_prompt_image = get_all_qa(org_answer, shape_color_info, class_names, answer_type = 'direct')
    org_answer = org_answer[0]
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
    
    why_answer, shape_color_info_visual_prompt_image  = get_all_qa(why_answer, shape_color_info, class_names, answer_type = 'direct')
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
    
    
    question_prompt = build_prompt('', why_answer)
    
    why_answer_prompt  =  answer_map[line['rationale_label']]
    conversations=  [             
              {
                  "from": "human",
                  "value":  '<image>\n' + f'I give you a question and its answer, I need you to provide a rationale explaining why the answer is right. "{question}" The answer is "{org_answer}".What is the rationale for this decision?{question_prompt}' 
              },
              {
                  "from": "gpt",
                  "value": why_answer_prompt
              }
          ]
    shape_color_info =  [shape_color_info[instance_index] for instance_index in all_instance_index]
    return shape_color_info,all_instance_index, conversations



    

def create_question_prompt(line, shape_choices, color_list):
    use_multichoice_question = random.random()<0.5
    use_multichoice_why = random.random()<0.5
    question =[ line['question']]
    if not use_multichoice_question:
        answer = [ line['answer_choices'][line['answer_label']] ]
    else:
        answer = line['answer_choices'] 
        
    if not use_multichoice_why:
        why_answer = [ line['rationale_choices'][line['rationale_label']] ]
    else:
        why_answer = line['rationale_choices']
    all_corpus = question + answer + why_answer
    all_instance_index =  get_all_instances(all_corpus)
    
    shape_color_info = get_color_shape(all_instance_index, shape_choices, color_list)
    class_names = line['class_names']
    shape_color_info_visual_prompt_image_all = []
    question, shape_color_info_visual_prompt_image = get_all_qa(question, shape_color_info, class_names)
    question = question[0]
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
    answer, shape_color_info_visual_prompt_image  = get_all_qa(answer, shape_color_info, class_names)
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
    why_answer, shape_color_info_visual_prompt_image  = get_all_qa(why_answer, shape_color_info, class_names)
    shape_color_info_visual_prompt_image_all.extend(shape_color_info_visual_prompt_image)
        
        
    question_prompt = get_question(question, answer, use_multichoice_question)
    answer_index = line['answer_label'] if use_multichoice_question else 0
    question_answer_prompt = get_answer(answer_index, answer[answer_index], use_multichoice_question)
    why_prompt = get_question(None, why_answer, use_multichoice_why, why_question=True)
    why_answer_index = line['rationale_label'] if use_multichoice_why else 0
    why_answer_prompt = get_answer(why_answer_index, why_answer[why_answer_index], use_multichoice_why)
    conversations=  [
              {
                  "from": "human",
                  "value":question_prompt
              },
              {
                  "from": "gpt",
                  "value": question_answer_prompt
              },              
              {
                  "from": "human",
                  "value":why_prompt
              },
              {
                  "from": "gpt",
                  "value": why_answer_prompt
              }
          ]
    shape_color_info =  [shape_color_info[instance_index] for instance_index in all_instance_index]
    return shape_color_info,all_instance_index, conversations



def create_question_prompt_flicker30k(line, shape_choices, color_list):
    describe_mode = random.choice(["semantic","semantic_visual_prompt" ]) 
    question = random.choice(questions[describe_mode])
    
    all_instance_index = range(len(line['bbox']))
    caption = line["grounding"]
    # print(caption, '\n', line['bbox'])
    shape_color_info = get_color_shape(all_instance_index, shape_choices, color_list)

    use_visual_prompt_hint =  random.random() < 0.5
    if use_visual_prompt_hint:
        question += random.choice([" Hint: the visual prompts are:", " The visual prompts are:"])
        for instance_index in all_instance_index:
            shape_color = shape_color_info.get(instance_index, (None, None, None))
            if shape_color[0] is not None : # and random.random() < 0.5
                question += ' ' + shape_color[0]  
            question += ' ' +  words_shape[shape_color[2]][1]
            if instance_index != len(all_instance_index) -1:
                question += ','
            if instance_index == len(all_instance_index) -2:
                question += ' and'
        question += '.'
                
    def replace_bbox(match):
        idx = int(match.group(1))
        shape_color = shape_color_info.get(idx, (None, None, None))
        if idx < len(line['bbox']):
            if describe_mode == "semantic":
                return ""
            elif describe_mode == "visual_prompt":
                assert False
            elif describe_mode == "semantic_visual_prompt":
                if shape_color[0]!= None:
                    return f" {words_shape[shape_color[2]][0]} the {shape_color[0]} {words_shape[shape_color[2]][1]}"
                elif shape_color[0]!= None:
                    return f" {words_shape[shape_color[2]][0]} the {words_shape[shape_color[2]][1]}"
        else:
            assert False
    
    question_answer_prompt = re.sub(r' <bbox(\d+)>', replace_bbox, caption)
    question_answer_prompt = add_period_and_autocorrect(question_answer_prompt)
    question_prompt = '<image>\n' + question
    
    conversations = [
        {
            "from": "human",
            "value": question_prompt
        },
        {
            "from": "gpt",
            "value": question_answer_prompt
        },              
    ]
    shape_color_info_new = []
    bboxes_all = []
    
    for i in all_instance_index:
        for j in range(len(line['bbox'][i])):
            shape_color_info_new.append(shape_color_info[i])
            bboxes_all.append(line['bbox'][i][j])
    
    return shape_color_info_new,conversations, bboxes_all



def create_question_prompt_direct(line, shape_choices, color_list, answer_type = ''):
    question =[ [line['question']]]
    line['answer_label'] = line['bboxes'].index(line['answer'])

    answer = [[[i]] for i in range(len(line['bboxes']))]
        
    all_corpus = question + answer #  + why_answer
    all_instance_index =  get_all_instances(all_corpus)
    shape_color_info = get_color_shape(all_instance_index, shape_choices, color_list)
        
    class_names = None 
    question= get_all_qa(question, shape_color_info, class_names, answer_type = answer_type)[0][0]
    answer= get_all_qa(answer, shape_color_info, class_names, answer_type = answer_type, )[0]
    question_prompt = build_prompt(question, answer)
    question_answer_prompt =  answer_map[line['answer_label']]
    conversation =  [
              {
                  "from": "human",
                  "value":  '<image>\n' + question_prompt
              },
              {
                  "from": "gpt",
                  "value": question_answer_prompt
              },              
          ]
    shape_color_info = [ shape_color_info[instance_index] for instance_index in all_instance_index]
    bboxes_all = [ line["bboxes"][instance_index] for instance_index in all_instance_index] 
    return shape_color_info,  conversation, bboxes_all


def create_question_prompt_direct_pointQA(line, question_type = 'general_question'):
    shape_color_info = [['red', (255, 0, 0), 'rectangle']]
    if type(question_type) == list:
        question_type_target = random.choice(question_type)
    elif type(question_type) == str:
        question_type_target = question_type
    conversation =  [
              {
                  "from": "human",
                  "value":  '<image>\n' + line[question_type_target] + ' The exemplary object is within the rectangle.'+"\nAnswer the question using a single word or phrase."
              },
              {
                  "from": "gpt",
                  "value": line['answer']
              },              
          ]
    return shape_color_info, conversation






visual_prompt_config = dict(
    refcocog=[ ["rectangle", "ellipse","triangle", "point", "scribble" ,  "mask contour" , "mask", "arrow"], ''],
    vcr = [ ["rectangle", "ellipse","triangle", "scribble" ,  "mask contour" , "mask", "arrow"], ''],
    vg_rel =  [ ["rectangle", "ellipse", ], ''], #  [ ["rectangle", "ellipse", "arrow"], ''],
    flickr30k =[ ["rectangle", "ellipse", "arrow"], ''],
    v7w = [ ["rectangle"], 'constant'],
    pointQA_twice = [ ["rectangle"], 'constant'],     
) 

visual_prompt_config_test = dict(
       vcr_qa = [ ["point"], 'constant'],
       vcr_qar = [ ["point"], 'constant'],
       
)
        
def vip_processor(source, image, image_size_anchor, data_args):
    dataset_type, sub_type = source['id'].split('-')[0],  source['id'].split('-')[1]
    if getattr(data_args, "visual_prompt_style", None) != None:
        visual_prompt_shape_choices, visual_prompt_style  = visual_prompt_config_test[data_args.visual_prompt_style]
    else:
        visual_prompt_shape_choices, visual_prompt_style  = visual_prompt_config[dataset_type]
    
    if dataset_type in {'vg_rel', 'v7w', 'pointQA_twice'}:
        source['segmentations'] = [None] * len(source['bboxes'])
        
        
    if dataset_type in {'vcr'}:
        source['meta_dir'] = source['meta_dir'].replace('./dataset', data_args.image_folder)
        meta_data = json.load(open(source['meta_dir']))
        if getattr(data_args, "visual_prompt_style", None) == 'vcr_qa':
            shape_color_info, all_instance_index, conversation  = create_question_direct_qa(source, visual_prompt_shape_choices, color_list = list(color_pool.items()) )
        elif getattr(data_args, "visual_prompt_style", None) == 'vcr_qar':
            shape_color_info, all_instance_index, conversation  = create_question_direct_qar(source, visual_prompt_shape_choices, color_list = list(color_pool.items()) )
        else:
            shape_color_info, all_instance_index, conversation  = create_question_prompt(source, visual_prompt_shape_choices, color_list = list(color_pool.items()) )
        source['bboxes'] = [meta_data['boxes'][instance_index][:-1] for instance_index in all_instance_index]
        source['segmentations'] = []
        for instance_index in all_instance_index:
            segmentation_data = []
            for seg_index in range(len(meta_data['segms'][instance_index])-1, -1, -1):
                if len(meta_data['segms'][instance_index][seg_index]) >=4:
                    segmentation_data.append( list(np.array(meta_data['segms'][instance_index][seg_index]).flatten()) )
            if len(segmentation_data)>0:
                source['segmentations'].append(segmentation_data)
            else:
                source['segmentations'].append(None)
    elif dataset_type in {'flickr30k'}:
        shape_color_info, conversation, bboxes = create_question_prompt_flicker30k(source, visual_prompt_shape_choices, color_list = list(color_pool.items()) )
        source['bboxes'] = bboxes
        source['segmentations'] = [None] * len(source['bboxes'])
    elif dataset_type in {'v7w'}:
        shape_color_info, conversation, bboxes = create_question_prompt_direct(source, visual_prompt_shape_choices, color_list = list(color_pool.items()), answer_type = 'direct' )
        source['bboxes'] = bboxes
    elif dataset_type in {'pointQA_twice'}:
        shape_color_info, conversation = create_question_prompt_direct_pointQA(source )
    else:
        used_colors = [] 
        predefined_shapes = [ random.choice(visual_prompt_shape_choices)  for _ in range(len(source['bboxes']))]
        if dataset_type in {'vg_rel'}:
            prob_random = 0 if predefined_shapes[0] == predefined_shapes[1] else 0.5
        else:
            prob_random = 0.5
        
        color_rgb = None
        shape_color_info = [] 
        for instance_idx, (bbox, segmentation) in enumerate(zip(source['bboxes'], source['segmentations'])):
            while color_rgb is None or color_rgb in used_colors:
                if random.random() < prob_random:
                    color_name, color_rgb = None, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color_name, color_rgb = random.choice(list(color_pool.items()))
            if prob_random == 0 :
                used_colors.append(color_rgb)
            shape_color_info.append([color_name, color_rgb, predefined_shapes[instance_idx]])
        conversation = vip_conv_generator(source, shape_color_info, dataset_type, sub_type = sub_type)
    # alpha = data_args.alpha if 'alpha' in data_args else None
    alpha = getattr(data_args, "alpha", None)
    for instance_idx, (bbox, segmentation) in enumerate(zip(source['bboxes'], source['segmentations'])):
        color_name, color_rgb, sampled_shape = shape_color_info[instance_idx] # random.choice(visual_prompt_shape_choices)
        image = image_blending(image,  shape = sampled_shape, image_size_anchor = image_size_anchor, rgb_value=color_rgb, bbox_coord= bbox, segmentation=segmentation, visual_prompt_style = visual_prompt_style, alpha = alpha)
    
    # from matplotlib import pyplot as plt
    # image.save('tmp.png')
    # print(conversation)
    # breakpoint()
    return image, conversation



