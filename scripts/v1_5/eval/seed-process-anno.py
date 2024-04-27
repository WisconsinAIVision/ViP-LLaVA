import json

json_path = './playground/data/eval/seed_bench/SEED-Bench.json'
with open(json_path, 'r') as f:
    data = json.load(f)

data_new = {}
data_new['question_type'] = data['question_type']
data_new['questions'] = []
# breakpoint() 
for question in data['questions']:
    if question['data_type'] == 'image':
            data_new['questions'].append(question)
with open('./playground/data/eval/seed_bench/SEED-Bench-image.json', 'w') as f:
      json.dump(data_new, f, indent=4)