import csv
import json

path = 'ielts_writing_dataset.csv'
csv_column = ["Task_Type", "Question", "Essay", "Overall"]
json_column = ["task_type", "question", "answer", "score"]
output_path = 'ielts_writing_dataset_graph_only.json'
json_data = []
Task_Type = ["1"]


with open(path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in rows:
    json_row = {}
    for i in range(len(csv_column)):
        if row["Task_Type"] in Task_Type:
            json_row[json_column[i]] = row[csv_column[i]]
    if row["Task_Type"] in Task_Type:
        json_data.append(json_row)

with open(output_path, 'w') as f:
    json.dump(json_data, f, indent=4)

