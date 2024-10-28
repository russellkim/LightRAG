import os
import json
import sys

def extract_data(WORKING_DIR, json_file_path, num):
    idx = -1
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                idx += 1
                if idx != num : continue
                data = json.loads(line.strip())
                qid = data.get("id", "")                
                paragraphs = data.get("paragraphs", [])
                question = data.get("question", "")
                answer = data.get("answer", "")
                answerable = data.get("answerable", False)

                print(f"id: {qid}")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f"Answerable: {answerable}\n")

                # Filter entries where "is_supporting" is true
                for entry in paragraphs:
                    if entry.get("is_supporting") == True:
                        title = entry.get("title", "No title provided")
                        paragraph_text = entry.get("paragraph_text", "No paragraph text provided")
                        print(f"Title: {title}")
                        print(f"Paragraph: {paragraph_text}\n")                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                exit(-1)


if __name__=='__main__':
    # Check if the argument [num] is passed
    if len(sys.argv) != 2:
        print("Usage: python get_support.py [num]")
        sys.exit(1)

    # Retrieve the [num] argument from the command line
    num = sys.argv[1]
    print(f"Input number: {num}")

    BASE_DIR = f"./exp/musique_ans_train/contents"

    # 사용 예시
    json_file_path = 'data/musique_ans_v1.0_train.jsonl'
    extracted_data = extract_data(BASE_DIR, json_file_path, int(num))
