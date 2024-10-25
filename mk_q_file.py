import os
import json

def extract_data(WORKING_DIR, json_file_path):
    idx = -1
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                idx += 1
                data = json.loads(line.strip())
                paragraphs = data.get("paragraphs", [])
                question = data.get("question", "")
                answer = data.get("answer", "")
                answerable = data.get("answerable", False)

                # paragraphs의 title과 text를 원하는 형식으로 결합
                combined_paragraphs = "\n\n".join(
                    f"{p.get('title', '')}\n{p.get('paragraph_text', '')}" for p in paragraphs
                )
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                exit(-1)


            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Answerable: {answerable}")

            extracted_data = "please, solve the following [question] based on [text]\n-----\n"
            extracted_data += f"[question]:{question} \n\n[text]:{combined_paragraphs}"


            # 추출된 데이터를 파일로 저장하려면 아래와 같이 사용 가능
            with open(f'{WORKING_DIR}/prompt/{idx}.txt', 'w', encoding='utf-8') as outfile:
                outfile.write(extracted_data)

            # 추출된 데이터를 파일로 저장하려면 아래와 같이 사용 가능
            with open(f'{WORKING_DIR}/contents/{idx}.txt', 'w', encoding='utf-8') as outfile:
                outfile.write(combined_paragraphs)

            # 추출된 데이터를 파일로 저장하려면 아래와 같이 사용 가능
            processed_data = [] 
            with open(f'{WORKING_DIR}/qna/{idx}.txt', 'w', encoding='utf-8') as outfile:
                processed_data.append({"question":question, "answer":answer, "answerable":answerable})
                json.dump(processed_data, outfile, indent=4, ensure_ascii=False)

if __name__=='__main__':
    BASE_DIR = f"./exp/musique_ans_train"
    for d in ['prompt', 'contents', 'qna']:
        WORKING_DIR = f"{BASE_DIR}/{d}"
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)

    # 사용 예시
    json_file_path = 'data/musique_ans_v1.0_train.jsonl'
    extracted_data = extract_data(BASE_DIR, json_file_path)
