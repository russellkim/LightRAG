import os
import json

def extract_data(WORKING_DIR, json_file_path):
    idx = -1
    q_list = []
    q_dic = {}
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                idx += 1
                data = json.loads(line.strip())
                qid = data.get("id", "")                
                #qtype = 2hop__465230_21416
                qtype = qid.split("_")[0]
                paragraphs = data.get("paragraphs", [])
                question = data.get("question", "")
                answer = data.get("answer", "")
                answerable = data.get("answerable", False)

                if qtype in q_dic:
                    q_list = q_dic[qtype]
                    q_list.append(f"{idx}:{question}")
                else:
                    q_dic[qtype] = [f"{idx}:{question}"] 

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                exit(-1)

            #print(f"{idx} : {qtype}: {question}")

        # 추출된 데이터를 파일로 저장하려면 아래와 같이 사용 가능
        for key, value in q_dic.items():
            with open(f'{WORKING_DIR}/{key}.txt', 'w', encoding='utf-8') as outfile:
                for q in value:
                    outfile.write(f"{q}\n")

if __name__=='__main__':
    BASE_DIR = f"./exp/musique_ans_train"
    for d in ['prompt', 'contents', 'qna']:
        WORKING_DIR = f"{BASE_DIR}/{d}"
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)

    # 사용 예시
    json_file_path = 'data/musique_ans_v1.0_train.jsonl'
    extracted_data = extract_data(BASE_DIR, json_file_path)
