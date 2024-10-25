import os
import json
import re
from openai import OpenAI # openai==1.2.0

def verifiy_answer(idx):

    BASE_DIR = 'exp/musique_ans_train'
    rps_file_path = f'{BASE_DIR}/response/{idx}.json'
    with open(rps_file_path, 'r', encoding='utf-8') as file:
        contents = file.read()	

    client = OpenAI(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar"
    )

    def chat_with_solar(messages):
        response = client.chat.completions.create(
            model="solar-pro",
            messages=messages
        )
        return response.choices[0].message.content

    prompt = (
        f"The correct answer is 'answer.' Given three responses (a_single, a_concat, a_multi), identify which one is incorrect."
        f"{contents}\n"
    )

    messages=[ { "role": "user", "content": prompt } ]
    response = chat_with_solar(messages)

    #list_q = [q.strip() for q in response.split('\n') if q.strip()]
    #list_q = [ item for item in list_q
    #    if not (item.startswith("To find the answer") or item.startswith("To answer the question"))
    #]
    #list_q = [q.lstrip('- ').strip() for q in list_q]

   
    # Add an "incorrect" field as an empty list for each dictionary in the list
    # 'no incorrect' 패턴이 문장에 포함된 경우
    no_incorrect_pattern = r"There are no incorrect responses"
    if re.search(no_incorrect_pattern, response, re.IGNORECASE):
        incorrect_fields = []
    else:
        # 틀린 응답을 찾기 위한 정규식
        incorrect_fields = re.findall(r"\b(a_single|a_concat|a_multi)\b", response)

    print(f"idx={idx} : incorrect = {incorrect_fields}")
    # Parse the JSON content
    # Convert the modified data back to JSON format
    with open(rps_file_path+'2', 'w', encoding='utf-8') as outfile:
        data = json.loads(contents)
        for item in data:
            item['incorrect'] = incorrect_fields
            item['verify'] = response
        json.dump(data, outfile, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    for idx in range(50):
        try:
            verifiy_answer(idx)
        except Exception as e:
            print(f"Error[idx={idx}]", e)
            continue

