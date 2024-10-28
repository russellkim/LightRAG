import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding, gpt_4o_complete, gpt_4o_mini_complete
from lightrag.utils import EmbeddingFunc
import numpy as np
import json
import logging
from openai import OpenAI # openai==1.2.0
 
USE_SOLAR = 0
#SOLAR_MODEL = "solar-mini"
SOLAR_MODEL = "solar-pro"
   
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    print("len(prompt)=", len(prompt))
    print("prompt=", prompt[:20])
    #print("len(system_prompt)=", len(system_prompt))
    #print("len(history_messages)=", len(history_messages))
    return await openai_complete_if_cache(
        SOLAR_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar"
    )

# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


def decompse_question(original_question):
    if USE_SOLAR == 1:
        client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar"
        )
         
        def chat_with_solar(messages):
            response = client.chat.completions.create(
                model=SOLAR_MODEL,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
    else:
        client = OpenAI(
            api_key=os.getenv("OPEN_API_KEY")
            #base_url="https://api.oepnai.com/v1/chat/completions"
        )
         
        def chat_with_solar(messages):
            response = client.chat.completions.create(
                #model="gpt-4o-mini",
                model="gpt-4o",
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content

    decomposition_prompt = (
       # f"The following questionis a multi-hop question. "
        f"Please decompose the question to answer correctly. \n"
        f"When you decompose it and make new following questions,"
        f"you assumed that we do not know the answer of the intermediate question.\n"
        f"You should decompse questions in bullet style directly, without any explanation and extra information.\n"
        f"Question: {original_question}\n"
        f"\n#####################\n"
        f"-Example- \n"
        f"#####################\n"
        f"Question: "
        f"What is the record label for the person who sang Beauty and the Beast with Celine Dion?\n"
        f"\nExpected Answer: \n"
        f"Who sang 'Beauty and the Beast' with Celine Dion?\n"
        f"What is the record label for previous answer's person?\n"
    )

    messages=[ { "role": "user", "content": decomposition_prompt } ]
    response = chat_with_solar(messages)

    list_q = [q.strip() for q in response.split('\n') if q.strip()]
    list_q = [ item for item in list_q 
        if not (item.startswith("To find the answer") or item.startswith("To answer the question"))
    ]
    list_q = [q.lstrip('- ').strip() for q in list_q]
    list_q = [q.lstrip('* ').strip() for q in list_q]
    print("decompose question: ", list_q)
    return list_q 


def get_answers_fromlightrag(question_type, idx, list_questions):
    contents_file_path = f'{BASE_DIR}/contents'
    with open(f'{contents_file_path}/{idx}.txt', 'r', encoding='utf-8') as file:
        contents = file.read()

    qna_file_path = f'{BASE_DIR}/qna'
    with open(f'{qna_file_path}/{idx}.txt', 'r', encoding='utf-8') as file:
        try:
            data_list = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            exit(-1)

    # extract each problem
    for data in data_list:
        question = data.get("question", "")
        answer = data.get("answer", "")
        answerable = data.get("answerable", False)

    WORKING_DIR = f"{BASE_DIR}/storage/{idx}" 
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    list_questions = []
    if question_type != 'single':
        list_questions = decompse_question(question)
    #exit(-1)

    if USE_SOLAR == 1:
        print("!!!!! USE SOLAR !!!!!")
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=4096,
                #max_token_size=8192,
                max_token_size=2048,
                func=embedding_func
            )
        )
    else:
        print("##### USE OPENAI #####")
        print("##### gpt_4o_mini #####")
        #print("##### gpt_4o #####")
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete
            # llm_model_func=gpt_4o_complete
        )

    #print(combined_paragraphs)
    rag.insert(contents)

    #question_type = 'single'
    #question_type = 'multi-hop'

    #processed_data = []
    dic_info ={}
    dic_info['question'] = question
    dic_info['answerable'] =answerable 
    dic_info['answer'] = answer 
    if question_type == 'all' or question_type == 'single':
        print("\n-----")
        # Perform hybrid search ( naive, local, global, hybrid)
        model_rps = rag.query(dic_info['question'], param=QueryParam(mode="hybrid"))
        print(model_rps)
        dic_info['a_single'] = model_rps 

    if question_type == 'all' or question_type == 'connected':
        _question = ""
        for q in list_questions:
            if _question != "" : _question += ' And then '
            _question += q 
        print("question=", _question)
        model_rps = rag.query(_question, param=QueryParam(mode="hybrid"))
        print(model_rps)
        dic_info['q_concat'] = _question 
        dic_info['a_concat'] = model_rps 
    
    if question_type == 'all' or question_type == 'multi-hop':
        prev_q = None 
        for _question in list_questions:
            question_orig = _question
            if prev_q != None: 
                _question += f"\nprevious question is: \n {prev_q}"
                _question += f"\nprevious answer is: \n {prev_a}"
            model_rps = rag.query(_question, param=QueryParam(mode="hybrid"))
            print(_question)
            print(model_rps)
            print("+++")
            prev_q = question_orig
            prev_a =  model_rps
        dic_info['q_multi'] = list_questions 
        dic_info['a_multi'] = model_rps
    print("===")
    print(idx, "question=", question)
    print("answerable=", answerable, "answer=", answer)
    print(f"{contents_file_path}/{idx}.txt", len(contents))
    print("===")

    with open(f'{BASE_DIR}/response/{idx}.json', 'w', encoding='utf-8') as outfile:
        #processed_data.append(dic_info)
        json.dump([dic_info], outfile, indent=4, ensure_ascii=False)        

def test_one_problem(idx, question_type = 'all'):
    question_type = 'single'
    #question_type = 'single'
    #question_type = 'multi-hop'
    #question_type = 'connected'
    #list_question = [
    #        "Who is the designer of Peter and Paul Fortress?",
    #        "Where did the previous answer's person die?"
    #] 
    list_question = []
    get_answers_fromlightrag(question_type, idx, list_question)

if __name__ == '__main__':
    BASE_DIR = 'exp/musique_ans_train'

    for idx in range(23):
        #if idx not in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        #if idx not in [31]:
        #    continue
        try:
            test_one_problem(idx)
        except Exception as e:
            print(f"Error(idx={idx})= ", e)
            continue

