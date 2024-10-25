import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
import json

    
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
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


if __name__ == '__main__':

    json_file_path = 'exp/musique/extracted_data.json'
    #asyncio.run(test_funcs())

    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as file:
        try:
            data_list = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            exit(-1)

    processed_data = []

    idx = -1   
    for data in data_list:
        idx += 1
        if idx not in [6, 7, 8, 9, 10]:
            continue

        WORKING_DIR = f"./exp/musique/index{idx}" 
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=4096,
                max_token_size=8192,
                func=embedding_func
            )
        )

        # extract each problem
        paragraphs = data.get("paragraphs", [])
        question = data.get("question", "")
        answer = data.get("answer", "")
        answerable = data.get("answerable", False)

        # make one string
        combined_paragraphs = "\n\n".join(
                f"{p.get('title','')}\n{p.get('paragraph_text','')}" for p in paragraphs
        )

        #print(combined_paragraphs)
        rag.insert(combined_paragraphs)

        print("\n-----")
        # Perform hybrid search ( naive, local, global, hybrid)
        print(rag.query(question, param=QueryParam(mode="hybrid")))

        print(idx, "question=", question)
        print("answerable=", answerable, "answer=", answer)
        print("-----")

