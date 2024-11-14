from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
import re


prompt = '''Ты - преподаватель русского языка в университете. 
Тебе нужно оценить текст, написанный студентом по шкале от 0 до 100.
Тебе нужно оценить грамматику текста и его стиль.
В ответе укажи просто балл в виде цифры без каких либо объяснений. 
Например: "100", если текст идеальный. Другой пример: "0", если все неправильно.
Вот текст для оценки:\n'''


def getLLMscore(model_name='mistralai/Mistral-7B-Instruct-v0.2',
                temperature=0.1,
                column='saiga3_original',
                ):
    #'mistralai/Mistral-Nemo-Instruct-2407'

    llm = LLM(model=model_name, dtype='float16')
    sampling_params = SamplingParams(temperature=temperature,
                                     min_tokens=1,
                                     max_tokens=20,
                                     n=1)

    generated = df[column].to_list()
    prompts = [f'{prompt} "{example}". Теперь выстави балл пожалуйста. <|eot_id|>' for example in generated]
    outputs = llm.generate(prompts, sampling_params)

    scores = [outputs[n].outputs[0].text for n in range(len(outputs))]
    scores = [int(re.search('\d\d', score).group(0)) for score in scores if re.search('\d\d', score) != None]

    mean_score = np.mean(scores)
    return round(mean_score, 2)


def LLMevaluate(dataset_path='saiga3keymoments_rlaif.parquet',
                model_name='mistralai/Mistral-7B-Instruct-v0.2',
                temperature=0.1,
                ):
    result = {}

    global df
    df = pd.read_parquet(dataset_path)
    take_columns = [col for col in df.columns if col != 'summary' and col.endswith('corrected') == False]
    for col in take_columns:
        score = getLLMscore(model_name=model_name,
                            temperature=temperature,
                            column=col,
                            )
        result[col] = score

    return result

