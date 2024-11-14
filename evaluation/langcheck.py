from collections import Counter
from evaluate import load
from razdel import tokenize
from langcheck_llm import LLMevaluate
import numpy as np
import pandas as pd
import joblib
import re



perplexity = load("perplexity", module_type="metric")
protocol_clf = joblib.load(f'evaluation/classifier/case5grams_large.pkl')


def check_mistakes(text2check, correctedtext, row_id):
    threshold = 4
    a = re.sub('\n', ' ', text2check)
    b = correctedtext
    tokens_orig = [token.text for token in list(tokenize(a)) if len(token.text) >= threshold]
    tokens_corr = [token.text for token in list(tokenize(b)) if len(token.text) >= threshold]
    mistakes = 0

    assert len(tokens_orig) == len(tokens_corr), f'Arrays are not same, row {row_id}'
    for n in range(len(tokens_orig)):
        if tokens_orig[n] != tokens_corr[n]:
            mistakes += 1
            # print(tokens_orig[n], tokens_corr[n])

    ratio = round(mistakes / len(tokens_orig) * 100, 3)
    return float(ratio)


def process_column(column2check, correctedcolumn):
    mistake_ratios = []
    length = len(df[column2check])
    for i in range(length):
        try:
            ratio = check_mistakes(text2check=df[column2check].iloc[i],
                                   correctedtext=df[correctedcolumn].iloc[i],
                                   row_id=i)
            mistake_ratios.append(ratio)
        except Exception as err:
            # print(err)
            pass

    return np.mean(mistake_ratios)


def check_gibberish(column_name):
    gibberish = []
    code = []
    results = {}

    for n in range(len(df[column_name])):
        lines = df[column_name].iloc[n].split('\n')
        lines4prediction = [line[:200] for line in lines]
        preds = protocol_clf.predict(lines4prediction)

        cnt = Counter(preds)
        gibbratio = cnt['gibberish'] / len(preds) * 100
        coderatio = cnt['code'] / len(preds) * 100

        gibberish.append(gibbratio)
        code.append(coderatio)
    results['gibberishscore'] = 100 - np.mean(gibberish)
    results['codescore'] = 100 - np.mean(code)
    return results


def see_perplexity(column_name):
    passages = [sample[:500] for sample in df[column_name].to_list()]
    results = perplexity.compute(predictions=passages, model_id='gpt2')
    return results['mean_perplexity']


def get_report(path='keymoments_generated_n_corrected.parquet',
               ask_llm=True,
               llm2score='mistralai/Mistral-7B-Instruct-v0.2'):

    global df
    df = pd.read_parquet(path)
    d = {}
    take_columns = [col for col in df.columns if col != 'summary' and col.endswith('corrected') == False]
    for col in take_columns:
        ratio = process_column(column2check=col,
                               correctedcolumn=f'{col}_corrected')
        results = check_gibberish(col)
        results['spelling'] = 100 - round(ratio, 2)
        results['perplexity'] = see_perplexity(col)
        d[col] = results

    if ask_llm == True:
        llmscores = LLMevaluate(dataset_path=path,
                                model_name=llm2score
                                )
        d['llmscoring'] = llmscores
    return d




