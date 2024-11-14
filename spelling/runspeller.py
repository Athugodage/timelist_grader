from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from razdel import sentenize
from tqdm import tqdm
import pandas as pd

import argparse

tqdm.pandas()


def correct(text):
    encodings = tokenizer(text, return_tensors="pt").to('cuda')
    generated_tokens = model.generate(
            **encodings, forced_bos_token_id=tokenizer.get_lang_id("ru"))
    answer = tokenizer.batch_decode(generated_tokens, max_length=len(text), skip_special_tokens=True)
    return answer


def correct_by_sents(text):
    full_corrected = []
    sents = list(sentenize(text))
    for sent in sents:
        cor_sent = correct(sent.text)[0]
        full_corrected.append(cor_sent)
    return ' '.join(full_corrected)


def run(path_to_model="ai-forever/RuM2M100-1.2B",
        datasetpath='saiga3keymoments_rlaif.parquet',
        savepath='keymoments_generated_n_corrected.parquet',
        ):

    global model, tokenizer
    model = M2M100ForConditionalGeneration.from_pretrained(path_to_model).to('cuda')
    tokenizer = M2M100Tokenizer.from_pretrained(path_to_model, src_lang="ru", tgt_lang="ru")

    df = pd.read_parquet(datasetpath)
    # for loop
    take_columns = [col for col in df.columns if col != 'summary' and col.endswith('corrected') == False]
    for col in take_columns:
        new_name = f'{col}_corrected'
        df[new_name] = df[col].progress_apply(correct_by_sents)

    df.to_parquet(savepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spelling')
    parser.add_argument('--datasetpath',
                        type=str,
                        required=True,
                        help='A path to .parquet dataset')
    parser.add_argument('--model',
                        type=str,
                        default="ai-forever/RuM2M100-1.2B",
                        help='Speller')
    parser.add_argument('--savepath',
                        type=str,
                        required=True,
                        help='A path to save annotated data')


    args = parser.parse_args()

    run(path_to_model=args.model,
        datasetpath=args.datasetpath,
        savepath=args.savepath,
        )




