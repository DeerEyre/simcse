import jsonlines
import random
import pandas as pd
original_sentence = []
similartiy_sentence = []
contractive_sentence = []

with open('deepl_translate_60W.jsonl', 'r+', encoding='utf8') as f:
    for item in jsonlines.Reader(f):
        original_sentence.append(item['original_sentence'])
        similartiy_sentence.append(item['translate_back_sentence'])
        contractive_sentence.append(item['translate_back_sentence'])

random.shuffle(contractive_sentence)
for i in range(len(original_sentence)):
    while similartiy_sentence[i] == contractive_sentence[i]:
        contractive_sentence[i] = random.choice(similartiy_sentence)

# print(len(original_sentence), len(similartiy_sentence), len(contractive_sentence))
# print(original_sentence[0]+'\n', similartiy_sentence[0]+'\n', contractive_sentence[0]+'\n')


dataframe = pd.DataFrame({'sen0': original_sentence, 'sen1': similartiy_sentence, 'hard_neg': contractive_sentence})
dataframe.to_csv("deepl_train_data1.csv", index=False, sep=',')

