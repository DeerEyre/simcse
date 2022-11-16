import jsonlines
n=0
save = jsonlines.open('test.jsonl', 'w')
with open('old.jsonl', 'r', encoding='utf8') as f:
    for item in jsonlines.Reader(f):
        save.write({'query': item['title'], 'content': item['content']})
save.close()
