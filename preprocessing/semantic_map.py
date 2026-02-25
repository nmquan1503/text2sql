from typing import List, Tuple
import re
import sqlglot
import sqlglot.expressions as exp

from utils.sql import extract_base_schema
from utils.text import text_to_canonical_form

class SemanticMap:
    def __init__(self):
        self.map = {}

    def _remove_entity(self, question: str, sql: str):
        try:
            expr = sqlglot.parse_one(sql)
            for literal in expr.find_all(exp.Literal):
                question = re.sub(literal.name, "", question, flags=re.IGNORECASE)
        except:
            pass
        return question

    def fit(self, questions: List[str], sqls: List[str]):
        word_count = {}
        
        for i in range(len(questions)):
            questions[i] = self._remove_entity(questions[i], sqls[i])
            questions[i] = " ".join([word.lower().strip(".,?!") for word in questions[i].split()])
            questions[i] = text_to_canonical_form(questions[i])

        for question, sql in zip(questions, sqls):
            try:
                base_schema = extract_base_schema(sql)
                keys = set()
                for table_name, column_names in base_schema.items():
                    table_keys = text_to_canonical_form(table_name).split()
                    keys.update(table_keys)
                    for column_name in column_names:
                        column_keys = text_to_canonical_form(column_name).split()
                        keys.update(column_keys)
                for word in set(question.split()):
                    word_count[word] = word_count.get(word, 0) + 1
                    if word not in self.map:
                        self.map[word] = keys.copy()
                    else:
                        self.map[word].intersection_update(keys)
            except:
                pass
        
        for question in questions:
            words = question.split()
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    a = words[i]
                    b = words[j]
                    if a not in self.map or b not in self.map:
                        continue
                    common = self.map[a] & self.map[b]
                    self.map[a] -= common
                    self.map[b] -= common

        self.map = {word:keys for word, keys in self.map.items() if keys and word_count[word] > 2}
    
    def display(self):
        if self.map:
            for word, keys in self.map.items():
                print('-' * 10)
                print(f"Word: {word}")
                print(f"Keys: {list(keys)}")


