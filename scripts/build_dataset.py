import json
import re
from typing import List, Dict
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing.semantic_map import SemanticMap
from preprocessing.subschema import extract_subschema
from utils.sqlite import introspect_db, schema_to_string, filter_schema
from utils.sql import extract_base_schema

def buid_semantic_map() -> SemanticMap:
    with open("data/train_bird.json", "r") as f:
        data = json.load(f)
        questions = [item["question"] for item in data]
        sqls = [item["SQL"] for item in data]
        semantic_map = SemanticMap()
        semantic_map.fit(questions, sqls)
        return semantic_map

def load_samples(mode="train") -> List[Dict]:
    if mode == "dev":
        with open("data/dev_bird.json", "r") as f:
            data = json.load(f)
            return data

    with open("data/filtered_train_bird.json", "r") as f:
        filtered_ds = json.load(f)
    with open("data/train_bird_think.json", "r") as f:
        think_ds = json.load(f)

    for item in think_ds:
        input_seq = item["input_seq"]
        output_seq = item["output_seq"]

        if "Question:" in input_seq:
            input_seq = input_seq.split("Question:")[1]
        if "Instructions:\n" in input_seq:
            input_seq = input_seq.split("Instructions:\n")[0]
        input_seq = re.sub(r"/s+", " ", input_seq).strip()
        
        reasoning = output_seq.split("<think>")[1].split("</think>")[0].strip()
        if "<answer>" in output_seq:
            output_seq = output_seq.split("<answer>")[1]
        if "</answer>" in output_seq:
            output_seq = output_seq.split("</answer>")[0]
        output_seq = re.sub(r"/s+", " ", output_seq).strip()

        item["input_seq"] = input_seq
        item["output_seq"] = output_seq
        item["reasoning"] = reasoning

    for item in filtered_ds:
        question = item["question"]
        question = re.sub(r"/s+", " ", question).strip()
        for i in range(len(think_ds)):
            think_item = think_ds[i]
            if item["question"] in think_item["input_seq"]:
                evidence = think_item["input_seq"].replace(question, "").strip()
                sql = think_item["output_seq"]
                reasoning = think_item["reasoning"]
                
                item["evidence"] = evidence
                item["SQL"] = sql
                item["reasoning"] = reasoning

                del think_ds[i]
                break

    return filtered_ds

def load_schemas(db_paths: List[str] | None = None, mode="train") -> Dict[str, Dict]:
    if mode == "train":
        with open("resources/train_column_meaning.json") as f:
            column_meaning = json.load(f)
    else:
        with open("resources/dev_column_meaning.json") as f:
            column_meaning = json.load(f)
    
    if db_paths is None:
        if mode == "train":
            with open("resources/schemas.json", "r") as f:
                schemas = json.load(f)
        else:
            with open("resources/dev_schemas.json", "r") as f:
                schemas = json.load(f)
    else:
        schemas = {}
        for db_path in db_paths:
            db_id = db_path.split("/")[-1].split(".sqlite")[0]
            schema = introspect_db(db_path)
            schemas[db_id] = schema
    
    for column_path, meaning in column_meaning.items():
        patterns = [
            r"The .+ column in the .+ table of the .+ database",
            r"In the .+ table of the .+ database, the .+ column",
            r"In the .+ database, within the .+ table, the .+ column",
            r"In the .+ database, the .+ table contains a .+ column",
            r"In the .+ database, the .+ table has an .+ column",
            r"The .+ column in the .+ table records the datetime",
            r"The .+ column in the .+ table .+ records",
            r"In the .+ table of .+ db, the .+ column",
            r"In the .+ table of .+ db, .+ is",
            r"In the .+ table, the .+ column",
            r"The .+ column in the .+ table",
            r"The .+ column",
        ]
        example_patterns = [
            r"\(\s*Example.+\)",
            r"Example.+",

            r"\(\s*with examples.+\)",
            r"with examples.+",

            r"\(\s*such as.+\)",
            r"such as.+",

            r"\(\s*with an example.+\)",
            r"with an example.+",

            r"\(\s*e\.g\..+\)",
            r"e\.g\..+",

            r"\(\s*exemplified.+\)",
            r"exemplified.+",

            r"\(\s*, with .+ as examples.+\)",
            r", with .+ as examples.+",

            r"\(\s*including example.+s\)",
            r"including example.+s",
        ]
        
        for pattern in patterns:
            match = re.match(pattern, meaning)
            if match:
                meaning = meaning[match.end():].lstrip(" ,.-")
                break
        
        for pattern in example_patterns:
            meaning = re.sub(pattern, "", meaning, flags=re.IGNORECASE)
        
        meaning = re.sub(r"\s+", " ", meaning).strip(" ,.-")

        parts = column_path.split("|")
        if len(parts) == 3:
            db_id, table_name, column_name = parts
            if db_id in schemas \
                and table_name in schemas[db_id] \
                and column_name in schemas[db_id][table_name]:
                schemas[db_id][table_name][column_name]["meaning"] = meaning
    
    return schemas

def generate_reasoning_prompt(
    item: Dict,
    example_item: Dict,
    schemas: Dict
) -> str:
    return f"""
You are an expert at generating step-by-step reasoning for a Text-to-SQL task.

Your goal is to analyze the Question, Evidence, Schema, and Ground Truth SQL, then generate a precise reasoning process that explains how the SQL is derived.

You MUST strictly follow the reasoning style, structure, granularity, and distribution of the provided Example.

# Schema:
{schema_to_string(filter_schema(schemas[item["db_id"]], extract_base_schema(item['SQL'])))}

# Question:
{item["question"]}

# Evidence:
{item["evidence"]}

# Groudth truth SQL:
{item['SQL']}

# Example:
## Example's schema:
{schema_to_string(filter_schema(schemas[example_item["db_id"]], extract_base_schema(example_item["SQL"])))}

## Example's question:
{example_item["question"]}

## Example's evidence:
{example_item["evidence"]}

## Example's grouth truth SQL:
{example_item["SQL"]}

## Example's reasoning output:
{example_item["reasoning"]}

# Instructions:

- Carefully analyze Question, Evidence, Schema, and Ground Truth SQL.
- Generate step-by-step reasoning that EXACTLY matches the logic of the Ground Truth SQL.
- Follow the SAME structure, wording style, reasoning depth, and ordering as the Example.
- Keep reasoning distribution consistent with the Example (no extra explanation, no missing steps).

- Clearly identify:
  • Target output (SELECT)
  • Relevant tables
  • Correct columns (no misalignment)
  • Join conditions (if any)
  • Filtering conditions (WHERE)
  • Aggregations (COUNT, SUM, AVG, MAX, MIN, etc.)
  • GROUP BY / ORDER BY / LIMIT (if present)

- Use ONLY schema elements that exist.
- Do NOT confuse similar column names.
- Do NOT invent tables, columns, or relationships.
- Ensure every clause in SQL is justified in reasoning.
- Ensure operators and comparison logic EXACTLY match the SQL.

- Do NOT generate SQL.
- Do NOT add extra commentary.
- Output ONLY the final reasoning.

Now generate the reasoning for the main sample.
"""

def generate_reasoning(data: List[Dict]) -> List[Dict]:
    for item in data:
        if "reasoning" in item:
            example = item
            break
    
    for item in data:
        prompt = generate_reasoning_prompt(item, example)
        
def generate_final_prompt(question, evidence, schema_description) -> str:
    return f"""
You are a expert in Text-to-SQL semantic parsing.

Your goal is to generate a syntactically correct and logically accurate SQLite query that answers the question EXACTLY based only on the provided schema and evidence.

You must strictly follow the instructions.

# Schema:
{schema_description}

# Question:
{question}

# Evidence:
{evidence}

# Instructions:

Instructions:
- Understand the exact intent of the question (aggregation, filtering, comparison, ranking, grouping, existence, etc.)
- Carefully map question phrases to exact tables and columns in the schema
- Use evidence only to help interpret schema references
- Do NOT invent tables or columns
- Use only columns that exist in the schema
- Use correct JOIN conditions based on foreign keys
- Apply correct SQL operations mentioned in the question (COUNT, SUM, AVG, MAX, MIN, DISTINCT, ORDER BY, GROUP BY, LIMIT, etc.)
- Select only necessary columns required to answer the question
- Avoid SELECT *
- Avoid redundant joins or conditions
- Ensure no missing filters or logic conditions
- Double-check correctness before final answer

Output format (strict):
- Put reasoning inside <think></think>
- Put final SQLite query inside <answer></answer>
- Generate exactly one SQL query
- Do not output anything after </answer>

Output:
"""

def build_dataset(data, schemas, semantic_map, mode="train"):
    final_data = []
    for item in data:
        question = item["question"]
        evidence = item["evidence"]
        sql = item["SQL"]
        db_id = item["db_id"]
        reasoning = item["reasoning"]

        subschema = extract_subschema(
            question,
            evidence,
            schemas[db_id],
            db_path=f"resources/{db_id}.sqlite",
            semantic_map=semantic_map
        )

        input_seq = generate_final_prompt(
            question, 
            evidence, 
            schema_to_string(subschema)
        )

        if mode == "train":
            output_seq = f"<think>{reasoning}</think><answer>{sql}</answer>"
        else:
            output_seq = sql

        final_data.append({
            "input_seq": input_seq,
            "output_seq": output_seq
        })

    return final_data


if __name__ == "__main__":
    data = load_samples()
    schemas = load_schemas()
    semantic_map = buid_semantic_map()
    generate_reasoning(data)
    final_data = build_dataset(data, schemas, semantic_map)
    with open("resources/final_train.json", "w") as f:
        json.dump(final_data, f, indent=2)
    
