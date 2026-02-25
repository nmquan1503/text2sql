from typing import List, Dict
import numpy as np
from rapidfuzz import fuzz
import pickle
import re
import wordninja

from utils.text import (
    text_to_canonical_form,
    to_snake_case
)

from utils.sqlite import (
    find_values
)

def transform_to_keys(texts: List[str]) -> List[str]:
    """
    Transform a list of names into keys composed of their rarest words.
    Returns a list of comma-separated keys, one per input text.

    text -> key1,key2
    """
    canonical_texts = [text_to_canonical_form(text) for text in texts]
    word_count = {}
    for text in canonical_texts:
        words = text.split()
        for w in words:
            word_count[w] = word_count.get(w, 0) + 1
    
    keys = []
    for text in canonical_texts:
        words = text.split()
        words = sorted(words, key=lambda w: word_count[w])
        if any(word_count[w] == 1 for w in words):
            words = [w for w in words if word_count[w] == 1]
        else:
            words = words[:1]
        words = [",".join(wordninja.split(word) + [word]) for word in words]
        keys.append(",".join(words))
    
    return keys

def _add_schema_item(
    subschema: Dict, 
    schema: Dict, 
    table_name: str,
    column_name: str | None = None,
    column_values: List | None = None
):
    if table_name not in schema:
        raise KeyError(f"Table '{table_name}' not found in schema")
    if column_name is not None:
        if column_name not in schema[table_name]:
            raise KeyError(f"Column '{column_name}' not found in '{table_name}'")

    if table_name not in subschema:
        subschema[table_name] = {}

    if column_name is not None:
        if column_name not in subschema[table_name]:
            subschema[table_name][column_name] = {
                "data_type": schema[table_name][column_name]["data_type"],
                "foreign_key": schema[table_name][column_name]["foreign_key"],
                "values": set()
            }
        elif column_values is not None:
            subschema[table_name][column_name]["values"].update(column_values)

def _extract_exact_columns(
    text: str,
    schema: Dict,
    subschema: Dict | None = None,
    remove_entities: bool = False
):
    if subschema is None:
        subschema = {}

    for table_name, tables in schema.items():
        for column_name in tables.keys():
            if column_name in text:
                _add_schema_item(subschema, schema, table_name, column_name)

    if remove_entities:
        for tables in subschema.values():
            for column_name in subschema.keys():
                text = text.replace(column_name, "")
        text = re.sub(r"\s+", " ", text)

    return subschema, text

def _extract_from_data(
    text: str,
    schema: Dict,
    db_path: str,
    subschema: Dict | None = None,
    remove_entities: bool = False
) -> Dict:
    if not subschema:
        subschema = {}
    
    cleaned_text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    
    entities = set()

    for table_name, table in schema.items():
        for column_name, column in table.items():
            value_prefixs = column["value_prefixs"]
            if not value_prefixs:
                continue
            for word in cleaned_text.split():
                word = word[:10]
                if word in value_prefixs:
                    if db_path:
                        values = find_values(db_path, word, table_name, column_name)
                    else:
                        values = []
                    entities.update(values)
                    _add_schema_item(subschema, schema, table_name, column_name, values)
                    break

    if remove_entities:
        for entity in entities:
            text = text.replace(entity, "")
        text = re.sub(r"\s+", " ", text)

    return subschema, text

def expand_semantics(text: str) -> str:
    text = re.sub(r"\btitle\b", "title, name", text, flags=re.IGNORECASE)
    text = re.sub(r"\bname\b", "title, name", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwho\b", "name, first name, last name, middle name, full name", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhich\b", "name, title", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhat\b", "name, title", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhen\b", "year, date, day, time", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhow many\b", "count, number, quantity", text, flags=re.IGNORECASE)   
    text = re.sub(r"\bpeople\b", "people, person", text, flags=re.IGNORECASE)
    text = re.sub(r"\bperson\b", "people, person", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfree\b", "free price, cost", text, flags=re.IGNORECASE)
    text = re.sub(r"\blist\b", "name, title", text, flags=re.IGNORECASE)
    text = re.sub(r"\bselling\b", "price, cost, bill, quantity", text, flags=re.IGNORECASE)
    return text

def generate_schema_keys(schema: Dict) -> Dict:
    key_map = {}

    table_names = list(schema.keys())
    table_keys = transform_to_keys(table_names)
    for name, key in zip(table_names, table_keys):
        key_map[name] = key

    for table_name, table in schema.items():
        column_names = list(table.keys())
        column_keys = transform_to_keys(column_names)
        for column_name, key in zip(column_names, column_keys):
            name = f"{table_name}|{column_name}"
            key_map[name] = key

    return key_map

def _update_foreign_keys(subschema: Dict, schema: Dict):
    additional_cols = []
    for table_name, table in subschema.items():
        for column_name, column in schema[table_name].items():
            if column["foreign_key"]:
                # Direct foreign key to another subschema table
                if column["foreign_key"]["ref_table"] in subschema:
                    additional_cols.append({
                        "table": table_name,
                        "column": column_name
                    })
                    additional_cols.append({
                        "table": column["foreign_key"]["ref_table"],
                        "column": column["foreign_key"]["ref_column"]
                    })
                
                # Foreign key through an intermediate table
                else:
                    mid_table_name = column["foreign_key"]["ref_table"]
                    for mid_table_colum_name, mid_table_column in schema[mid_table_name].items():
                        if mid_table_column["foreign_key"]:
                            if mid_table_column["foreign_key"]["ref_table"] in subschema:
                                additional_cols.append({
                                    "table": table_name,
                                    "column": column_name
                                })
                                additional_cols.append({
                                    "table": mid_table_name,
                                    "column": column["foreign_key"]["ref_column"]
                                })
                                additional_cols.append({
                                    "table": mid_table_name,
                                    "column": mid_table_colum_name
                                })
                                additional_cols.append({
                                    "table": mid_table_column["foreign_key"]["ref_table"],
                                    "column": mid_table_column["foreign_key"]["ref_column"]
                                })   
    for item in additional_cols:
        _add_schema_item(subschema, schema, item["table"], item["column"])

def extract_subschema(
    question: str, 
    evidence: str,
    schema: Dict,
    db_path: str,
    semantic_map = None
) -> Dict:
    """
    Extract the subschema relevant to the question.

    Args:
        question: Natural language question.
        schema: Full database schema.
    Returns:
        A subschema containing only tables and columns relevant to the question.
    """

    subschema, evidence = _extract_exact_columns(
        text=evidence,
        schema=schema,
        remove_entities=True
    )
    subschema, _ = _extract_exact_columns(
        text=question,
        schema=schema,
        subschema=subschema,
        remove_entities=False
    )
    full_question = evidence + " " + question
    subschema, full_question = _extract_from_data(
        text=full_question,
        schema=schema,
        db_path=db_path,
        subschema=subschema,
        remove_entities=True
    )

    full_question = expand_semantics(full_question)
    full_question = text_to_canonical_form(full_question)

    key_map = generate_schema_keys(schema)

    # Select tables relevant to the question
    for table_name in schema.keys():
        keys_str = key_map[table_name]
        keys = keys_str.split(",")
        for k in keys:       
            # If key appears in the question     
            if k in full_question:
                _add_schema_item(subschema, schema, table_name)
    
    # Select columns relevant to the question
    # Initialize table entry in subschema if not already present
    for table_name, table in schema.items():
        for column_name, column in table.items():
            keys_str = key_map[f"{table_name}|{column_name}"]
            keys = keys_str.split(",")
            for k in keys:
                # If key appears in the question
                if k in full_question:
                    _add_schema_item(subschema, schema, table_name, column_name)

    _update_foreign_keys(subschema, schema)

    # Remove tables with no selected columns
    subschema = {
        table_name: table
        for table_name, table in subschema.items()
        if table
    }

    for table in subschema.values():
        for column in table.values():
            column["values"] = list(column["values"])

    return subschema