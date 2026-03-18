# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import json
import os
from typing import List

import PIL
import numpy as np
import pandas
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from numpy._typing import NDArray
from tqdm import tqdm

from annflux.shared import AnnfluxSource


def numpy_load(
    path,
    key,
    check_for_split_format: bool = True,
    split_indices=None,
    select_per_split: List[List[int]] | None = None,
) -> np.ndarray:
    """
    Load implementation of numpy.load that supports split files
    """

    if not os.path.exists(path) and check_for_split_format:
        path = split_template(path)
    is_split_template = "{split_index}" in path

    if not is_split_template:
        array = np.load(path)[key]
    else:
        if split_indices is None:
            paths_to_load = get_part_paths(path)
        else:
            paths_to_load = [path.format(split_index=i_) for i_ in split_indices]

        array = []
        for split_index, part_path in tqdm(
            enumerate(paths_to_load), desc=f"Reading {path}"
        ):
            data: NDArray = np.load(part_path)[key]
            if select_per_split is not None:
                # noinspection PyTypeChecker
                data = data[select_per_split[split_index]]
                print("|data", len(data))
            array.append(data)

        array = np.vstack(array)
    return array


def split_template(path) -> str:
    """
    Returns the split file template for a path consisting of basename.{split_index}.ext
    """
    basename, extension = os.path.splitext(path)
    return basename + ".{split_index}" + extension


def get_part_paths(path):
    split_index = 0
    paths_to_load = []
    while os.path.exists(path.format(split_index=split_index)):
        paths_to_load.append(path.format(split_index=split_index))
        split_index += 1
    return paths_to_load


def read_table_pandas(
    filename, check_for_split_format=True, split_indices=None
) -> pandas.DataFrame:
    if not os.path.exists(filename) and check_for_split_format:
        filename = split_template(filename)
    is_split_template = "{split_index}" in filename

    if not is_split_template:
        table = pandas.read_csv(filename)
    else:
        if split_indices is None:
            paths_to_load = get_part_paths(filename)
        else:
            paths_to_load = [filename.format(split_index=i_) for i_ in split_indices]

        tables = []
        for part_path in tqdm(paths_to_load, desc=f"loading {filename}"):
            tables.append(pandas.read_csv(part_path))
        table = pandas.concat(tables, axis=0)
    return table


def create_directory(*path):
    if len(path) > 1:
        path = os.path.join(path[0], *path[1:])
    else:
        path = path[0]
    if not os.path.isdir(path):
        os.makedirs(path)

    return path


def file_hash(path):
    """
    Computes a hash from a file.
    :param path:
    :return:
    """
    block_size = 65536
    hasher = hashlib.sha224()
    with open(path, "rb") as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()


def basename_no_extension(path: str):
    return os.path.splitext(os.path.basename(path))[0]


def generate_missing_thumbnail(uid, size=(224, 224)) -> PIL.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), uid, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2

    draw.text((x, y), uid, font=font, fill="black")

    return image


def to_js_arrow(annflux_data_path, annflux_pq_cache_path):
    pandas.read_csv(
        annflux_data_path, dtype={"score_possible": str, "scores_predicted": str}
    ).to_parquet(annflux_pq_cache_path)

import pandas as pd

def sql_to_pandas_query(pseudo_sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Translates pseudo-SQL conditions (including AND, OR, and groups) into pandas queries.

    Args:
        pseudo_sql (str): Pseudo-SQL condition (e.g., '("Papilionidae" IN row.label_predicted) AND (row.value > 10)').
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame based on the pseudo-SQL condition.
    """
    # Replace SQL-like syntax with pandas-compatible syntax
    def parse_condition(condition):
        # Remove row. prefix and replace with df[]
        condition = condition.replace("row.", "df.")

        # Replace IN with str.contains
        not_in_condition = " NOT IN " in condition
        if not_in_condition:
            condition = condition.split()[-1] + " NOT IN " + " ".join(condition.split()[0:len(condition.split())-2])
            condition = "~" + condition.replace(" IN NOT ", ".str.contains('")
        in_condition = " IN " in condition
        if in_condition:
            condition = condition.split()[-1] + " IN " + " ".join(condition.split()[0:len(condition.split())-2])

        condition = condition.replace(" IN ", ".str.contains('")
        if in_condition or not_in_condition:
            condition += "', na$IS$False)"

        # Replace =, !=, >, <, >=, <= with pandas-compatible operators
        condition = (
            condition.replace(" = ", " == ")
            # .replace("!=", "!=")
            # .replace(">", ">")
            # .replace("<", "<")
            # .replace(">=", ">=")
            # .replace("<=", "<=")
        )

        # Replace IS NULL with .isna()
        condition = condition.replace(" IS NULL", ".isna()")

        # Replace IS NOT NULL with .notna()
        condition = condition.replace(" IS NOT NULL", ".notna()")

        # Replace LIKE with str.contains (note: this is a simplified version)
        if " LIKE " in condition:
            column, pattern = condition.split(" LIKE ", 1)
            pattern = pattern.strip().strip("'").replace("%", ".*")
            condition = f"{column}.str.contains(r'{pattern}', na=False, regex=True)"

        # Replace quoted strings with Python strings
        condition = condition.replace("'", '"')

        # condition += "]"

        return condition

    # Parse the entire pseudo-SQL into a pandas-compatible boolean expression
    def parse_expression(expr):
        # Split into clauses for AND/OR
        expr = expr.strip()

        # Handle parentheses (groups)
        while "(" in expr:
            start = expr.rfind("(")
            end = expr.find(")", start)
            if end == -1:
                raise ValueError("Mismatched parentheses in pseudo-SQL condition.")
            group = expr[start + 1:end]
            parsed_group = parse_expression(group)
            expr = expr[:start] + f"$OPEN${parsed_group}$CLOSE$" + expr[end + 1:]
            print(expr)

        # Split by AND/OR
        and_clauses = [c.strip() for c in expr.split(" AND ") if c]
        if len(and_clauses) > 1:
            return " & ".join(f"({parse_condition(c)})" for c in and_clauses)

        or_clauses = [c.strip() for c in expr.split(" OR ") if c]
        if len(or_clauses) > 1:
            return " | ".join(f"({parse_condition(c)})" for c in or_clauses)

        return parse_condition(expr)

    # Parse the pseudo-SQL into a pandas-compatible boolean expression
    try:
        boolean_expr = parse_expression(pseudo_sql).replace("$OPEN$", "(").replace("$CLOSE$", ")").replace("$IS$", '=')
    except Exception as e:
        raise ValueError(f"Failed to parse pseudo-SQL: {e}")

    # Evaluate the boolean expression safely
    try:
        # Create a dictionary of column names for eval()
        namespace = {col: df[col] for col in df.columns}
        mask = eval(boolean_expr) #, {"__builtins__": None}, namespace)
    except Exception as e:
        raise ValueError(f"Failed to evaluate pseudo-SQL: {e}, {boolean_expr}")

    print(f"{mask=}")
    return df[mask]

def compute_hash(input_string, algorithm='sha256'):
    # Create a hash object
    hash_object = hashlib.new(algorithm)

    # Update the hash object with the bytes of the string
    hash_object.update(input_string.encode('utf-8'))

    # Get the hexadecimal digest of the hash
    return hash_object.hexdigest()

if __name__ == "__main__":
    to_js_arrow(
        "/mnt/big/indeed/legasea_big/annflux/annflux.csv",
        "/mnt/big/indeed/legasea_big/annflux/annflux.arrow",
    )


def write_label_defs(annflux_folder: str, label_defs: list[tuple[str, str]]):
    source = AnnfluxSource(os.path.join(annflux_folder, ".."))
    os.makedirs(source.working_folder, exist_ok=True)
    with open(source.label_definitions_path, "w") as f:
        json.dump({"labels": label_defs}, f, indent=2)
