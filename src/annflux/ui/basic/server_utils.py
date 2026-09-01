# Copyright 2025 Intel Corporation
# Copyright 2025 Naturalis Biodiversity Center
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

"""
Server utility functions for AnnFlux
"""

import os
import pandas
import flask
from flask import make_response, Response
from functools import wraps
from typing import Dict, Any


def normalize_filter_query(filter_query: str) -> str:
    """
    Normalize filter query to handle both simple and pseudo-SQL formats.
    
    Args:
        filter_query: Raw filter query from request
        
    Returns:
        Normalized query in pseudo-SQL format
    """
    if not filter_query:
        return filter_query
    
    # If query already uses row. prefix, assume it's already in pseudo-SQL format
    if "row." in filter_query:
        return filter_query
    
    # Convert simple format like "label_predicted = 'label1'" 
    # to pseudo-SQL format like "row.label_predicted = 'label1'"
    
    # Handle simple equality conditions
    if " = " in filter_query and ("'" in filter_query or '"' in filter_query):
        # Split on the equals sign
        parts = filter_query.split(" = ", 1)
        if len(parts) == 2:
            column = parts[0].strip()
            value = parts[1].strip()
            # Add row. prefix to column if it doesn't already have it
            if not column.startswith("row."):
                normalized = f"row.{column} = {value}"
                return normalized
    
    # Handle IN conditions
    if " IN " in filter_query:
        parts = filter_query.split(" IN ", 1)
        if len(parts) == 2:
            column = parts[0].strip()
            values = parts[1].strip()
            if not column.startswith("row."):
                normalized = f"row.{column} IN {values}"
                return normalized
    
    # Handle NOT IN conditions
    if " NOT IN " in filter_query:
        parts = filter_query.split(" NOT IN ", 1)
        if len(parts) == 2:
            column = parts[0].strip()
            values = parts[1].strip()
            if not column.startswith("row."):
                normalized = f"row.{column} NOT IN {values}"
                return normalized
    
    # If no patterns match, return original query
    return filter_query


def standard_json_response(error_code: str, error_message: str | None, http_status_code: int) -> Response:
    """
    Create a standard JSON error response.

    Args:
        error_code: Error code string
        error_message: Error message string
        http_status_code: HTTP status code

    Returns:
        JSON response with compiled message
    """
    if error_message is None:
        error_message = " ".join(error_code.split("_")).capitalize()
    
    from annflux.tools.mixed import get_logger
    logger = get_logger("annflux_server")
    logger.info("status: {}, error message: {}".format(http_status_code, error_message))
    
    response = make_response(
        flask.jsonify({"error": {"code": error_code, "message": error_message}}),
        http_status_code,
    )
    return response


def get_group_uids(g_state, group_images_path: str) -> set[str]:
    """
    Get group UIDs from group data file.
    
    Args:
        g_state: AnnFlux state object
        group_images_path: Path to group images directory
        
    Returns:
        Set of group UIDs
    """
    global group_uids
    
    group_data_path = os.path.join(
        g_state.project_folder, "annflux", "group0_annflux.csv"
    )

    if os.path.exists(group_data_path):
        group_uids = set(pandas.read_csv(group_data_path)["uid"])
    return group_uids


# Global variable for group UIDs
group_uids = set()
