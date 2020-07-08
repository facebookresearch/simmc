"""Collection of support tools for reading the Furniture / Fashion data.

Author(s): Paul Crook, Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import csv
import functools
import os
import math
import numpy as np
import sqlite3

# Static values for accessing annotation

# Message Sender / Speaker
USER = "user"
ASSISTANT = "assistant"

# Annotation field names
ANNOTATED_USER_TRANSCRIPT = "transcript_annotated"
ANNOTATED_ASSISTANT_TRANSCRIPT = "system_transcript_annotated"


class ExponentialSmoothing:
    """Exponentially smooth and track losses.
    """

    def __init__(self):
        self.value = None
        self.blur = 0.95
        self.op = lambda x, y: self.blur * x + (1 - self.blur) * y

    def report(self, new_val):
        """Adds a new value.
        """
        if self.value is None:
            self.value = new_val
        else:
            self.value = {
                key: self.op(value, new_val[key]) for key, value in self.value.items()
            }
        return self.value


def setup_cuda_environment(gpu_id):
    """Setup the GPU/CPU configuration for PyTorch.
    """
    if gpu_id < 0:
        print("Running on CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return False
    else:
        print("Running on GPU {0}...".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return True


def pretty_print_dict(parsed):
    """Pretty print a parsed dictionary.
    """
    max_len = max(len(ii) for ii in parsed.keys())
    format_str = "\t{{:<{width}}}: {{}}".format(width=max_len)
    print("Arguments:")
    # Sort in alphabetical order and print.
    for key in sorted(parsed.keys()):
        print(format_str.format(key, parsed[key]))
    print("\n")


def sort_eval_metrics(eval_metrics):
    """Sort a dictionary of evaluation metrics.

    Args:
        eval_metrics: Dict of evaluation metrics.

    Returns:
        sorted_evals: Sorted evaluated metrics, best first.
    """
    # Sort based on 'perplexity' (lower is better).
    # sorted_evals = sorted(eval_metrics.items(), key=lambda x: x[1]['perplexity'])
    # return sorted_evals

    # Sort based on average %increase across all metrics (higher is better).
    def mean_relative_increase(arg1, arg2):
        _, metric1 = arg1
        _, metric2 = arg2
        rel_gain = []
        # higher_better is +1 if true and -1 if false.
        for higher_better, key in [
            (-1, "perplexity"),
            (1, "action_accuracy"),
            (1, "action_attribute"),
        ]:
            rel_gain.append(
                higher_better
                * (metric1[key] - metric2[key])
                / (metric1[key] + metric2[key])
            )
        return np.mean(rel_gain)

    sorted_evals = sorted(
        eval_metrics.items(),
        key=functools.cmp_to_key(mean_relative_increase),
        reverse=True,
    )
    return sorted_evals


def read_furniture_metadata(metadata_path):
    """Reads the CSV file for furniture metadata.

    Args:
        metadata_path: Path to the CSV file for furniture assets

    Returns:
        assets: Dictionary of assets with attribute dictionary
    """
    print("Reading: {}".format(metadata_path))
    with open(metadata_path, "r") as file_id:
        csv_reader = csv.reader(file_id)
        rows = list(csv_reader)
    assert all(len(ii) == len(rows[0]) for ii in rows), "Invalid CSV file!"
    # Convert into a dictionary of attribute dictionaries.
    keys = rows[0]
    assets = {}
    for row_id, row in enumerate(rows[1:]):
        new_asset = {keys[index]: row[index] for index in range(len(keys))}
        new_asset["id"] = int(new_asset["obj"].split("/")[-1].split(".")[0])
        new_asset["row_id"] = row_id
        assets[new_asset["id"]] = new_asset
    return assets


class FurnitureDatabase:
    """ Class object that encapsulates an in memory sqlite3 database
    containing the furniture metadatai, and provides methods to query the data
    """

    # class variable: sqlite table holding furniture metadata
    METADATA_TABLE = "furniture"

    def __init__(self, metadata_path):
        """ Constructor that reads the furniture metadata CSV file and creates a
        indatabase table holding the asset details.

        Args:
            metadata_path: Path to the CSV file for furniture assets
        """
        # create database in memory
        self.conn = sqlite3.connect(':memory:')
        # format query response rows as dictionaries
        self.conn.row_factory = self._dict_factory
        # cursor for operating on database
        self.cur = self.conn.cursor()

        # TODO: extract common read from disk from this and share with above read-to-dict method
        with open(metadata_path, 'r') as handle:
            reader = csv.reader(handle)
            headers = next(reader)
            data_rows = list(reader)

        # add a prefab 'id' column
        headers.append('id')
        for row in data_rows:
            row.append(row[headers.index('obj')].split('/')[-1].split('.zip')[0])
        # NB: Although appealing in concept DON'T make 'id' an INTEGER PRIMARY KEY.
        # SQLite has an auto _rowid_ column that preserves the item ordering as
        # read from the CSV file. This ordering needs to be retained to reproduce
        # the carousel ordering that was seen by the human assisant and user.
        # Adding an INTERGER PRIMARY KEY aliases _rowid_ thus breaking that
        # ordering; see https://www.sqlite.org/lang_createtable.html#rowid .

        # ensure sale_price column is marked as containing REAL values
        table_spec = [
            header if header != 'sale_price' else 'sale_price REAL' for header in headers
        ]

        # create table
        self.cur.execute(f"CREATE TABLE {self.METADATA_TABLE} ({', '.join(table_spec)});")
        self.cur.executemany(
            f"INSERT INTO {self.METADATA_TABLE} ({', '.join(headers)}) VALUES ({', '.join(['?'] * len(headers))});",
            data_rows
        )
        self.conn.commit()

    def get_min_max_price_per_class(self):
        """Extract the minimum and maximum prices for each furniture class.

        Returns:
            price_dict: Dictionary of min-max prices for each furniture class
        """
        self.cur.execute(f"""
        SELECT class_name, min(sale_price), max(sale_price) FROM {self.METADATA_TABLE}
        GROUP BY class_name
        """)
        price_dict = {}
        for row in self.cur.fetchall():
            price_dict[row['class_name']] = (
                row['min(sale_price)'],
                row['max(sale_price)']
            )
        return price_dict

    def search_furniture(self, args):
        """ Search sqlite METADATA_TABLE using given arguments

        Args:
            args: dictionary of search arguments key-values

        Returns:
            List of matching furniture prefab ids
        """
        # example input args:
        # {'furnitureType': 'Kitchen Islands', 'color': 'Gray', 'material': '',
        #  'decorStyle': '', 'intendedRoom': '', 'minPrice': -1, 'maxPrice': 1000.0}

        # METADATA_TABLE fields:
        # ['sku', 'product_name', 'product_description', 'product_page_url',
        #  'class_name', 'sale_price REAL', 'thumbnail_image_url', 'obj', 'glb',
        #  'x_dim', 'y_dim', 'z_dim', 'color', 'material', 'decor_style',
        #  'intended_room']

        # arg translation table:
        arg_map = {
            'furnitureType': {'field': 'class_name', 'operator': '='},
            'color': {'field': 'color', 'operator': 'LIKE'},
            'material': {'field': 'material', 'operator': 'LIKE'},
            'decorStyle': {'field': 'decor_style', 'operator': 'LIKE'},
            'intendedRoom': {'field': 'intended_room', 'operator': 'LIKE'},
            'minPrice': {'field': 'sale_price', 'operator': '>='},
            'maxPrice': {'field': 'sale_price', 'operator': '<='},
        }
        # build WHERE clause
        where = []
        values = []
        for arg, value in args.items():
            field = arg_map[arg]['field']
            op = arg_map[arg]['operator']
            # assuming ignore min and max price if -1 or 0.0
            if (field == 'sale_price' and value > 0) or \
                    (field != 'sale_price' and value):
                where.append(f'{field} {op} ?')
                if op == 'LIKE':
                    values.append(f"%{value}%")
                    # values.append(f"%'{value}'%") <-- seems overly precise
                    # when compared to the results seen in the Wizard UI
                elif arg == 'minPrice':
                    values.append(math.floor(value))
                elif arg == 'maxPrice':
                    values.append(math.ceil(value))
                else:
                    values.append(value)
        query = f"SELECT id FROM {self.METADATA_TABLE} WHERE {' AND '.join(where)}"
        self.cur.execute(query, values)
        return [row['id'] for row in self.cur.fetchall()]

    def get_basic_info(self, furniture_id):
        """Return a basic set of furniture information give the furniture_id

            Args:
                furniture_id: furniture id

            Returns:
                List of dicts contanting basic furniture information for items
                who's id matches. Empty list if id doesn't match any.
                Expectation is that an id is unique to only one item.
        """
        self.cur.execute(f"""
        SELECT class_name, product_name, intended_room, color, material, decor_style, sale_price
        FROM {self.METADATA_TABLE}
        WHERE id == ?
        """, furniture_id)
        return self.cur.fetchall()

    def shutdown(self):
        """Shutdown the database by ending connection
        """
        self.conn.close()

    @staticmethod
    def _dict_factory(cursor, row):
        """Static method to return query results formatted as a dict per row
        """
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d


def get_intents(speaker, round_datum):
    """ Get speaker's intents from NLU/NLG annotation.

    Args:
        speaker: either USER or ASSISTANT
        round_datum: user-assistant turn-pair, including annotations

    Returns:
        list of intents.
    """
    if speaker == USER:
        key = ANNOTATED_USER_TRANSCRIPT
    elif speaker == ASSISTANT:
        key = ANNOTATED_ASSISTANT_TRANSCRIPT
    else:
        raise Exception(
            f"Invalid speaker argument value '{speaker}' passed to get_intents(.)"
        )
    # note annotation stored as python in a string
    annotation = eval(round_datum[key])
    all_intents = [ii["intent"] for ii in annotation]
    return all_intents


def get_object_references(turn_label, reversed_dialog_coref_map):
    """From turn label data extract local object references associated with
    each action and also resolve them to global furniture_ids

    Args:
        turn_labels: action, object, slot annotations for this turn
        reversed_dialog_coref_map: maps per dialog object ids to furniture id

    Returns:
        dictionary index by actions containing a list of tuples of the format
        (local id, global furniture_id)
    """
    action_refs = {}
    for action_obj_slot in turn_label:
        action = action_obj_slot['act']
        local_ids = [ii['obj_idx'] for ii in action_obj_slot['objects']]
        action_refs[action] = [
            (ii, reversed_dialog_coref_map[ii]) for ii in local_ids
        ]
    return action_refs
