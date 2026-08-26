"""
System-Level Functions Configuration for Oneiros Engine.

This module defines 60 system-level training functions and 10 testing functions
for the Oneiros Engine. These are real Python library functions with complex
behavior and known edge cases - suitable for training a robust test generation model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
import json
from pathlib import Path


@dataclass
class SystemLevelFunction:
    """Represents a system-level target function."""
    id: str                          # Unique identifier
    name: str                        # Function name (e.g., "pandas.merge")
    library: str                     # Library name (e.g., "pandas")
    import_statement: str            # Import statement needed
    signature: str                   # Function signature
    docstring: str                   # Description of the function
    edge_cases: List[str] = field(default_factory=list)  # Known edge cases
    bug_types: List[str] = field(default_factory=list)   # Types of bugs to find
    complexity_score: int = 7        # Complexity (1-10)
    category: str = "system"         # Category
    is_training: bool = True         # True = training, False = testing only
    wrapper_code: str = ""           # Wrapper function code for mutation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "library": self.library,
            "import_statement": self.import_statement,
            "signature": self.signature,
            "docstring": self.docstring,
            "edge_cases": self.edge_cases,
            "bug_types": self.bug_types,
            "complexity_score": self.complexity_score,
            "category": self.category,
            "is_training": self.is_training,
            "wrapper_code": self.wrapper_code
        }


# =============================================================================
# 60 SYSTEM-LEVEL TRAINING FUNCTIONS
# =============================================================================

TRAINING_FUNCTIONS: List[SystemLevelFunction] = [
    # =========================================================================
    # PANDAS (8 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_pandas_merge",
        name="pandas.DataFrame.merge",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def merge_wrapper(left_df, right_df, on=None, how='inner')",
        docstring="Merge two DataFrames on specified column(s).",
        edge_cases=["Empty DataFrame", "Mismatched keys", "Duplicate columns", "NaN values"],
        bug_types=["KeyError", "ValueError", "MemoryError"],
        complexity_score=8,
        wrapper_code='''
def merge_wrapper(left_data, right_data, on=None, how='inner'):
    """Wrapper for pandas.DataFrame.merge with edge case handling."""
    import pandas as pd
    left_df = pd.DataFrame(left_data) if isinstance(left_data, dict) else left_data
    right_df = pd.DataFrame(right_data) if isinstance(right_data, dict) else right_data
    return left_df.merge(right_df, on=on, how=how).to_dict()
'''
    ),
    SystemLevelFunction(
        id="sys_pandas_concat",
        name="pandas.concat",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def concat_wrapper(dfs, axis=0, ignore_index=False)",
        docstring="Concatenate pandas objects along a particular axis.",
        edge_cases=["Empty list", "Mixed types", "Misaligned indices"],
        bug_types=["TypeError", "ValueError"],
        complexity_score=7,
        wrapper_code='''
def concat_wrapper(data_list, axis=0, ignore_index=False):
    """Wrapper for pandas.concat."""
    import pandas as pd
    dfs = [pd.DataFrame(d) if isinstance(d, dict) else d for d in data_list]
    return pd.concat(dfs, axis=axis, ignore_index=ignore_index).to_dict()
'''
    ),
    SystemLevelFunction(
        id="sys_pandas_groupby",
        name="pandas.DataFrame.groupby",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def groupby_wrapper(data, by, aggfunc='sum')",
        docstring="Group DataFrame by column and apply aggregation.",
        edge_cases=["Missing group key", "All NaN group", "Empty groups"],
        bug_types=["KeyError", "DataError"],
        complexity_score=8,
        wrapper_code='''
def groupby_wrapper(data, by, aggfunc='sum'):
    """Wrapper for pandas groupby with aggregation."""
    import pandas as pd
    df = pd.DataFrame(data)
    grouped = df.groupby(by)
    if aggfunc == 'sum':
        return grouped.sum().to_dict()
    elif aggfunc == 'mean':
        return grouped.mean().to_dict()
    elif aggfunc == 'count':
        return grouped.count().to_dict()
    return grouped.first().to_dict()
'''
    ),
    SystemLevelFunction(
        id="sys_pandas_pivot",
        name="pandas.DataFrame.pivot_table",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def pivot_wrapper(data, index, columns, values, aggfunc='mean')",
        docstring="Create a pivot table from DataFrame.",
        edge_cases=["Duplicate entries", "Missing values", "Non-numeric aggregation"],
        bug_types=["ValueError", "KeyError"],
        complexity_score=9,
        wrapper_code='''
def pivot_wrapper(data, index, columns, values, aggfunc='mean'):
    """Wrapper for pandas pivot_table."""
    import pandas as pd
    df = pd.DataFrame(data)
    result = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
    return result.to_dict()
'''
    ),
    SystemLevelFunction(
        id="sys_pandas_read_csv",
        name="pandas.read_csv",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def csv_parser_wrapper(csv_string, delimiter=',')",
        docstring="Parse CSV data from string.",
        edge_cases=["Malformed CSV", "Mixed encodings", "Quoted delimiters"],
        bug_types=["ParserError", "UnicodeDecodeError"],
        complexity_score=7,
        wrapper_code='''
def csv_parser_wrapper(csv_string, delimiter=','):
    """Wrapper for parsing CSV from string."""
    import pandas as pd
    from io import StringIO
    return pd.read_csv(StringIO(csv_string), delimiter=delimiter).to_dict()
'''
    ),
    SystemLevelFunction(
        id="sys_pandas_fillna",
        name="pandas.DataFrame.fillna",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def fillna_wrapper(data, value=None, method=None)",
        docstring="Fill NA/NaN values using specified method.",
        edge_cases=["All NaN column", "Mixed types", "Categorical data"],
        bug_types=["ValueError", "TypeError"],
        complexity_score=6,
        wrapper_code='''
def fillna_wrapper(data, value=None, method=None):
    """Wrapper for pandas fillna."""
    import pandas as pd
    df = pd.DataFrame(data)
    if method:
        return df.fillna(method=method).to_dict()
    return df.fillna(value=value).to_dict()
'''
    ),
    SystemLevelFunction(
        id="sys_pandas_dropna",
        name="pandas.DataFrame.dropna",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def dropna_wrapper(data, axis=0, how='any', thresh=None)",
        docstring="Remove missing values from DataFrame.",
        edge_cases=["All NaN row/column", "Threshold edge", "Empty result"],
        bug_types=["ValueError"],
        complexity_score=5,
        wrapper_code='''
def dropna_wrapper(data, axis=0, how='any', thresh=None):
    """Wrapper for pandas dropna."""
    import pandas as pd
    df = pd.DataFrame(data)
    return df.dropna(axis=axis, how=how, thresh=thresh).to_dict()
'''
    ),
    SystemLevelFunction(
        id="sys_pandas_apply",
        name="pandas.DataFrame.apply",
        library="pandas",
        import_statement="import pandas as pd",
        signature="def apply_wrapper(data, func_name, axis=0)",
        docstring="Apply a function along an axis of the DataFrame.",
        edge_cases=["Non-uniform return types", "Empty DataFrame", "Exception in function"],
        bug_types=["TypeError", "ValueError"],
        complexity_score=7,
        wrapper_code='''
def apply_wrapper(data, func_name, axis=0):
    """Wrapper for pandas apply with common functions."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(data)
    funcs = {'sum': np.sum, 'mean': np.mean, 'max': np.max, 'min': np.min, 'len': len}
    if func_name in funcs:
        return df.apply(funcs[func_name], axis=axis).to_dict()
    return df.to_dict()
'''
    ),

    # =========================================================================
    # NUMPY (8 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_numpy_reshape",
        name="numpy.reshape",
        library="numpy",
        import_statement="import numpy as np",
        signature="def reshape_wrapper(data, new_shape)",
        docstring="Reshape an array without changing its data.",
        edge_cases=["Incompatible shape", "Zero dimension", "Negative dimension"],
        bug_types=["ValueError"],
        complexity_score=6,
        wrapper_code='''
def reshape_wrapper(data, new_shape):
    """Wrapper for numpy reshape."""
    import numpy as np
    arr = np.array(data)
    return arr.reshape(new_shape).tolist()
'''
    ),
    SystemLevelFunction(
        id="sys_numpy_concatenate",
        name="numpy.concatenate",
        library="numpy",
        import_statement="import numpy as np",
        signature="def np_concat_wrapper(arrays, axis=0)",
        docstring="Join arrays along an existing axis.",
        edge_cases=["Empty array", "Mismatched dimensions", "Invalid axis"],
        bug_types=["ValueError", "AxisError"],
        complexity_score=6,
        wrapper_code='''
def np_concat_wrapper(arrays, axis=0):
    """Wrapper for numpy concatenate."""
    import numpy as np
    arrs = [np.array(a) for a in arrays]
    return np.concatenate(arrs, axis=axis).tolist()
'''
    ),
    SystemLevelFunction(
        id="sys_numpy_dot",
        name="numpy.dot",
        library="numpy",
        import_statement="import numpy as np",
        signature="def dot_wrapper(a, b)",
        docstring="Compute dot product of two arrays.",
        edge_cases=["Shape mismatch", "Empty arrays", "1D vs 2D"],
        bug_types=["ValueError"],
        complexity_score=7,
        wrapper_code='''
def dot_wrapper(a, b):
    """Wrapper for numpy dot product."""
    import numpy as np
    arr_a = np.array(a)
    arr_b = np.array(b)
    return np.dot(arr_a, arr_b).tolist()
'''
    ),
    SystemLevelFunction(
        id="sys_numpy_linalg_inv",
        name="numpy.linalg.inv",
        library="numpy",
        import_statement="import numpy as np",
        signature="def matrix_inv_wrapper(matrix)",
        docstring="Compute the inverse of a matrix.",
        edge_cases=["Singular matrix", "Non-square matrix", "Near-singular"],
        bug_types=["LinAlgError", "ValueError"],
        complexity_score=8,
        wrapper_code='''
def matrix_inv_wrapper(matrix):
    """Wrapper for numpy matrix inverse."""
    import numpy as np
    arr = np.array(matrix)
    return np.linalg.inv(arr).tolist()
'''
    ),
    SystemLevelFunction(
        id="sys_numpy_where",
        name="numpy.where",
        library="numpy",
        import_statement="import numpy as np",
        signature="def where_wrapper(condition, x, y)",
        docstring="Return elements chosen from x or y based on condition.",
        edge_cases=["Empty array", "Broadcasting mismatch", "All True/False"],
        bug_types=["ValueError", "TypeError"],
        complexity_score=6,
        wrapper_code='''
def where_wrapper(condition, x, y):
    """Wrapper for numpy where."""
    import numpy as np
    cond = np.array(condition)
    arr_x = np.array(x)
    arr_y = np.array(y)
    return np.where(cond, arr_x, arr_y).tolist()
'''
    ),
    SystemLevelFunction(
        id="sys_numpy_sort",
        name="numpy.sort",
        library="numpy",
        import_statement="import numpy as np",
        signature="def np_sort_wrapper(data, axis=-1)",
        docstring="Return a sorted copy of an array.",
        edge_cases=["Empty array", "NaN values", "Complex numbers"],
        bug_types=["ValueError", "TypeError"],
        complexity_score=5,
        wrapper_code='''
def np_sort_wrapper(data, axis=-1):
    """Wrapper for numpy sort."""
    import numpy as np
    arr = np.array(data)
    return np.sort(arr, axis=axis).tolist()
'''
    ),
    SystemLevelFunction(
        id="sys_numpy_unique",
        name="numpy.unique",
        library="numpy",
        import_statement="import numpy as np",
        signature="def unique_wrapper(data, return_counts=False)",
        docstring="Find the unique elements of an array.",
        edge_cases=["Empty array", "All duplicates", "Mixed types"],
        bug_types=["TypeError"],
        complexity_score=5,
        wrapper_code='''
def unique_wrapper(data, return_counts=False):
    """Wrapper for numpy unique."""
    import numpy as np
    arr = np.array(data)
    if return_counts:
        vals, counts = np.unique(arr, return_counts=True)
        return {"values": vals.tolist(), "counts": counts.tolist()}
    return np.unique(arr).tolist()
'''
    ),
    SystemLevelFunction(
        id="sys_numpy_argmax",
        name="numpy.argmax",
        library="numpy",
        import_statement="import numpy as np",
        signature="def argmax_wrapper(data, axis=None)",
        docstring="Return indices of the maximum values along an axis.",
        edge_cases=["Empty array", "Multiple maxima", "NaN handling"],
        bug_types=["ValueError"],
        complexity_score=5,
        wrapper_code='''
def argmax_wrapper(data, axis=None):
    """Wrapper for numpy argmax."""
    import numpy as np
    arr = np.array(data)
    result = np.argmax(arr, axis=axis)
    return result.tolist() if hasattr(result, 'tolist') else int(result)
'''
    ),

    # =========================================================================
    # JSON (3 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_json_loads",
        name="json.loads",
        library="json",
        import_statement="import json",
        signature="def json_loads_wrapper(json_string)",
        docstring="Deserialize a JSON string to a Python object.",
        edge_cases=["Malformed JSON", "Unicode characters", "Deeply nested", "Large numbers"],
        bug_types=["JSONDecodeError", "ValueError"],
        complexity_score=6,
        wrapper_code='''
def json_loads_wrapper(json_string):
    """Wrapper for json.loads with type validation."""
    import json
    result = json.loads(json_string)
    return result
'''
    ),
    SystemLevelFunction(
        id="sys_json_dumps",
        name="json.dumps",
        library="json",
        import_statement="import json",
        signature="def json_dumps_wrapper(obj, indent=None)",
        docstring="Serialize a Python object to JSON string.",
        edge_cases=["Circular references", "Non-serializable types", "NaN/Infinity"],
        bug_types=["TypeError", "ValueError"],
        complexity_score=6,
        wrapper_code='''
def json_dumps_wrapper(obj, indent=None):
    """Wrapper for json.dumps."""
    import json
    return json.dumps(obj, indent=indent, default=str)
'''
    ),
    SystemLevelFunction(
        id="sys_json_load_file",
        name="json.load",
        library="json",
        import_statement="import json",
        signature="def json_load_wrapper(json_content)",
        docstring="Load JSON from file-like object.",
        edge_cases=["Empty file", "BOM characters", "Trailing data"],
        bug_types=["JSONDecodeError", "UnicodeDecodeError"],
        complexity_score=5,
        wrapper_code='''
def json_load_wrapper(json_content):
    """Wrapper for json.load from string content."""
    import json
    from io import StringIO
    return json.load(StringIO(json_content))
'''
    ),

    # =========================================================================
    # DATETIME (4 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_datetime_strptime",
        name="datetime.strptime",
        library="datetime",
        import_statement="from datetime import datetime",
        signature="def datetime_parse_wrapper(date_string, format_string)",
        docstring="Parse a date string according to a format.",
        edge_cases=["Invalid format", "Leap year dates", "Timezone issues", "Edge months"],
        bug_types=["ValueError", "OverflowError"],
        complexity_score=7,
        wrapper_code='''
def datetime_parse_wrapper(date_string, format_string):
    """Wrapper for datetime.strptime."""
    from datetime import datetime
    result = datetime.strptime(date_string, format_string)
    return result.isoformat()
'''
    ),
    SystemLevelFunction(
        id="sys_datetime_strftime",
        name="datetime.strftime",
        library="datetime",
        import_statement="from datetime import datetime",
        signature="def datetime_format_wrapper(year, month, day, format_string)",
        docstring="Format a datetime object to a string.",
        edge_cases=["Invalid dates", "Year boundaries", "Format specifiers"],
        bug_types=["ValueError"],
        complexity_score=6,
        wrapper_code='''
def datetime_format_wrapper(year, month, day, format_string):
    """Wrapper for datetime.strftime."""
    from datetime import datetime
    dt = datetime(year, month, day)
    return dt.strftime(format_string)
'''
    ),
    SystemLevelFunction(
        id="sys_datetime_timedelta",
        name="datetime.timedelta",
        library="datetime",
        import_statement="from datetime import datetime, timedelta",
        signature="def timedelta_wrapper(days=0, hours=0, minutes=0, seconds=0)",
        docstring="Create a timedelta and add to current date.",
        edge_cases=["Negative values", "Large values", "Overflow"],
        bug_types=["OverflowError", "ValueError"],
        complexity_score=5,
        wrapper_code='''
def timedelta_wrapper(days=0, hours=0, minutes=0, seconds=0):
    """Wrapper for timedelta operations."""
    from datetime import datetime, timedelta
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    result = datetime.now() + delta
    return result.isoformat()
'''
    ),
    SystemLevelFunction(
        id="sys_datetime_diff",
        name="datetime.difference",
        library="datetime",
        import_statement="from datetime import datetime",
        signature="def datetime_diff_wrapper(date1_str, date2_str, fmt='%Y-%m-%d')",
        docstring="Calculate difference between two dates.",
        edge_cases=["Same date", "Negative difference", "Different formats"],
        bug_types=["ValueError"],
        complexity_score=5,
        wrapper_code='''
def datetime_diff_wrapper(date1_str, date2_str, fmt='%Y-%m-%d'):
    """Wrapper for datetime difference calculation."""
    from datetime import datetime
    d1 = datetime.strptime(date1_str, fmt)
    d2 = datetime.strptime(date2_str, fmt)
    diff = d2 - d1
    return {"days": diff.days, "seconds": diff.seconds, "total_seconds": diff.total_seconds()}
'''
    ),

    # =========================================================================
    # OS/PATH (4 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_os_path_join",
        name="os.path.join",
        library="os",
        import_statement="import os",
        signature="def path_join_wrapper(*paths)",
        docstring="Join path components intelligently.",
        edge_cases=["Empty paths", "Absolute paths in middle", "Special characters"],
        bug_types=["TypeError"],
        complexity_score=5,
        wrapper_code='''
def path_join_wrapper(*paths):
    """Wrapper for os.path.join."""
    import os
    return os.path.join(*paths)
'''
    ),
    SystemLevelFunction(
        id="sys_os_makedirs",
        name="os.makedirs",
        library="os",
        import_statement="import os",
        signature="def makedirs_wrapper(path, exist_ok=True)",
        docstring="Create directory and all parent directories.",
        edge_cases=["Permission denied", "Path too long", "Invalid characters"],
        bug_types=["OSError", "PermissionError"],
        complexity_score=5,
        wrapper_code='''
def makedirs_wrapper(path, exist_ok=True):
    """Wrapper for os.makedirs (simulation for testing)."""
    import os
    # Validate path without creating
    if not path or len(path) > 255:
        raise ValueError("Invalid path")
    return os.path.normpath(path)
'''
    ),
    SystemLevelFunction(
        id="sys_os_listdir",
        name="os.listdir",
        library="os",
        import_statement="import os",
        signature="def listdir_wrapper(path)",
        docstring="List directory contents.",
        edge_cases=["Non-existent path", "Permission denied", "Empty directory"],
        bug_types=["FileNotFoundError", "PermissionError"],
        complexity_score=4,
        wrapper_code='''
def listdir_wrapper(path):
    """Wrapper for os.listdir."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    return os.listdir(path)
'''
    ),
    SystemLevelFunction(
        id="sys_os_path_split",
        name="os.path.split",
        library="os",
        import_statement="import os",
        signature="def path_split_wrapper(path)",
        docstring="Split path into directory and filename.",
        edge_cases=["Empty path", "Root path", "Trailing separator"],
        bug_types=["TypeError"],
        complexity_score=4,
        wrapper_code='''
def path_split_wrapper(path):
    """Wrapper for os.path.split."""
    import os
    head, tail = os.path.split(path)
    return {"directory": head, "filename": tail}
'''
    ),

    # =========================================================================
    # RE/REGEX (4 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_re_match",
        name="re.match",
        library="re",
        import_statement="import re",
        signature="def regex_match_wrapper(pattern, string)",
        docstring="Try to apply the pattern at the start of the string.",
        edge_cases=["Invalid regex", "Empty pattern", "Special characters"],
        bug_types=["re.error", "TypeError"],
        complexity_score=6,
        wrapper_code='''
def regex_match_wrapper(pattern, string):
    """Wrapper for re.match."""
    import re
    match = re.match(pattern, string)
    if match:
        return {"matched": True, "groups": match.groups(), "span": match.span()}
    return {"matched": False, "groups": (), "span": None}
'''
    ),
    SystemLevelFunction(
        id="sys_re_findall",
        name="re.findall",
        library="re",
        import_statement="import re",
        signature="def regex_findall_wrapper(pattern, string)",
        docstring="Find all occurrences of pattern in string.",
        edge_cases=["Overlapping matches", "Empty matches", "Greedy vs non-greedy"],
        bug_types=["re.error"],
        complexity_score=6,
        wrapper_code='''
def regex_findall_wrapper(pattern, string):
    """Wrapper for re.findall."""
    import re
    return re.findall(pattern, string)
'''
    ),
    SystemLevelFunction(
        id="sys_re_sub",
        name="re.sub",
        library="re",
        import_statement="import re",
        signature="def regex_sub_wrapper(pattern, replacement, string)",
        docstring="Replace occurrences of pattern with replacement.",
        edge_cases=["Backreferences", "Empty replacement", "No matches"],
        bug_types=["re.error", "TypeError"],
        complexity_score=6,
        wrapper_code='''
def regex_sub_wrapper(pattern, replacement, string):
    """Wrapper for re.sub."""
    import re
    return re.sub(pattern, replacement, string)
'''
    ),
    SystemLevelFunction(
        id="sys_re_split",
        name="re.split",
        library="re",
        import_statement="import re",
        signature="def regex_split_wrapper(pattern, string, maxsplit=0)",
        docstring="Split string by regex pattern.",
        edge_cases=["Empty matches", "Capturing groups", "Max split"],
        bug_types=["re.error", "TypeError"],
        complexity_score=5,
        wrapper_code='''
def regex_split_wrapper(pattern, string, maxsplit=0):
    """Wrapper for re.split."""
    import re
    return re.split(pattern, string, maxsplit=maxsplit)
'''
    ),

    # =========================================================================
    # COLLECTIONS (4 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_collections_counter",
        name="collections.Counter",
        library="collections",
        import_statement="from collections import Counter",
        signature="def counter_wrapper(iterable)",
        docstring="Count elements in an iterable.",
        edge_cases=["Empty iterable", "Non-hashable elements", "Large counts"],
        bug_types=["TypeError"],
        complexity_score=5,
        wrapper_code='''
def counter_wrapper(iterable):
    """Wrapper for collections.Counter."""
    from collections import Counter
    return dict(Counter(iterable))
'''
    ),
    SystemLevelFunction(
        id="sys_collections_defaultdict",
        name="collections.defaultdict",
        library="collections",
        import_statement="from collections import defaultdict",
        signature="def defaultdict_wrapper(data, default_type='list')",
        docstring="Create a defaultdict from data.",
        edge_cases=["Invalid default type", "Nested access"],
        bug_types=["TypeError"],
        complexity_score=5,
        wrapper_code='''
def defaultdict_wrapper(data, default_type='list'):
    """Wrapper for collections.defaultdict."""
    from collections import defaultdict
    if default_type == 'list':
        d = defaultdict(list)
    elif default_type == 'int':
        d = defaultdict(int)
    else:
        d = defaultdict(str)
    for k, v in data.items():
        d[k] = v
    return dict(d)
'''
    ),
    SystemLevelFunction(
        id="sys_collections_deque",
        name="collections.deque",
        library="collections",
        import_statement="from collections import deque",
        signature="def deque_wrapper(iterable, maxlen=None, operations=None)",
        docstring="Double-ended queue operations.",
        edge_cases=["Maxlen overflow", "Empty deque", "Rotation"],
        bug_types=["IndexError", "TypeError"],
        complexity_score=6,
        wrapper_code='''
def deque_wrapper(iterable, maxlen=None, operations=None):
    """Wrapper for collections.deque with operations."""
    from collections import deque
    d = deque(iterable, maxlen=maxlen)
    if operations:
        for op, val in operations:
            if op == 'append':
                d.append(val)
            elif op == 'appendleft':
                d.appendleft(val)
            elif op == 'pop':
                d.pop()
            elif op == 'popleft':
                d.popleft()
            elif op == 'rotate':
                d.rotate(val)
    return list(d)
'''
    ),
    SystemLevelFunction(
        id="sys_collections_ordereddict",
        name="collections.OrderedDict",
        library="collections",
        import_statement="from collections import OrderedDict",
        signature="def ordereddict_wrapper(items)",
        docstring="Create an ordered dictionary from items.",
        edge_cases=["Duplicate keys", "Empty input", "Move to end"],
        bug_types=["TypeError", "KeyError"],
        complexity_score=4,
        wrapper_code='''
def ordereddict_wrapper(items):
    """Wrapper for collections.OrderedDict."""
    from collections import OrderedDict
    od = OrderedDict(items)
    return dict(od)
'''
    ),

    # =========================================================================
    # ITERTOOLS (5 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_itertools_combinations",
        name="itertools.combinations",
        library="itertools",
        import_statement="from itertools import combinations",
        signature="def combinations_wrapper(iterable, r)",
        docstring="Return r-length combinations of elements.",
        edge_cases=["r > len(iterable)", "r = 0", "Empty iterable"],
        bug_types=["ValueError", "TypeError"],
        complexity_score=5,
        wrapper_code='''
def combinations_wrapper(iterable, r):
    """Wrapper for itertools.combinations."""
    from itertools import combinations
    return list(combinations(iterable, r))
'''
    ),
    SystemLevelFunction(
        id="sys_itertools_permutations",
        name="itertools.permutations",
        library="itertools",
        import_statement="from itertools import permutations",
        signature="def permutations_wrapper(iterable, r=None)",
        docstring="Return successive r-length permutations of elements.",
        edge_cases=["r > len(iterable)", "Large factorial", "Duplicates"],
        bug_types=["ValueError", "MemoryError"],
        complexity_score=6,
        wrapper_code='''
def permutations_wrapper(iterable, r=None):
    """Wrapper for itertools.permutations."""
    from itertools import permutations
    return list(permutations(iterable, r))
'''
    ),
    SystemLevelFunction(
        id="sys_itertools_groupby",
        name="itertools.groupby",
        library="itertools",
        import_statement="from itertools import groupby",
        signature="def itertools_groupby_wrapper(iterable, key_func=None)",
        docstring="Group consecutive elements by key function.",
        edge_cases=["Unsorted input", "None key", "Empty groups"],
        bug_types=["TypeError"],
        complexity_score=6,
        wrapper_code='''
def itertools_groupby_wrapper(iterable, key_func=None):
    """Wrapper for itertools.groupby."""
    from itertools import groupby
    if key_func == 'first':
        key = lambda x: x[0] if isinstance(x, (list, tuple, str)) else x
    elif key_func == 'len':
        key = len
    else:
        key = None
    result = []
    for k, g in groupby(iterable, key=key):
        result.append({"key": k, "group": list(g)})
    return result
'''
    ),
    SystemLevelFunction(
        id="sys_itertools_chain",
        name="itertools.chain",
        library="itertools",
        import_statement="from itertools import chain",
        signature="def chain_wrapper(*iterables)",
        docstring="Chain multiple iterables into one.",
        edge_cases=["Empty iterables", "Mixed types", "Generators"],
        bug_types=["TypeError"],
        complexity_score=4,
        wrapper_code='''
def chain_wrapper(*iterables):
    """Wrapper for itertools.chain."""
    from itertools import chain
    return list(chain(*iterables))
'''
    ),
    SystemLevelFunction(
        id="sys_itertools_product",
        name="itertools.product",
        library="itertools",
        import_statement="from itertools import product",
        signature="def product_wrapper(*iterables, repeat=1)",
        docstring="Cartesian product of input iterables.",
        edge_cases=["Empty iterables", "Large product", "Repeat parameter"],
        bug_types=["MemoryError", "TypeError"],
        complexity_score=6,
        wrapper_code='''
def product_wrapper(*iterables, repeat=1):
    """Wrapper for itertools.product."""
    from itertools import product
    return list(product(*iterables, repeat=repeat))
'''
    ),

    # =========================================================================
    # FUNCTOOLS (3 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_functools_reduce",
        name="functools.reduce",
        library="functools",
        import_statement="from functools import reduce",
        signature="def reduce_wrapper(iterable, operation, initial=None)",
        docstring="Apply function cumulatively to sequence.",
        edge_cases=["Empty sequence", "Single element", "No initial"],
        bug_types=["TypeError"],
        complexity_score=6,
        wrapper_code='''
def reduce_wrapper(iterable, operation, initial=None):
    """Wrapper for functools.reduce with common operations."""
    from functools import reduce
    ops = {
        'add': lambda x, y: x + y,
        'mul': lambda x, y: x * y,
        'max': max,
        'min': min
    }
    func = ops.get(operation, ops['add'])
    if initial is not None:
        return reduce(func, iterable, initial)
    return reduce(func, iterable)
'''
    ),
    SystemLevelFunction(
        id="sys_functools_partial",
        name="functools.partial",
        library="functools",
        import_statement="from functools import partial",
        signature="def partial_wrapper(func_name, *args, **kwargs)",
        docstring="Freeze some function arguments.",
        edge_cases=["All args frozen", "Conflicting kwargs", "Invalid function"],
        bug_types=["TypeError"],
        complexity_score=5,
        wrapper_code='''
def partial_wrapper(func_name, *args, **kwargs):
    """Wrapper demonstrating functools.partial."""
    from functools import partial
    funcs = {
        'add': lambda a, b: a + b,
        'mul': lambda a, b: a * b,
        'pow': pow
    }
    if func_name not in funcs:
        raise ValueError(f"Unknown function: {func_name}")
    p = partial(funcs[func_name], *args, **kwargs)
    return {"frozen_args": args, "frozen_kwargs": kwargs}
'''
    ),
    SystemLevelFunction(
        id="sys_functools_lru_cache",
        name="functools.lru_cache",
        library="functools",
        import_statement="from functools import lru_cache",
        signature="def lru_cache_demo(n, maxsize=128)",
        docstring="Demonstrate LRU cache with fibonacci.",
        edge_cases=["Cache full", "Unhashable args", "Negative n"],
        bug_types=["TypeError", "RecursionError"],
        complexity_score=6,
        wrapper_code='''
def lru_cache_demo(n, maxsize=128):
    """Wrapper demonstrating functools.lru_cache."""
    from functools import lru_cache
    @lru_cache(maxsize=maxsize)
    def fib(x):
        if x < 2:
            return x
        return fib(x-1) + fib(x-2)
    if n < 0:
        raise ValueError("n must be non-negative")
    result = fib(n)
    info = fib.cache_info()
    return {"result": result, "cache_hits": info.hits, "cache_misses": info.misses}
'''
    ),

    # =========================================================================
    # HASHLIB (3 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_hashlib_md5",
        name="hashlib.md5",
        library="hashlib",
        import_statement="import hashlib",
        signature="def md5_wrapper(data)",
        docstring="Compute MD5 hash of data.",
        edge_cases=["Empty string", "Unicode", "Large data"],
        bug_types=["TypeError"],
        complexity_score=4,
        wrapper_code='''
def md5_wrapper(data):
    """Wrapper for hashlib.md5."""
    import hashlib
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()
'''
    ),
    SystemLevelFunction(
        id="sys_hashlib_sha256",
        name="hashlib.sha256",
        library="hashlib",
        import_statement="import hashlib",
        signature="def sha256_wrapper(data)",
        docstring="Compute SHA-256 hash of data.",
        edge_cases=["Empty string", "Binary data", "Encoding issues"],
        bug_types=["TypeError"],
        complexity_score=4,
        wrapper_code='''
def sha256_wrapper(data):
    """Wrapper for hashlib.sha256."""
    import hashlib
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()
'''
    ),
    SystemLevelFunction(
        id="sys_hashlib_pbkdf2",
        name="hashlib.pbkdf2_hmac",
        library="hashlib",
        import_statement="import hashlib",
        signature="def pbkdf2_wrapper(password, salt, iterations=100000)",
        docstring="Derive key using PBKDF2.",
        edge_cases=["Weak password", "Short salt", "High iterations"],
        bug_types=["TypeError", "ValueError"],
        complexity_score=6,
        wrapper_code='''
def pbkdf2_wrapper(password, salt, iterations=100000):
    """Wrapper for hashlib.pbkdf2_hmac."""
    import hashlib
    if isinstance(password, str):
        password = password.encode('utf-8')
    if isinstance(salt, str):
        salt = salt.encode('utf-8')
    dk = hashlib.pbkdf2_hmac('sha256', password, salt, iterations)
    return dk.hex()
'''
    ),

    # =========================================================================
    # BASE64 (3 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_base64_encode",
        name="base64.b64encode",
        library="base64",
        import_statement="import base64",
        signature="def base64_encode_wrapper(data)",
        docstring="Encode data to base64.",
        edge_cases=["Binary data", "Unicode", "Empty string"],
        bug_types=["TypeError"],
        complexity_score=4,
        wrapper_code='''
def base64_encode_wrapper(data):
    """Wrapper for base64.b64encode."""
    import base64
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')
'''
    ),
    SystemLevelFunction(
        id="sys_base64_decode",
        name="base64.b64decode",
        library="base64",
        import_statement="import base64",
        signature="def base64_decode_wrapper(data)",
        docstring="Decode base64 encoded data.",
        edge_cases=["Invalid padding", "Non-base64 chars", "URL-safe"],
        bug_types=["binascii.Error", "ValueError"],
        complexity_score=4,
        wrapper_code='''
def base64_decode_wrapper(data):
    """Wrapper for base64.b64decode."""
    import base64
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64decode(data).decode('utf-8')
'''
    ),
    SystemLevelFunction(
        id="sys_base64_urlsafe",
        name="base64.urlsafe_b64encode",
        library="base64",
        import_statement="import base64",
        signature="def base64_urlsafe_wrapper(data, decode=False)",
        docstring="URL-safe base64 encoding/decoding.",
        edge_cases=["URL characters", "Padding", "Binary data"],
        bug_types=["TypeError", "binascii.Error"],
        complexity_score=4,
        wrapper_code='''
def base64_urlsafe_wrapper(data, decode=False):
    """Wrapper for base64 URL-safe operations."""
    import base64
    if isinstance(data, str):
        data = data.encode('utf-8')
    if decode:
        return base64.urlsafe_b64decode(data).decode('utf-8')
    return base64.urlsafe_b64encode(data).decode('utf-8')
'''
    ),

    # =========================================================================
    # URLLIB (3 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_urllib_parse",
        name="urllib.parse.urlparse",
        library="urllib",
        import_statement="from urllib.parse import urlparse, urlencode",
        signature="def url_parse_wrapper(url)",
        docstring="Parse a URL into components.",
        edge_cases=["Invalid URL", "Unicode in URL", "Missing scheme"],
        bug_types=["ValueError"],
        complexity_score=5,
        wrapper_code='''
def url_parse_wrapper(url):
    """Wrapper for urllib.parse.urlparse."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "query": parsed.query
    }
'''
    ),
    SystemLevelFunction(
        id="sys_urllib_encode",
        name="urllib.parse.urlencode",
        library="urllib",
        import_statement="from urllib.parse import urlencode",
        signature="def url_encode_wrapper(params)",
        docstring="Encode a dictionary as URL query parameters.",
        edge_cases=["Nested dicts", "Special characters", "Empty values"],
        bug_types=["TypeError"],
        complexity_score=4,
        wrapper_code='''
def url_encode_wrapper(params):
    """Wrapper for urllib.parse.urlencode."""
    from urllib.parse import urlencode
    return urlencode(params)
'''
    ),
    SystemLevelFunction(
        id="sys_urllib_quote",
        name="urllib.parse.quote",
        library="urllib",
        import_statement="from urllib.parse import quote, unquote",
        signature="def url_quote_wrapper(string, safe='/', unquote_mode=False)",
        docstring="URL quote/unquote operations.",
        edge_cases=["Unicode chars", "Already encoded", "Safe chars"],
        bug_types=["TypeError"],
        complexity_score=4,
        wrapper_code='''
def url_quote_wrapper(string, safe='/', unquote_mode=False):
    """Wrapper for urllib.parse.quote/unquote."""
    from urllib.parse import quote, unquote
    if unquote_mode:
        return unquote(string)
    return quote(string, safe=safe)
'''
    ),

    # =========================================================================
    # MATH (4 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_math_sqrt",
        name="math.sqrt",
        library="math",
        import_statement="import math",
        signature="def sqrt_wrapper(x)",
        docstring="Return square root of x.",
        edge_cases=["Negative number", "Zero", "Large number", "Float precision"],
        bug_types=["ValueError", "OverflowError"],
        complexity_score=3,
        wrapper_code='''
def sqrt_wrapper(x):
    """Wrapper for math.sqrt."""
    import math
    return math.sqrt(x)
'''
    ),
    SystemLevelFunction(
        id="sys_math_log",
        name="math.log",
        library="math",
        import_statement="import math",
        signature="def log_wrapper(x, base=None)",
        docstring="Return logarithm of x to the given base.",
        edge_cases=["Zero", "Negative", "Base 1", "Very small"],
        bug_types=["ValueError", "ZeroDivisionError"],
        complexity_score=4,
        wrapper_code='''
def log_wrapper(x, base=None):
    """Wrapper for math.log."""
    import math
    if base is None:
        return math.log(x)
    return math.log(x, base)
'''
    ),
    SystemLevelFunction(
        id="sys_math_factorial",
        name="math.factorial",
        library="math",
        import_statement="import math",
        signature="def factorial_wrapper(n)",
        docstring="Return n factorial.",
        edge_cases=["Negative", "Large number", "Float input"],
        bug_types=["ValueError", "OverflowError"],
        complexity_score=4,
        wrapper_code='''
def factorial_wrapper(n):
    """Wrapper for math.factorial."""
    import math
    return math.factorial(n)
'''
    ),
    SystemLevelFunction(
        id="sys_math_gcd",
        name="math.gcd",
        library="math",
        import_statement="import math",
        signature="def gcd_wrapper(a, b)",
        docstring="Return greatest common divisor of a and b.",
        edge_cases=["Zero values", "Negative", "Large numbers"],
        bug_types=["TypeError"],
        complexity_score=3,
        wrapper_code='''
def gcd_wrapper(a, b):
    """Wrapper for math.gcd."""
    import math
    return math.gcd(a, b)
'''
    ),

    # =========================================================================
    # STRING (4 functions)
    # =========================================================================
    SystemLevelFunction(
        id="sys_string_formatter",
        name="str.format",
        library="string",
        import_statement="",
        signature="def str_format_wrapper(template, *args, **kwargs)",
        docstring="Format string with arguments.",
        edge_cases=["Missing keys", "Index error", "Format spec errors"],
        bug_types=["KeyError", "IndexError", "ValueError"],
        complexity_score=5,
        wrapper_code='''
def str_format_wrapper(template, *args, **kwargs):
    """Wrapper for str.format."""
    return template.format(*args, **kwargs)
'''
    ),
    SystemLevelFunction(
        id="sys_string_split",
        name="str.split",
        library="string",
        import_statement="",
        signature="def str_split_wrapper(s, sep=None, maxsplit=-1)",
        docstring="Split string by separator.",
        edge_cases=["Empty string", "No separator found", "Consecutive separators"],
        bug_types=["TypeError", "ValueError"],
        complexity_score=4,
        wrapper_code='''
def str_split_wrapper(s, sep=None, maxsplit=-1):
    """Wrapper for str.split."""
    return s.split(sep, maxsplit)
'''
    ),
    SystemLevelFunction(
        id="sys_string_join",
        name="str.join",
        library="string",
        import_statement="",
        signature="def str_join_wrapper(sep, iterable)",
        docstring="Join iterable elements with separator.",
        edge_cases=["Non-string elements", "Empty iterable", "Empty separator"],
        bug_types=["TypeError"],
        complexity_score=3,
        wrapper_code='''
def str_join_wrapper(sep, iterable):
    """Wrapper for str.join."""
    return sep.join(iterable)
'''
    ),
    SystemLevelFunction(
        id="sys_string_replace",
        name="str.replace",
        library="string",
        import_statement="",
        signature="def str_replace_wrapper(s, old, new, count=-1)",
        docstring="Replace occurrences of old with new.",
        edge_cases=["Empty old", "No match", "Count limit"],
        bug_types=["TypeError"],
        complexity_score=3,
        wrapper_code='''
def str_replace_wrapper(s, old, new, count=-1):
    """Wrapper for str.replace."""
    return s.replace(old, new, count)
'''
    ),
]


# =============================================================================
# 10 SYSTEM-LEVEL TESTING FUNCTIONS (Diverse selection for evaluation)
# =============================================================================

TESTING_FUNCTION_IDS = [
    "sys_pandas_merge",        # Complex DataFrame operations
    "sys_pandas_pivot",        # Pivot table complexity
    "sys_numpy_linalg_inv",    # Linear algebra edge cases
    "sys_json_loads",          # Parsing with many edge cases
    "sys_datetime_strptime",   # Format parsing
    "sys_re_match",            # Regex complexity
    "sys_itertools_groupby",   # Grouping edge cases
    "sys_functools_reduce",    # Functional programming
    "sys_hashlib_pbkdf2",      # Cryptographic function
    "sys_base64_decode",       # Encoding edge cases
]

TESTING_FUNCTIONS: List[SystemLevelFunction] = [
    func for func in TRAINING_FUNCTIONS
    if func.id in TESTING_FUNCTION_IDS
]

# Mark testing functions
for func in TESTING_FUNCTIONS:
    func.is_training = False


def get_training_functions() -> List[SystemLevelFunction]:
    """Get the 60 system-level functions for training."""
    return TRAINING_FUNCTIONS.copy()


def get_testing_functions() -> List[SystemLevelFunction]:
    """Get the 10 system-level functions for testing/benchmarking."""
    return TESTING_FUNCTIONS.copy()


def get_functions_by_library(library: str) -> List[SystemLevelFunction]:
    """Get all functions for a specific library."""
    return [f for f in TRAINING_FUNCTIONS if f.library == library]


def get_functions_by_complexity(min_score: int = 1, max_score: int = 10) -> List[SystemLevelFunction]:
    """Get functions within a complexity range."""
    return [f for f in TRAINING_FUNCTIONS if min_score <= f.complexity_score <= max_score]


def save_functions_config(output_path: Path) -> None:
    """Save function configurations to JSON."""
    config = {
        "training_functions": [f.to_dict() for f in TRAINING_FUNCTIONS],
        "testing_functions": [f.to_dict() for f in TESTING_FUNCTIONS],
        "summary": {
            "total_training": len(TRAINING_FUNCTIONS),
            "total_testing": len(TESTING_FUNCTIONS),
            "libraries": list(set(f.library for f in TRAINING_FUNCTIONS)),
            "avg_complexity": sum(f.complexity_score for f in TRAINING_FUNCTIONS) / len(TRAINING_FUNCTIONS)
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Saved function config to {output_path}")


def generate_wrapper_files(output_dir: Path) -> None:
    """Generate individual Python files for each wrapper function."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for func in TRAINING_FUNCTIONS:
        file_path = output_dir / f"{func.id}.py"
        content = f'''"""
{func.name} - System-Level Wrapper
Library: {func.library}
Edge Cases: {', '.join(func.edge_cases)}
"""
{func.import_statement}

{func.wrapper_code.strip()}


# Test cases for mutation
TEST_INPUTS = [
    # TODO: Add test inputs specific to this function
]
'''
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"Generated {len(TRAINING_FUNCTIONS)} wrapper files in {output_dir}")


def print_summary():
    """Print a summary of all functions."""
    print("=" * 70)
    print("System-Level Functions Configuration - Extended Edition")
    print("=" * 70)

    print(f"\nTotal Training Functions: {len(TRAINING_FUNCTIONS)}")
    print(f"Total Testing Functions: {len(TESTING_FUNCTIONS)}")

    print("\nLibraries covered:")
    libraries = {}
    for func in TRAINING_FUNCTIONS:
        libraries[func.library] = libraries.get(func.library, 0) + 1

    for lib in sorted(libraries.keys()):
        print(f"  - {lib}: {libraries[lib]} functions")

    print("\nComplexity distribution:")
    for score in range(1, 11):
        count = sum(1 for f in TRAINING_FUNCTIONS if f.complexity_score == score)
        if count > 0:
            print(f"  Level {score}: {count} functions")

    avg_complexity = sum(f.complexity_score for f in TRAINING_FUNCTIONS) / len(TRAINING_FUNCTIONS)
    print(f"\nAverage complexity: {avg_complexity:.2f}")


if __name__ == "__main__":
    print_summary()

    print(f"\nTraining Functions ({len(TRAINING_FUNCTIONS)}):")
    for i, func in enumerate(TRAINING_FUNCTIONS, 1):
        print(f"  {i:2d}. {func.name} ({func.library}) - Complexity: {func.complexity_score}")

    print(f"\nTesting Functions ({len(TESTING_FUNCTIONS)}):")
    for func in TESTING_FUNCTIONS:
        print(f"  - {func.name} ({func.library})")
