

import bisect
from typing import Generator

import numpy as np


def find_row_for_target(row_dividers, ytarget):
    """
    Find the row that a box belongs to, according to the row dividers.
    The row_dividers do not include endbounds.
    """
    return bisect.bisect_left(row_dividers, ytarget)

def find_column_for_target(column_dividers, xtarget):
    """
    Find the column that a box belongs to, according to the column dividers.
    The column_dividers do not include endbounds.
    """
    return bisect.bisect_left(column_dividers, xtarget)

# def find_rows_for_interval(row_dividers, table_bounds, yinterval, threshold=0):
#     """
#     Find the rows that intersect with the interval by at least threshold.
#     Assume that yinterval is larger, and therefore the ioa is divided by the row size.

#     Returns list of indices, with domain [0, len(row_dividers)+1]
#     """
#     leftmost = bisect.bisect_left(row_dividers, yinterval[0])
#     rightmost = bisect.bisect_right(row_dividers, yinterval[1])

#     # valid rows are [leftmost, rightmost], inclusive
#     # now filter by threshold
#     if threshold == 0:
#         return list(range(leftmost, rightmost+1))
#     valid_rows = []
#     consider = [table_bounds[1]] + row_dividers + [table_bounds[3]] # len: len(row_dividers) + 2
#     for i in range(leftmost, rightmost+1):
#         row_y_interval = (consider[i], consider[i+1])
#         if _ioa(row_y_interval, yinterval) > threshold:
#             valid_rows.append(i)

def _find_all_intervals_for_interval(sorted_intervals, interval, threshold=0):
    """
    Find all intervals that intersect with the interval.
    """
    start, end = interval
    left = bisect.bisect_right([i[1] for i in sorted_intervals], start)
    right = bisect.bisect_left([i[0] for i in sorted_intervals], end)
    
    result = sorted_intervals[left:right]
    if threshold == 0:
        return result
    return [x for x in result if _ioa(x, interval) > threshold]

def find_nearest_divider(dividers, target):
    """
    Find the index of the nearest divider to the target.
    """
    if len(dividers) == 0:
        return None
    return min(range(len(dividers)), key=lambda i: abs(dividers[i] - target))

def fill_using_true_partitions(text_positions: Generator[tuple[float, float, float, float, str], None, None], 
                          row_dividers: list[dict], column_dividers: list[dict], table_bounds: tuple[float, float, float, float], projecting_indices: list[dict] = None):
    """
    Given estimated positions of text positions,
    row dividers (does not include endbounds), column dividers (does not include endbounds),
    fills the table array.

    assumes dividers are sorted
    """

    num_rows = len(row_dividers) + 1
    num_columns = len(column_dividers) + 1
    table_array = np.empty([num_rows, num_columns], dtype="object")
    table_array_bbox = [[[] for _ in range(num_columns)] for _ in range(num_rows)]
    #print(proj_divider_intervals)
    #print(row_dividers)
    is_projecting_matches = []
    space_size = 0
    space_size_count = 0
    for xmin, ymin, xmax, ymax, text in text_positions:
        if text is not None:
            text = text.replace('·','.').replace('  ',' ')
        # 5. let row_num be row with max iob
        xtarget = (xmin + xmax) / 2
        ytarget = (ymin + ymax) / 2

        # if completely outside the bounds (no intersection), ignore
        if not (table_bounds[0] <= xtarget <= table_bounds[2] and table_bounds[1] <= ytarget <= table_bounds[3]):
            continue

        # to find x, we need the first column divider (moving LTR) where xtarget < xdivider

        column_num = find_column_for_target(column_dividers, xtarget)
        # then, it belongs to the xi-th column of the np array (no off by 1 error)
        is_projecting = False
        row_num = find_row_for_target(row_dividers, ytarget)
        if projecting_indices is not None and row_num in projecting_indices:
            is_projecting = True
            #column_num = 0
            if row_num not in is_projecting_matches:
                is_projecting_matches.append(row_num)
            
        # if is_projecting and space_size > 0 and row_num in is_projecting_matches:
        #     if column_num > 0:
        #         if len(table_array_bbox[row_num][column_num-1]) > 0 and xmin - table_array_bbox[row_num][column_num-1][-1][2] > 1.5 * space_size:
        #     #remove from is_projecting_matches
        #             is_projecting_matches.remove(row_num)
        

            
        # if proj_divider_intervals is not None and ( len(proj_divider_intervals) > 0):
        #     row_num_projecting =  find_nearest_divider(proj_divider_intervals, ytarget )
        #     if row_num_projecting >-1:
        #         print(text + " -- " + str(ytarget))
        #         print(row_num)
        #         print(row_num_projecting)
        #         if(row_num_projecting > 0):
        #             row_num_is_projecting = find_nearest_divider(row_dividers , proj_divider_intervals[row_num_projecting])
        #             print(row_num_is_projecting)
        #             print("=======")
        #             if row_num_is_projecting == row_num:
        #                 is_projecting = True
        #                 column_num = 0
        
        if text is not None:
            if table_array[row_num, column_num] is not None:
                table_array[row_num, column_num] += ' ' + text
                # get last bbox 
                if not is_projecting:
                    xmin_p, ymin_p, xmax_p, ymax_p = table_array_bbox[row_num][column_num][-1]
                    if(xmin > xmax_p):
                        if space_size == 0:
                            space_size = xmin - xmax_p
                        else:
                            space_size = (space_size * space_size_count + xmin - xmax_p) / (space_size_count + 1)
                        space_size_count += 1
                    else:
                        print(f"ERROR with bbox while calculating space size {str(xmin)} - {str(xmax_p)}")
            else:
                table_array[row_num, column_num] = text
        table_array_bbox[row_num][column_num].append((xmin, ymin, xmax, ymax))
                    
    for i in is_projecting_matches:
        if table_array[i, 0] is None or table_array[i, 0] == '':
            continue
        if(num_columns > 1):
            if space_size > 0 and len(table_array_bbox[i][1]) > 0 and len(table_array_bbox[i][0]) > 0 and table_array_bbox[i][1][0][0] - table_array_bbox[i][0][-1][2] > 2.5 * space_size:
                print(space_size)
                #continue
        for j in range(1, num_columns):
            if table_array[i, j] is not None and table_array[i, j] != '':
                table_array[i, 0] += ' ' + table_array[i, j]
                table_array[i, j] = '' 
        table_array[i, 0] = '**' + table_array[i, 0] + '**'

    return table_array

def _ioa(a: tuple[float, float], b: tuple[float, float]) -> float:
    """
    Calculate the intersection of (closed) intervals, divided by the first interval a.
    
    If a is a single point [x, x], then return 1 if x is in the interior of b, 0 otherwise.
    """
    a0, a1 = a
    b0, b1 = b
    if a0 > b1 or a1 < b0:
        return 0
    if a0 == a1:
        return 1 if b0 < a0 < b1 else 0
    
    return (min(a1, b1) - max(a0, b0)) / (a1 - a0)

def get_good_between_dividers(dividers: list[tuple[float, float]], min_val: float, max_val: float, add_inverted=True):
    """
    Get the good content between dividers.
    
    :param dividers: list of dividers, each a tuple of form (start, end).
    :param rows: only 
    """
    result = []

    prev_end = min_val
    for i, (start, end) in enumerate(dividers):
        if start > prev_end:
            result.append((prev_end, start))
        else:
            if add_inverted:
                # begrudgingly add the inverted interval (psuedo-row, which is likely very thin) to keep things balanced
                result.append((start, prev_end))
            else:
                pass
        prev_end = end
    
    # the last interval
    if prev_end < max_val:
        result.append((prev_end, max_val))
    else:
        if add_inverted:
            result.append((max_val, prev_end))
    return result