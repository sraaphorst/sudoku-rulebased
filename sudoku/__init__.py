import logging
import numpy as np
import numpy.typing as npt
from typing import List, Mapping, Optional, Tuple


def from_string(board: str) -> npt.NDArray[int]:
    """
    Convert a string of length 81 with values from 0 to 9 into a 9x9 matrix.
    Note that this does not mean that the board is legal: a board can have zero solutions.
    Raises a ValueError for incorrectly sized boards or boards containing illegal characters.
    """
    if len(board) != 81:
        raise ValueError(f'Illegal board length: {len(board)}')
    return np.array([int(i) for i in board], dtype=np.int).reshape((9, 9))


def solve(board: npt.NDArray[int]) -> Optional[npt.NDArray[int]]:
    """
    Solve the sudoku board using a rule-based approach and backtracking when necessary.
    """
    board_solved = board.copy()

    # Extract a grid.
    def extract_grid(grid_row: int, grid_col: int) -> npt.NDArray[int]:
        row_low, col_low = grid_to_board(grid_row, grid_col, 0, 0)
        row_high, col_high = grid_to_board(grid_row, grid_col, 3, 3)
        return board_solved[row_low:row_high, col_low:col_high]

    def extract_grid_candidates(grid_row: int, grid_col: int) -> npt.NDArray[int]:
        row_low, col_low = grid_to_board(grid_row, grid_col, 0, 0)
        row_high, col_high = grid_to_board(grid_row, grid_col, 3, 3)
        return board_candidates[row_low:row_high, col_low:col_high]

    # Convert from grid coordinates to board coordinates.
    def grid_to_board(grid_row: int, grid_col: int, row_offset: int, col_offset: int) -> Tuple[int, int]:
        return 3 * grid_row + row_offset, 3 * grid_col + col_offset

    # Convert from board coordinates to grid coordinates:
    def board_to_grid(row: int, col: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (row // 3, col // 3), (row % 3, col % 3)

    # Get the candidates for the row and column from the currently solved board.
    # This will never be called for a board with a value already fixed in that position.
    # If this returns an empty list, we have an unsolvable board.
    def find_candidates(row: int, col: int) -> npt.NDArray[int]:
        # Determine what is already covered to get the non-candidates and then take the set difference.
        # Since this cell is included in the "covered" numbers and is zero, zero will be eliminated
        # automatically by the setdiff1d.
        row_set = np.unique(board_solved[row, :])
        col_set = np.unique(board_solved[:, col])
        grid_set = np.unique(extract_grid(row // 3, col // 3))
        covered = np.unique(np.concatenate((row_set, col_set, grid_set)))
        return np.setdiff1d(np.arange(10), covered)

    # Given the data for a row, column, or grid, for each number not already appearing in that row / column / grid,
    # determine the cells for which is can be a candidate.
    # For the output, we get a map of lists of indexes where the number appears in the candidate lists.
    def find_cells(candidate_lists: npt.NDArray[npt.NDArray[int]], board_contents: npt.NDArray[int]) ->\
            Mapping[int, List[int]]:
        cells = {}
        for number in range(1, 10):
            # Skip over numbers that are already in this row, column, or grid since we don't need to think about them.
            if number not in board_contents:
                cells[number] = [i for i, candidates in enumerate(candidate_lists) if number in candidates]
        return cells

    # We continue until there are no zeros left.
    prev_zeros = 81 - np.count_nonzero(board_solved)

    # For some boards, we will need to backtrack.
    # When we backtrack, we will find the first cell (row, col) with the least candidates, and
    # append the following to the backtracking list:
    # (current_board_state, row, col, cell_candidates, 0)
    # and set (row, col) to cell_candidates[0].
    #
    # If we don't reach a solution from this backtrack, we will pop off the state we saved,
    # and then reset the board and try the next cell candidate, e.g. if the last one was 0, we would write:
    # (current_board_state, row, col, cell_candidates, 1)
    # and set (row, col) to cell_candidates[1].
    #
    # If we ever run out of cell_candidates, the board is unsolvable.
    backtracking = []

    it = 1
    while prev_zeros:
        logging.info(f'*** Iter {it} ***')
        it += 1

        # Calculate the candidates for each cell.
        # Ragged arrays are deprecated, but this will simplify things considerably by being able
        # to use numpy slices.
        board_candidates = np.array([np.array([]) if board_solved[row, col] else find_candidates(row, col)
                                     for row in range(9)
                                     for col in range(9)], dtype=object).reshape((9,9))

        # *** LOCKED CELLS ***
        # If, for a given grid, the places where a number can occur all appear in the same row or the same column,
        # then they must appear in that grid in that row and that column.
        # They cannot, then, appear in other grids in the same row (resp col) in that row (resp col).
        # Example: Say we have a grid row like this:
        # x x . | . x . | . x .
        # . . . | x x . | . . x
        # . . . | . x x | x . .
        # In the first grid, x must appear in row 0. This means that it cannot appear in row 0 in the other
        # two grids and must be eliminated:
        # x x . | . . . | . . .
        # . . . | x x . | . . x
        # . . . | . x x | x . .
        for grid_row in range(3):
            for grid_col in range(3):
                for number in range(1, 10):
                    # Get the possible cell locations of number in the given grid.
                    # Note these are offsets.
                    cells = [(r, c) for r in range(3) for c in range(3)
                             if number in extract_grid_candidates(grid_row, grid_col)[r][c]]

                    # Check to see if all the row_offsets are the same. Cannot use generators here.
                    row_offsets = np.unique([r for r, _ in cells])
                    if row_offsets.size == 1:
                        # Eliminate from the other row_offsets in the other two grids in this grid_row.
                        for other_grid_col in range(3):
                            if other_grid_col == grid_col: continue
                            for col_offset in range(3):
                                row, col = grid_to_board(grid_row, grid_col, row_offsets[0], col_offset)
                                board_candidates[row, col] = np.setdiff1d(board_candidates[row, col], number)

                    # Check to see if all the col_offsets are the same. Cannot use generators here.
                    col_offsets = np.unique([c for _, c in cells])
                    if col_offsets.size == 1:
                        # Eliminate from the other col_offsets in the other two grids in this grid_col.
                        for other_grid_row in range(3):
                            if other_grid_row == grid_row: continue
                            for row_offset in range(3):
                                row, col = grid_to_board(grid_row, grid_col, row_offset, col_offsets[0])
                                board_candidates[row, col] = np.setdiff1d(board_candidates[row, col], number)

        # *** BACKTRACKING ***
        # First check to see if we've reached a dead end. If we have, we have to backtrack.
        dead_end = False
        for row, col in [(x, y) for x in range(9) for y in range(9)]:
            if board_solved[row, col] == 0 and board_candidates[row][col].size == 0:
                logging.info(f'Dead end found at ({row},{col}), candidates: {board_candidates[row][col]}')
                logging.info(f'Board:\n{board_solved}')
                dead_end = True
                break

        # Backtrack if we reached a dead end, and we can do so.
        if dead_end:
            # If there was no backtracking done at all, then we can't backtrack and the board is not solvable.
            if not backtracking:
                return None

            # Revert to the state at the time of the backtracking.
            board_solved, row, col, cell_candidates, idx = backtracking.pop()

            # If we have exhausted all possibilities in the backtracking, then the board is not solvable.
            idx += 1
            if idx == len(cell_candidates):
                return None

            # We can backtrack, so set the new state, and loop.
            logging.info(f'Setting ({row},{col}) to {cell_candidates[idx]} via backtracking')
            board_solved[row, col] = cell_candidates[idx]
            backtracking.append((board_solved.copy(), row, col, cell_candidates, idx))
            continue

        # *** NUMBER CANDIDATES ***
        # Calculate the candidate cells for each number in each row, column, and grid.
        # For example, determine in row 0 all the places 6 can appear.
        number_row_candidates = [find_cells(board_candidates[row,:], board_solved[row,:]) for row in range(9)]
        number_col_candidates = [find_cells(board_candidates[:,col], board_solved[:,col]) for col in range(9)]

        # The grids are much more complicated.
        # The final result will be a map of grid_row and grid_col.
        number_grid_candidates = {}
        for grid_row in range(3):
            for grid_col in range(3):
                # We have to reshape to 1D lists to make this work.
                candidate_lists = board_candidates[
                                  (3 * grid_row):(3 * (grid_row + 1)),
                                  (3 * grid_col):(3 * (grid_col + 1))
                                  ].reshape((9,))
                board_contents = board_solved[
                                 (3 * grid_row):(3 * (grid_row + 1)),
                                 (3 * grid_col):(3 * (grid_col + 1))
                                 ].reshape((9,))
                unadjusted_cells = find_cells(candidate_lists, board_contents)

                # We now have a 1D list of cells, but we want them in a 2D form of 3x3 to represent what is
                # really happening here, i.e. in the grid indexed by (grid_row, grid_col), the subcell at
                # (row, col) has the number as a candidate.
                adjusted_cells = {}
                for number, cells in unadjusted_cells.items():
                    adjusted_cells[number] = [divmod(idx, 3) for idx in cells]
                number_grid_candidates[(grid_row, grid_col)] = adjusted_cells

        # *** NAKED SINGLES ***
        # Handle all the naked singles, i.e. there is only one possible candidate.
        for row, board_candidates_row in enumerate(board_candidates):
            for col, cell_candidates in enumerate(board_candidates_row):
                if cell_candidates.size == 1:
                    logging.info(f'Setting ({row},{col}) to {cell_candidates[0]} via naked singles')
                    board_solved[row, col] = cell_candidates[0]

        # *** HIDDEN SINGLES ***
        # Hidden singles occur if we have a row, column, or grid where there is only one possible position
        # where a number can appear. The cell containing it might have many overall candidates, but that number
        # only occurs in this cell for the row, column, or grid.
        # Check each row, column, and grid for hidden singles.
        for number in range(1, 10):
            for row, number_cells in enumerate(number_row_candidates):
                if number in number_cells and len(number_cells[number]) == 1:
                    board_solved[row, number_cells[number][0]] = number
            for col, number_cells in enumerate(number_col_candidates):
                if number in number_cells and len(number_cells[number]) == 1:
                    board_solved[number_cells[number][0], col] = number
            for (grid_row, grid_col), cells in number_grid_candidates.items():
                if number in cells and len(cells[number]) == 1:
                    row_offset, col_offset = cells[number][0]
                    row, col = grid_to_board(grid_row, grid_col, row_offset, col_offset)
                    logging.info(f'Setting ({row},{col}) to {number} via hidden singles')
                    board_solved[row, col] = number

        # *** BACKTRACK REVISITED ***
        new_zeros = 81 - np.count_nonzero(board_solved)
        if prev_zeros == new_zeros:
            # We could not get any further with the logic we have, so initiate backtracking.
            # Find one cell that is unsolved with the fewest candidates.
            nonempty = [(r, c, board_candidates[r, c]) for r in range(9) for c in range(9)
                        if board_candidates[r, c].size > 0]
            row, col, cell_candidates = min(nonempty, key=lambda x: x[2].size)
            logging.info(f'Setting ({row},{col}) to {cell_candidates[0]} via backtracking initiation')
            board_solved[row, col] = cell_candidates[0]
            backtracking.append((board_solved.copy(), row, col, cell_candidates, 0))
            new_zeros -= 1

        prev_zeros = new_zeros

    print(board_solved)
    return None
