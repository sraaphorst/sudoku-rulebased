#!/usr/bin/env python3
import logging
import sys
import sudoku

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} board')
        print('\twhere board is an 81 character string with values from 0 (unspecified) to 9')
        sys.exit(0)

    board = sudoku.from_string(sys.argv[1])
    print(sudoku.solve(board))


