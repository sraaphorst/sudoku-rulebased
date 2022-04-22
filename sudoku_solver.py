#!/usr/bin/env python3
import logging
import sys
import sudoku

# Here are two test boards:
# 000000573800020000700900810580706000001800060230040009915000000000080601000000040
# 800000000003600000070090200050007000000045700000100030001000068008500010090000400

# Ultra-hard boards:
# Platinum Blonde:
# 000000012000000003002300400001800005060070800000009000008500000900040500470006000

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} board')
        print('\twhere board is an 81 character string with values from 0 (unspecified) to 9')
        sys.exit(0)

    board = sudoku.from_string(sys.argv[1])
    logging.info(board)
    solved = sudoku.solve(board)
    print(board)
    print(solved)
    print(f'Solved: {sudoku.check_valid(solved)}')
