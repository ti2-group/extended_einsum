from typing import get_args

from extended_einsum.language import BinaryOperator

operator = "1"
match operator:
    case operator if operator in get_args(BinaryOperator):
        print("Binary operator")
    case _:
        print("Not a binary operator")
