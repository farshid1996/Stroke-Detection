import random

Num = 3


def numberss():
    numbers = list("0123456789")
    random.shuffle(numbers)
    empty = ""
    for i in range(Num):
        empty += str(numbers[i])
    print(empty)
    return numbers


numberss()
