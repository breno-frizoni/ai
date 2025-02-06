#!/usr/bin/python3
import random
from string import ascii_letters as str_letters, digits as str_digits
class password_generator:
    charset = str_letters + str_digits + '@#'
    l = 1
    def __init__(self, pass_len: int, char_set=charset):
        self.l = pass_len
        if char_set:
            self.charset = char_set

    def gen(self) -> str:
        result = ''
        while len(result) < self.l:
            rindex = random.randint(0, len(self.charset)-1)
            result += self.charset[rindex]
        return result

if __name__ == '__main__':
    l = int(input('Insert the password(s) length:'))
    chars = input('Insert your character set to use in password generation or let it blank to use the default: ')
    gerador = password_generator(l, chars)
    p1 = gerador.gen()
    p2 = gerador.gen()
    print(f'{p1}\n{p2}')