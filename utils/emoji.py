from numpy.random import choice

pairs = ('๐ฅ.๐', 'โ๏ธ.โ๏ธ', 'โ.โ๏ธ', '๐.๐', '๐.๐', '๐ป.๐', '๐.๐', '๐.๐ง')
pairs = pairs + ('๐.๐ฆ ', '๐.๐ฆ', '๐.๐', '๐.๐ฅซ', '๐งจ.๐ฅ', '๐.๐ฅ')
pairs = pairs + ('๐.๐ค', '๐.๐งถ', '๐ฑ.๐ฅ', '๐.๐', '๐.๐ง', '๐ถ.๐บ')
pairs = pairs + ('๐ฅฅ๐ด๐ฑ', '๐ฅช๐๐พ', '๐ฆ๐ฅ๐ฃ๐ฅ', '๐งฎ๐ค๐น๐ฆง', '๐น๐บ๐ฅ', '๐๐โข๏ธ', '๐ค๐ฝ๐')


def get_train_validation_pair():
    ngram = choice(pairs)
    if '.' in ngram:
        return ngram.split('.')
    a = tuple(a+b for a,b in zip(ngram, ngram[1:]))
    return choice(a)

# get_test_icon = lambda: choice(tuple('๐๐ฎ'))

# if __name__ == '__main__':
#     for i in range(1000):
#         t, v = get_train_validation_pair()
#         print(t + v, end = ' ')