from numpy.random import choice

pairs = ('🔥.🎃', '⛄️.❄️', '⛈.☁️', '🌚.🌝', '🌞.🌜', '👻.😇', '🌕.🌗', '🍄.🌧')
pairs = pairs + ('💊.🦠', '🐚.🦕', '🎓.📖', '🍝.🥫', '🧨.💥', '💎.💥')
pairs = pairs + ('🍙.🍤', '👘.🧶', '🍱.🥟', '🎋.🎐', '🍜.🧂', '🍶.🍺')
pairs = pairs + ('🥥🌴🌱', '🥪🍞🌾', '🦆🐥🐣🥚', '🧮🤔🛹🦧', '🎹🎺🥁', '🌏🌞☢️', '🤖👽🚀')


def get_train_validation_pair():
    ngram = choice(pairs)
    if '.' in ngram:
        return ngram.split('.')
    a = tuple(a+b for a,b in zip(ngram, ngram[1:]))
    return choice(a)

# get_test_icon = lambda: choice(tuple('🏁🔮'))

# if __name__ == '__main__':
#     for i in range(1000):
#         t, v = get_train_validation_pair()
#         print(t + v, end = ' ')