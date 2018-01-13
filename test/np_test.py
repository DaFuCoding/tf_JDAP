import numpy as np
import numpy.random as random

counter = [0, 0, 0]
# for i in range(100000):
#     value = random.random(3)
#     id = value.argmax()
#     counter[id] += 1
#print(counter)


def test_repeat():
    a = np.array([1, 2, 3])
    print(np.repeat(a, 5))
    b = np.reshape(a, [1, a.shape[-1]])
    print(b)

def test_dot():
    a = np.array(range(6))
    a = np.reshape(a, [2, 3])
    scale = [3, 4, 5]
    print(a)
    print(np.dot(a, scale))

def test_add():
    a = np.array(range(6))
    a = np.reshape(a, [2, 3])
    b = np.array([1, 2, 3])
    print(a + b)

if __name__ == '__main__':
    #test_repeat()
    #test_dot()
    #test_add()
    word = 'asd'
    num = 2
    print('%s\n%d' % (word, num))