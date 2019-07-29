def Find(self, target, array):
    # write code here
    i = -1
    for item in array:
        if target > item[0]:
            i += 1
        else:
            break
    if i == -1:
        return False

    for item in array[i]:
        if target == item:
            return True

            return False


def Fibonacci(n):
    f = 0
    g = 1

    while n>=0:

        g = f + g
        f = g - f
        n = n - 1
    return f;



def fun(x) :
    countx = 0;
    while x>0:
        countx+=1;
        x = (x & (x - 1))

    return countx;

print(fun(500))