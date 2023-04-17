# importing libraries
import time


# decorator to calculate duration taken by any function.
# https://www.geeksforgeeks.org/decorators-in-python/
def calculate_time(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner(*args, **kwargs):
        # storing time before function execution
        begin = time.time()

        func(*args, **kwargs)

        # storing time after function execution
        end = time.time()
        print("Total time taken in {} was {} seconds.".format(func.__name__, end - begin))

    return inner
