import torch
import time

# define a timer class to record time
class myTimer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # start the timer
        self.start_time = time.time()

    def stop(self):
        # stop the timer and record time into a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # calculate the average and return
        return sum(self.times)/len(self.times)

    def sum(self):
        # return the sum of recorded time
        return sum(self.times)
#  Test the time class
# init variable a, b as 1000 dimension vector
# n = 1000
# a = torch.ones(n)
# b = torch.ones(n)
# timer = myTimer()
# c = torch.zeros(n)
# for i in range(n):
#     c[i] = a[i] + b[i]
# print('%.10f sec' % timer.stop())
#
# timer.start()
# d = a + b
# print('%.10f sec' % timer.stop())