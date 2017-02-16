from math import *
a = [1.0, 1.3, 0.7]
b= [1.0, -2.0, 2.3]

a_2 = [x*x for x in a]
b_2 = [x*x for x in b]

sums = 0.0
for x,y in zip(a, b):
  sums += x*y

print sums/(sqrt(sum(a_2))*sqrt(sum(b_2)))

e = 0.0
for x,y in zip(a, b):
  e += (x-y)*(x-y)
print sqrt(e)
