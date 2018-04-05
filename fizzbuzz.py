def fb(n):
  if n%15==0: return 3
  elif n%5==0: return 2
  elif n%3==0: return 1
  else: return 0

fz = ('','fizz', 'buzz','fizzbuzz')

for i in range(1,101):
  print(i,fz[fb(i)])
