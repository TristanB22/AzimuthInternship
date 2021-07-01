a, _, b = (1, 2, 3) #1 skip 2 print 3
print(a, b)

a, *_, b = (7, 6, 5, 4, 3, 2, 1)
print(a, b) # 7 skip * (all) grab 1 it's like the linux *. rm -rf * is probably not a good idea in some locations. 


for _ in range(5):
	print(_) #it is able to loop through a range of 5 starting from 0, so the output is 4. 

names = ["And", "The", "Cat"]
for _ in names:
	print(_) #We can see it is able to be used in loops to grab list of names


_ = 5
while _ < 10:
	print(_, end = '') # it is able to be used to start a count from 5 to a number less than that which is defined. 
	_+=1