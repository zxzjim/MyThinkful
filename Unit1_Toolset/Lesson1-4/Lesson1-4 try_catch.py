a = 1
b = 0
try:
    a / b
except ZeroDivisionError: #indent here is very important
    print "Cannot divide by zero."