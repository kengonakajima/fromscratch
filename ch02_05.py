import re
text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)

print(result)
