import re 

chat1 = "codebasics: you ask alot of questions , 1234568912, abc@xyz.com, 1111111111"
chat2 = "codebasics: here it is: (123)-567-8912, abc@xyz.com"
chat3 = "codebasics: yes, phone: 1234568912 email: abc@xyz.com"
pattern = '\(\d{3}\)-\d{3}-\d{4}'
matches = re.findall(pattern, chat2)
print(matches)