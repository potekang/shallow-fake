with open ("japanese_text", encoding = "utf-8") as myfile:
    chars = myfile.read().replace('\n', '')
charset = list(chars)
print(charset)

