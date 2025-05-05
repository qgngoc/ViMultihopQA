import urllib.parse

def convert_encoded_text(encoded_text):
    return urllib.parse.unquote(encoded_text)