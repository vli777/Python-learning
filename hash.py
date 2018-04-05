#VL 3/13/2018
import requests as r
import hashlib

def encrypt_string(hash_string):
    sha_signature = \
        hashlib.sha256(hash_string.encode()).hexdigest()
    return sha_signature

url = ''
req = r.get(url)
req.status_code==r.codes.ok

content = req.text
email = ''
concat = content + email

sha256 = encrypt_string(concat)
print(sha256)
