import json

json_string = '''
{
    "imageUrl": "https://"
}
'''


json_object = json.loads(json_string)
    
print(json_object['imageUrl'])

