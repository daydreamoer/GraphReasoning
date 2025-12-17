from json_repair import repair_json

bad_json = '{"user": {"name": "Alice", "age": 25, "city": "Los Angeles'
repaired_json = repair_json(bad_json)

print(repaired_json)