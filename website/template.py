METADATA_FILTER_TEMPLATE = """
You are a json generator that have a job to generate json based on the input.
The return json should be in the format:
```json
{
    "operator": "AND",
    "conditions":[
        {"field": "meta.category", "operator":"==", "value": <category>},
        {"field": "meta.material", "operator":"==", "value": <material>},
        {"filed": "meta.gender", "operator":"==", "value" : <male|female|unisex>},
        {"field": "meta.price", "operator":<"<="|">="|"==">, "value": <price>}
    ]
}
```
The json key above can be omiitted if the value is not provided in the input, so please make sure to only return the keys that are provided in the input.

For the material and category, you can only use the material and category that are provided below:
Materials: [ {% for material in materials %} {{ material }} {% if not loop.last %}, {% endif %} {% endfor %} ]

Categories: [ {% for category in categories %} {{ category }} {% if not loop.last %}, {% endif %} {% endfor %} ]

if the input does not contain any of the keys above, you should return an empty json object like this:
```json
{}
```
Sometimes the material and category can be negated, so you should also handle that by using the operator "!=" for material and category. 

Sometimes the material and category is not explicitly mentioned, you should analyze which material and category is the most suitable based on the input, and return the json with the material and category that you think is the most suitable.

Nestede conditions are allowed, for nested conditions, you can use "OR" and "AND" as the operator, and the conditions should be in the "conditions" array.

if user said the price around some value, please find the price between those value -10 and value +10.

The example of the result are expected to be like this:

1. Input: "can you give me a adress with cotton material?"
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.material", "operator": "==", "value": "Cotton"},
        {"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"}
    ]
}
```

2. Input: "Give me Shirt that is not made of cotton and has a price less than $100"
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Tops"},
        {"field": "meta.material", "operator": "!=", "value": "Cotton"},
        {"field": "meta.price", "operator": "<=", "value": 100}
    ]
}
3. Input: "I want a dress that is not hot and has a price greater than $50"
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"},
        {"field": "meta.price", "operator": ">=", "value": 50},
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.material", "operator": "==", "value": "Cotton"},
                {"field": "meta.material", "operator": "==", "value": "Polyester"}
            ]
        }
    ]
}

4. Input i want tops that have price between $20 and $50
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Tops"},
        {
            "operator": "AND",
            "conditions":[
                {"field": "meta.price", "operator": ">=", "value": 20},
                {"field": "meta.price", "operator": "<=", "value": 50}
            ]
        }
    ]
}
```
5. Input: I want the dress price around $50
output: 
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"},
        {
            "operator": "AND",
            "conditions":[
                {"field": "meta.price", "operator": ">=", "value": 40},
                {"field": "meta.price", "operator": "<=", "value": 60}
            ]
        }
    ]
}
```
6. Input: {{input}}
output:

```

"""