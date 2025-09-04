from dataclasses import dataclass
from typing import Union
from dataclass_wizard import YAMLWizard

@dataclass
class ConfigA(YAMLWizard):
    type: str = "ConfigA"
    value: int = 1

@dataclass
class ConfigB(YAMLWizard):
    type: str = "ConfigB"
    text: str = "test"

ConfigUnion = Union[ConfigA, ConfigB]

@dataclass
class Container(YAMLWizard):
    item: ConfigUnion

# Test 1: YAML with type field  
yaml_str = '''
item:
  type: ConfigA
  value: 42
'''

try:
    c = Container.from_yaml(yaml_str)
    print('Test 1 Success:', c)
except Exception as e:
    print('Test 1 Error:', e)

# Test 2: Try with actual instance
c2 = Container(item=ConfigA(value=99))
yaml_out = c2.to_yaml()
print('\nGenerated YAML:')
print(yaml_out)

# Test 3: Try to load it back
try:
    c3 = Container.from_yaml(yaml_out)
    print('Round-trip Success:', c3)
except Exception as e:
    print('Round-trip Error:', e)
