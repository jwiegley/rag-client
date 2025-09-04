from dataclasses import dataclass
from typing import Union
from dataclass_wizard import YAMLWizard

@dataclass
class ConfigA(YAMLWizard):
    value: int = 1

@dataclass
class ConfigB(YAMLWizard):
    text: str = "test"

ConfigUnion = Union[ConfigA, ConfigB]

@dataclass
class Container(YAMLWizard):
    item: ConfigUnion

# Test with __tag__ field  
yaml_str = '''
item:
  __tag__: ConfigA
  value: 42
'''

try:
    c = Container.from_yaml(yaml_str)
    print('With __tag__ Success:', c)
except Exception as e:
    print('With __tag__ Error:', e)
