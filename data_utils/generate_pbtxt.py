import pandas as pd

def generate_pbtxt(xml_df):
    i = 1
    pbtxt_str = ''
    for cl in xml_df['class'].unique():
        pbtxt_str += 'item {\n  name: \"' + cl + '\"\n  id: ' + str(i) + '\n}\n'
        i += 1
    return pbtxt_str