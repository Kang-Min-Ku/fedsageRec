import re
import os
import pandas as pd

class Extractor:
    def __init__(
        self,
        pattern=r"### final Test ### precision Score = (?P<precision>0.[\d]+), recall Score = (?P<recall>0.[\d]+), ndcg Score = (?P<ndcg>0.[\d]+)"
    ):
        self.pattern = pattern
        self.regex = re.compile(self.pattern)

    def extract_file(self, filename):
        result = []
        with open(filename, "r") as fd:
            text = fd.read()
        for l in self.regex.finditer(text):
            result.append(l.groupdict())

        result = result[-1]

        return result

    def extract_dir(self, path, extension=".log"):
        result = {}
        for filename in os.listdir(path):
            if filename.endswith(extension):
                result[filename] = self.extract_file(os.path.join(path, filename))

        result_df = pd.DataFrame(result).T

        return result_df
    
