import csv
import os
import json
from typing import Dict
from alive_progress import alive_bar


def main():
    # load the last 200 patents
    patents = get_patents(200)
    print(patents)


JSON_CACHE_PATH = "ibm_patents/patent_metadata_with_keywords.json"
ORIGINAL_PATENTS_PATH = "ibm_patents/patent_metadata.csv"


def get_patents(n_to_get):
    """
    Loads the last `n_to_get` patents from the filesystem.
    """
    with open(ORIGINAL_PATENTS_PATH) as file:
        reader = csv.DictReader(file)
        patents = list(reader)
    patents_with_keywords: Dict = {}
    # a+ = append or create, can also read
    if os.path.exists(JSON_CACHE_PATH):
        with open(JSON_CACHE_PATH, "r") as file:
            patents_with_keywords = json.load(file)
    else:
        with open(JSON_CACHE_PATH, "w") as file:
            json.dump({}, file)
    patents.sort(key=lambda p: p["Date_Filed"], reverse=True)
    patents_to_process = patents[:n_to_get]
    # This is here because we don't want to return everything in `patents_with_keywords`,
    # but we still want to keep it to write it back to disk.
    patents_with_keywords_to_return = {}
    with alive_bar(len(patents_to_process)) as progress_bar:
        for patent_metadata in patents_to_process:
            patent_name = patent_metadata["Name"]
            if patent_name not in patents_with_keywords:
                # replace .pdf with .txt
                # filename = patent_metadata["Name"].split(".")[0] + ".txt"
                # with open("ibm_patents/texts/texts/" + filename) as patent_file:
                patent = {"date": patent_metadata["Date_Filed"], "keywords": ["TODO"]}
                patents_with_keywords_to_return[patent_name] = patent
                patents_with_keywords[patent_name] = patent
            else:  # read from cache
                patents_with_keywords_to_return[patent_name] = patents_with_keywords[
                    patent_name
                ]
            progress_bar()
    with open(JSON_CACHE_PATH, "w") as file:
        json.dump(patents_with_keywords, file)
    return patents_with_keywords_to_return


if __name__ == "__main__":
    main()
