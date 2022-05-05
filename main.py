import csv
import os
import json
from typing import Dict, List
from alive_progress import alive_bar


def main():
    # load the last 200 patents
    patents = get_patents(10)
    print(patents)


JSON_CACHE_PATH = "ibm_patents/patent_metadata_with_keywords.json"
ORIGINAL_PATENTS_PATH = "ibm_patents/patent_metadata.csv"


def get_patents(n_to_get: int, write_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Loads the last `n_to_get` patents from the filesystem,
    with their metadata.
    See the json file for the returned data's shape.
    """
    with open(ORIGINAL_PATENTS_PATH) as file:
        reader = csv.DictReader(file)
        patents = list(reader)
    patents_with_keywords: Dict = {}
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
        # Load patents and their data from cache or calculate the data (and complete the cache)
        for patent_metadata in patents_to_process:
            patent_name = patent_metadata["Name"]
            if patent_name not in patents_with_keywords:
                keywords = get_patent_keywords(patent_name)
                patent = {"date": patent_metadata["Date_Filed"], "keywords": keywords}
                patents_with_keywords_to_return[patent_name] = patent
                patents_with_keywords[patent_name] = patent
            else:  # read from cache
                patents_with_keywords_to_return[patent_name] = patents_with_keywords[
                    patent_name
                ]
            progress_bar()
    # write the cache back to file
    if write_cache:
        with open(JSON_CACHE_PATH, "w") as file:
            json.dump(patents_with_keywords, file, indent=2)
    return patents_with_keywords_to_return


def get_patent_keywords(filename: str) -> List[Dict]:
    import pyate
    import spacy

    # replace .pdf with .txt
    filename = filename.split(".")[0] + ".txt"
    try:
        with open(f"ibm_patents/texts/texts/{filename}") as patent_file:
            content = patent_file.read().replace("\n", " ")
    except:
        return []

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("combo_basic")

    doc = nlp(content)
    res = str(doc._.combo_basic.sort_values(ascending=False).head(5))
    res_lines = res.splitlines()
    res_lines.pop()

    def entry_to_dict(entry: str) -> Dict:
        keyword = entry.split("   ", 1)[0]
        return {"keyword": keyword, "occurences": content.count(keyword)}

    return list(map(entry_to_dict, res_lines))


if __name__ == "__main__":
    main()
