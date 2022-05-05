import csv
from datetime import date
from itertools import count
import itertools
import os
import json
from typing import Any, Dict, List, Sequence, Set, TypedDict
from alive_progress import alive_bar
from dateutil import rrule
from datetime import datetime, timedelta

TW = 0.05


def main():
    # load the last x patents
    patents = get_patents(1000)
    patent_list = patents.values()
    # Sort by date.
    patent_list = sorted(patent_list, key=lambda p: p["date"])
    # Convert dates to datetime
    def date_to_datetime(patent):
        [year, month, day] = patent["date"].split("-")
        patent["date"] = datetime(int(year), int(month), int(day))
        return patent

    patent_list = list(map(date_to_datetime, patent_list))
    # The number of documents in each month.
    documents_per_month = {}
    for patent in patent_list:
        date: datetime = patent["date"]
        month = datetime(date.year, date.month, 1)
        if month in documents_per_month:
            documents_per_month[month] += 1
        else:
            documents_per_month[month] = 0
    number_of_months = len(documents_per_month)
    # Every unique keyword present in an article.
    unique_keywords: Set[str] = set()
    for patent in patent_list:
        for keyword_data in patent["keywords"]:
            unique_keywords.add(keyword_data["keyword"])
    month_list = sorted(documents_per_month.keys())
    first_month = month_list[0]
    last_month = month_list[-1]
    for [idx, month] in zip(
        itertools.count(start=0, step=1),
        rrule.rrule(rrule.MONTHLY, dtstart=first_month, until=last_month),
    ):
        patents_in_month = [
            patent
            for patent in patent_list
            if patent["date"].month == month.month and patent["date"].year == month.year
        ]
        for keyword in unique_keywords:
            patents_containing_keyword = [
                patent
                for patent in patents_in_month
                if any([k["keyword"] == keyword for k in patent["keywords"]])
            ]
            # docs_containing_keyword = list(
            #     filter(
            #         lambda p: any(
            #             map(lambda k: k["keyword"] == keyword, p["keywords"])
            #         ),
            #         patents_in_month,
            #     )
            # )
            df = len(patents_containing_keyword)

            def extract_occurences(patent, keyword):
                matching_keywords = [
                    k["occurences"]
                    for k in patent["keywords"]
                    if k["keyword"] == keyword
                ]
                if len(matching_keywords) != 0:
                    return matching_keywords[0]
                else:
                    return 0

            occurences_of_keyword_in_each_patent = [
                extract_occurences(patent, keyword)
                # There should only be one item, we can take do [0].
                for patent in patents_containing_keyword
            ]
            # occurences_of_keyword_in_each_patent = map(
            #     lambda p: filter(
            #         lambda kw: kw["keyword"] == keyword, p["keywords"]
            #     ).__next__()["occurences"],
            #     patents_containing_keyword,
            # )
            tf = sum(occurences_of_keyword_in_each_patent)

            nn = documents_per_month[month]

            intermediate_value = ((1 - TW) * idx) / nn

            print(date.year, date.month, keyword, "dod", df * intermediate_value)
            print(date.year, date.month, keyword, "dov", tf * intermediate_value)


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
