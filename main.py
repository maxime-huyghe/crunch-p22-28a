import csv
from functools import reduce
import itertools
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Set
from alive_progress import alive_bar
from dateutil import rrule
from datetime import datetime, timedelta
import argparse
import numpy
from scipy.stats.mstats import gmean

TW = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()
    n_to_get = vars(args)["n"]
    # load the last x patents
    patents = get_patents(n_to_get)
    patent_list = patents.values()
    # Sort by date.
    patent_list = sorted(patent_list, key=lambda p: p["date"])
    # Convert dates to datetime
    def date_to_datetime(patent):
        [year, month, day] = patent["date"].split("-")
        patent["date"] = datetime(int(year), int(month), int(day))
        return patent

    patent_list = list(map(date_to_datetime, patent_list))
    # The number of documents in each year.
    documents_per_year = {}
    for patent in patent_list:
        date: datetime = patent["date"]
        year = datetime(date.year, 1, 1)
        if year in documents_per_year:
            documents_per_year[year] += 1
        else:
            documents_per_year[year] = 0
    number_of_years = len(documents_per_year)
    # Every unique keyword present in an article.
    unique_keywords: Set[str] = set()
    for patent in patent_list:
        for keyword_data in patent["keywords"]:
            unique_keywords.add(keyword_data["keyword"])
    year_list = sorted(documents_per_year.keys())
    first_year = year_list[0]
    last_year = year_list[-1]
    dods = {}  # {[keyword, year] -> dod}
    dovs = {}

    for [idx, year] in zip(
        itertools.count(start=0, step=1),
        rrule.rrule(rrule.YEARLY, dtstart=first_year, until=last_year),
    ):
        print(year)
        patents_in_year = [
            patent for patent in patent_list if patent["date"].year == year.year
        ]
        for keyword in unique_keywords:
            patents_containing_keyword = [
                patent
                for patent in patents_in_year
                if any([k["keyword"] == keyword for k in patent["keywords"]])
            ]
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
            tf = sum(occurences_of_keyword_in_each_patent)

            nn = documents_per_year[year]

            # pas idx mais nb de périodes - idx
            intermediate_value = (1 - TW * (len(year_list) - idx)) / nn

            dod = df * intermediate_value
            dods[(keyword, year.year)] = {"dod": dod}
            if (keyword, year.year - 1) in dods:
                old_value = dods[(keyword, year.year - 1)]["dod"]
                if old_value != 0:
                    increasing_rate = (dod - old_value) / old_value
                    dods[(keyword, year.year)]["increasing_rate"] = increasing_rate
                else:
                    # increasing_rate = 1
                    pass
            dov = tf * intermediate_value
            dovs[(keyword, year.year)] = {"dov": dov}
            if (keyword, year.year - 1) in dovs:
                old_value = dovs[(keyword, year.year - 1)]["dov"]
                if old_value != 0:
                    increasing_rate = (dov - old_value) / old_value
                    dovs[(keyword, year.year)]["increasing_rate"] = increasing_rate
                else:
                    # increasing_rate = 1
                    pass
    print(dods)
    print(dovs)

    keywords_averages = {}
    for keyword in unique_keywords:
        dod_keys_with_keyword = [key for key in dods.keys() if key[0] == keyword]
        dov_keys_with_keyword = [key for key in dovs.keys() if key[0] == keyword]
        dod_increasing_rates = [
            dods[key]["increasing_rate"]
            for key in dod_keys_with_keyword
            if "increasing_rate" in dods[key]
        ]
        dov_increasing_rates = [
            dovs[key]["increasing_rate"]
            for key in dov_keys_with_keyword
            if "increasing_rate" in dovs[key]
        ]
        # Geometric means
        keywords_averages[keyword] = {
            "dod": gmean([dods[key]["dod"] for key in dod_keys_with_keyword]),
            "dov": gmean([dovs[key]["dov"] for key in dov_keys_with_keyword]),
            "dod_increasing_rate": sum(dod_increasing_rates) / len(dod_increasing_rates)
            if len(dod_increasing_rates) != 0
            else 0,
            "dov_increasing_rate": sum(dov_increasing_rates) / len(dov_increasing_rates)
            if len(dov_increasing_rates) != 0
            else 0,
        }
    with open("out", "w") as f:
        f.write(str(keywords_averages))
    print(keywords_averages)

    export_png(
        [
            (
                keyword,
                keywords_averages[keyword]["dov"],
                keywords_averages[keyword]["dov_increasing_rate"],
            )
            for keyword in keywords_averages.keys()
        ]
    )


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


from typing import Tuple, List
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def export_png(data: List[Tuple[str, int, int]], filename: str = "out.png"):
    # notre tableau de tableaux de 3 valeurs avec mot,dov ou dod et increasing rate
    # data = [
    #     ["label", 2, 5],
    #     ["énergie", 2, 7],
    #     ["valeur", 5, 7],
    #     ["coucou", 9, 3],
    #     ["valeur", 4, 5],
    #     ["coucou", -4, 16],
    #     ["coucou", -4, -5],
    # ]

    x = []
    y = []
    xy = []

    # on les ajoute dans un tableau de x et de y
    # on écrit les mots correspondant directement dans la boucle
    for j in data:
        # if j[1] != 0 and j[2] > 0:
        x.append(j[1])
        y.append(j[2])
        plt.text(j[1], j[2], j[0], fontsize=12)

    # on fusionne les en un tableau xy
    xy.append(x)
    xy.append(y)

    # print('#############"')
    # print(x)
    # print(y)
    # print('#############"')
    # print(xy)

    # librairie qui détermine les centres des clusters, il y en a 3 pour le noises, weak signals et strong signals
    xy = np.dstack((x, y))
    model = KMeans(3).fit(xy[0])

    # le nombre de couleurs en fonction du nombre de clusters
    colors = [i for i in model.labels_]

    # print('############# center"')
    # print(model.cluster_centers_)

    # on affiche les points avce les différentes couleurs des clusters
    plt.scatter(x, y, c=colors)

    plt.title("DoV")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    main()
