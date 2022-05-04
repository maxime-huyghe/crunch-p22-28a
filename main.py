import csv
from alive_progress import alive_bar


def main():
    # load the last 200 patents
    patents = get_patents(200)
    for patent in patents:
        print(patent)
        print("\n")


def get_patents(n_to_get):
    """
    Loads the last `n_to_get` patents from the filesystem.
    """
    with open("ibm_patents/patent_metadata.csv") as file:
        reader = csv.DictReader(file)
        patents = list(reader)
    patents.sort(key=lambda p: p["Date_Filed"], reverse=True)
    patents = patents[:n_to_get]
    # print(list(map(lambda p: p["Date_Filed"], metadata[:10])))
    with alive_bar(n_to_get) as progress_bar:
        for patent_metadata in patents:
            # replace .pdf with .txt
            patent_metadata["Name"].split(".")[0] + ".txt"
            with open("ibm_patents/patent_metadata.csv") as patent_file:
                patent_metadata["Keywords"] = truc_pyate(patent_file.read())
            progress_bar()
    return patents


if __name__ == "__main__":
    main()
