import matplotlib.pyplot
from pyvis.network import Network
import community
import networkx
import pandas
import numpy
import spacy
import datetime
import json
import os


base_path_novel = "./Novel Volumes"

df_persons = pandas.read_json("./Data/df_persons.json")


volumes = sorted(
    [volume.rstrip(".txt") for volume in os.listdir(base_path_novel)]
)


def filter_entity(ent_list: list, character_df: pandas.DataFrame):
    return [
        ent
        for ent in ent_list
        if ent in list(character_df["First Name"])
        or ent in list(character_df["Name"])
    ]


def create_graph_per_vol(volume: str) -> pandas.DataFrame:
    print(f"\n\n{volume}")
    with open(f"{base_path_novel}/{volume}.txt", encoding="utf-8") as f:
        novel_text = f.read()

    start = datetime.datetime.now()

    # Load English model language
    NLP = spacy.load("en_core_web_sm")

    # A void the problemns with bigest text
    NLP.max_length = len(novel_text)

    # Process text
    novel_doc = NLP(novel_text)

    end = datetime.datetime.now()
    print(f"Tempo de processando do texto: {end - start} ")

    list_sent_entity = []

    start = datetime.datetime.now()

    for sent in novel_doc.sents:
        if sent.ents:
            list_sent_entity.append(
                {"sentence": sent, "entities": [ent.text for ent in sent.ents]}
            )

    end = datetime.datetime.now()
    print(f"Tempo de validação das sentenças: {end - start} ")

    sent_entity_df = pandas.DataFrame(list_sent_entity)

    sent_entity_df["character_entities"] = sent_entity_df["entities"].apply(
        lambda x: filter_entity(x, df_persons)
    )

    sent_entity_df_filtered = sent_entity_df[
        sent_entity_df["character_entities"].map(len) > 0
    ].reset_index(drop=True)
    sent_entity_df_filtered["character_entities"] = sent_entity_df_filtered[
        "character_entities"
    ].apply(lambda x: [item.split()[0] for item in x])

    # Itentify relationship between the persons

    window_size = 5
    relation_ships = []
    for index in range(sent_entity_df_filtered.index[-1]):
        end_index = min(index + window_size, sent_entity_df_filtered.index[-1])
        char_list = sum(
            (sent_entity_df_filtered.loc[index:end_index].character_entities),
            [],
        )
        char_unique = [
            char_list[i]
            for i in range(len(char_list))
            if (i == 0) or char_list[i] != char_list[i - 1]
        ]
        if len(char_unique) > 1:
            for idx, char_a in enumerate(char_unique[:-1]):
                char_b = char_unique[idx + 1]
                relation_ships.append({"source": char_a, "target": char_b})

    df_relation_ships = pandas.DataFrame(relation_ships)
    # Sort relations_ship a -> b and b -> a
    df_relation_ships = pandas.DataFrame(
        numpy.sort(df_relation_ships.values, axis=1),
        columns=df_relation_ships.columns,
    )

    # Count Number of iterations of each relation_ship
    df_relation_ships["value"] = 1
    df_relation_ships = df_relation_ships.groupby(
        ["source", "target"], sort=False, as_index=False
    ).sum()

    # Create networkx fram pandas dataframe
    Gph = networkx.from_pandas_edgelist(
        df_relation_ships,
        source="source",
        target="target",
        edge_attr="value",
        create_using=networkx.Graph(),
    )

    # Start position usando kamada kawai layout
    pos = networkx.kamada_kawai_layout(Gph)
    networkx.draw(
        Gph,
        with_labels=True,
        node_color="skyblue",
        edge_cmap=matplotlib.cm.Blues,
        pos=pos,
    )

    # Create pyvis network
    net = Network(
        # notebook=True,
        width="1800px",
        height="920px",
        bgcolor="#121212",
        font_color="white",
    )

    # Node Attributes
    node_degree = dict(Gph.degree)
    degree_dict = networkx.degree_centrality(Gph)
    betweenness_dict = networkx.betweenness_centrality(Gph)
    closeness_dict = networkx.closeness_centrality(Gph)
    communities = community.best_partition(Gph)

    # Set Node Attributes
    networkx.set_node_attributes(Gph, node_degree, "size")
    networkx.set_node_attributes(Gph, degree_dict, "degree_centrality")
    networkx.set_node_attributes(
        Gph, betweenness_dict, "betweenness_centrality"
    )
    networkx.set_node_attributes(Gph, closeness_dict, "closeness_centrality")
    networkx.set_node_attributes(Gph, communities, "group")

    net.from_nx(Gph)
    net.save_graph(f"./Networks/{volume.replace(' ', '')}.html")
    print("\n\n\n\n")
    return df_relation_ships


list_df_relation_ships = []
new_df = pandas.DataFrame()
val = 1
for volume in volumes:
    df = create_graph_per_vol(volume)
    list_df_relation_ships.append(json.loads(df.to_json()))
    new_df.join(df)

    with open(f"./BKPs/list_df{val}.json", "w") as f:
        json.dump(list_df_relation_ships, f, indent=4)

    new_df.to_json(f"./BKPs/df_relation_bkp{val}.json")
    val += 1

with open("./BKPs/list_df.json", "w") as f:
    json.dump(list_df_relation_ships, f, indent=4)

new_df.to_json("./BKPs/df_relation_bkp.json")
