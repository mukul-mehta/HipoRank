from hipo_rank.dataset_iterators.legaldocs import LegalDataset
from hipo_rank.embedders.bert import BertEmbedder

from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased

from hipo_rank.scorers.add import AddScorer
from hipo_rank.scorers.multiply import MultiplyScorer

from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge

from pathlib import Path
import json
import time
from tqdm import tqdm

DEBUG = False
ROUGE_ARGS = "-e /Users/mukul-mehta/CS/HipoRank/ROUGE-1.5.5/data -c 95 -n 2 -a -m"

"""
LegalDocs test set
"""

DATASETS = [
    (
        "legal",
        LegalDataset,
        {"file_path": "data/legal-dataset/legal.test.h5df"},
    ),
]
EMBEDDERS = [
    (
        "pacsum_bert",
        BertEmbedder,
        {
            "bert_config_path": "models/pacssum_models/bert_config.json",
            "bert_model_path": "models/pacssum_models/pytorch_model_finetuned.bin",
            "bert_tokenizer": "bert-base-uncased",
        },
    ),
]
SIMILARITIES = [
    ("cos", CosSimilarity, {}),
]
DIRECTIONS = [
    ("edge", EdgeBased, {}),
]

SCORERS = [
    ("add_f=0.0_b=1.0_s=1.0", AddScorer, {}),
]


Summarizer = DefaultSummarizer(num_words=100)

experiment_time = int(time.time())
# results_path = Path(f"results/{experiment_time}")
results_path = Path(f"results/legal_test_summary")

for embedder_id, embedder, embedder_args in EMBEDDERS:
    Embedder = embedder(**embedder_args)
    for dataset_id, dataset, dataset_args in DATASETS:
        DataSet = dataset(**dataset_args)
        exit()
        docs = list(DataSet)
        if DEBUG:
            docs = docs[:5]
        print(f"embedding dataset {dataset_id} with {embedder_id}")
        embeds = [Embedder.get_embeddings(doc) for doc in tqdm(docs)]
        for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print(f"calculating similarities with {similarity_id}")
            sims = [Similarity.get_similarities(e) for e in embeds]
            for direction_id, direction, direction_args in DIRECTIONS:
                print(f"updating directions with {direction_id}")
                Direction = direction(**direction_args)
                sims = [Direction.update_directions(s) for s in sims]
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    experiment = f"{dataset_id}-{embedder_id}-{similarity_id}-{direction_id}-{scorer_id}"
                    experiment_path = results_path / experiment
                    try:
                        experiment_path.mkdir(parents=True)

                        print("running experiment: ", experiment)
                        results = []
                        references = []
                        summaries = []
                        for sim, doc in zip(sims, docs):
                            scores = Scorer.get_scores(sim)
                            summary = Summarizer.get_summary(doc, scores)
                            results.append(
                                {
                                    "num_sects": len(doc.sections),
                                    "num_sents": sum(
                                        [len(s.sentences) for s in doc.sections]
                                    ),
                                    "summary": summary,
                                }
                            )
                            summaries.append([s[0] for s in summary])
                            references.append([doc.reference])
                        rouge_result = evaluate_rouge(
                            summaries, references, rouge_args=ROUGE_ARGS
                        )
                        (experiment_path / "rouge_results.json").write_text(
                            json.dumps(rouge_result, indent=2)
                        )
                        (experiment_path / "summaries.json").write_text(
                            json.dumps(results, indent=2)
                        )
                    except FileExistsError:
                        print(f"{experiment} already exists, skipping...")
                        pass
