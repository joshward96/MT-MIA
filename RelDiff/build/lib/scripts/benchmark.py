import sys
import logging
import argparse

from xgboost import XGBClassifier

from syntherela.benchmark import Benchmark
from syntherela.metrics.single_column.detection import SingleColumnDetection
from syntherela.metrics.single_table.detection import SingleTableDetection
from syntherela.metrics.multi_table.detection import AggregationDetection
from syntherela.metrics.multi_table.statistical import CardinalityShapeSimilarity

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="rossmann_subsampled")
args.add_argument("--real-data-dir", type=str, default="data/original")
args.add_argument("--synthetic-data-dir", type=str, default="data/synthetic")
args.add_argument("--run-id", type=str, default="1")

args = args.parse_args()
dataset_name = args.dataset_name
run_id = args.run_id

logger = logging.getLogger(f"{dataset_name}_logger")

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f"START LOGGING Dataset: {dataset_name}")

xgb_cls = XGBClassifier
xgb_args = {"seed": 0}

single_column_metrics = [
    SingleColumnDetection(
        classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
    ),
]
single_table_metrics = [
    SingleTableDetection(
        classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
    ),
]
multi_table_metrics = [
    CardinalityShapeSimilarity(),
    AggregationDetection(
        classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
    ),
]

benchmark = Benchmark(
    real_data_dir=args.real_data_dir,
    synthetic_data_dir=args.synthetic_data_dir,
    results_dir=f"results/{run_id}",
    benchmark_name="Benchmark",
    single_column_metrics=single_column_metrics,
    single_table_metrics=single_table_metrics,
    multi_table_metrics=multi_table_metrics,
    run_id=run_id,
    sample_id="sample1",
    datasets=[dataset_name],
    methods=["RelDiff_gen", "RelDiff"],
)

benchmark.run()
