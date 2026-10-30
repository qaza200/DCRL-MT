import argparse
import json
import sys
from time import perf_counter
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer


class MetricXScorer:
    def __init__(self, args):
        sys.path.insert(0, args.metricx_repo)

        import torch
        from metricx24 import models
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = models.MT5ForRegression.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            local_files_only=True,
        ).to(self.device).eval()

    def score(self, rows):
        started = perf_counter()
        num_batches = (len(rows) + self.batch_size - 1) // self.batch_size
        print(f"[MetricX] request_rows={len(rows)}, batches={num_batches}, "
              f"batch_size={self.batch_size}, device={self.device}", flush=True)
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(rows), self.batch_size):
                features = []
                for row in rows[start : start + self.batch_size]:
                    text = (
                        f"source: {row['source']} "
                        f"candidate: {row['hypothesis']}"
                    )
                    item = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                    )
                    item["input_ids"] = item["input_ids"][:-1]
                    item["attention_mask"] = item["attention_mask"][:-1]
                    features.append(item)

                batch = self.tokenizer.pad(
                    features,
                    padding=True,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                predictions = self.model(**batch).predictions.cpu()
                scores.extend(float(value) for value in predictions)
                print(f"[MetricX] batch={start // self.batch_size + 1}/{num_batches}, "
                      f"scored={len(scores)}/{len(rows)}, elapsed={perf_counter() - started:.2f}s",
                      flush=True)
        print(f"[MetricX] done, scored={len(scores)}, elapsed={perf_counter() - started:.2f}s",
              flush=True)
        return scores


class Handler(BaseHTTPRequestHandler):
    scorer = None

    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "mode": "reference-free-qe"})
        else:
            self.send_json(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/score_batch":
            self.send_json(404, {"detail": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            rows = json.loads(self.rfile.read(size).decode("utf-8"))
            scores = self.scorer.score(rows)
            self.send_json(200, [{"error": score} for score in scores])
        except Exception as error:
            self.send_json(500, {"detail": str(error)})


def parse_args():
    import yaml
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(Path(__file__).parents[1] / "training/config.yaml"))
    config, remaining = config_parser.parse_known_args()
    with open(config.config, encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle)["reward_service"]["metricx"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./models/metricx-24-hybrid-xl-v2p6-bfloat16",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=(
            "./models/"
            "metricx-24-hybrid-xl-v2p6-bfloat16/mt5-xl-tokenizer"
        ),
    )
    parser.add_argument(
        "--metricx_repo",
        default="./third_party/metricx",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8002)
    parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


if __name__ == "__main__":
    args = parse_args()
    Handler.scorer = MetricXScorer(args)
    print(f"MetricX server: http://{args.host}:{args.port}")
    HTTPServer((args.host, args.port), Handler).serve_forever()
