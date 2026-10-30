"""COMET environment: COMETKiwi + LID, with MetricX in a separate service."""
import argparse
import json
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from formula import clean_text, reward_result

ACCEPTED_LID_LABELS = {
    "Nepali": {"npi_Deva"}, "Urdu": {"urd_Arab"}, "Bengali": {"ben_Beng"},
    "Tamil": {"tam_Taml"}, "Khmer": {"khm_Khmr"}, "Burmese": {"mya_Mymr"},
    "Somali": {"som_Latn"}, "Swahili": {"swh_Latn", "swc_Latn"},
    "English": {"eng_Latn"}, "Chinese": {"zho_Hans", "zho_Hant"},
}


class RewardEngine:
    def __init__(self, config):
        import fasttext
        from comet import load_from_checkpoint
        self.config = config

        print(f"Loading LID model: {config['lid_model']}")
        self.lid = fasttext.load_model(config["lid_model"])
        print(f"LID labels: {len(self.lid.get_labels())}")

        print(f"Loading COMETKiwi: {config['comet_model']}")
        self.kiwi = load_from_checkpoint(config["comet_model"])

    def score_batch(self, rows):
        for row in rows:
            if not clean_text(row["source_text"]):
                raise ValueError("Missing source_text (data error)")
            if row["split"] not in ("train", "validation"):
                raise ValueError("split must be train or validation")
        nonempty = [i for i, row in enumerate(rows) if clean_text(row["translation"])]
        kiwi, errors = [0.0] * len(rows), [0.0] * len(rows)
        if nonempty:
            pairs = [{"src": clean_text(rows[i]["source_text"]),
                      "mt": clean_text(rows[i]["translation"])} for i in nonempty]
            scores = self.kiwi.predict(pairs, batch_size=self.config["comet_batch_size"],
                                       gpus=self.config["comet_gpus"], progress_bar=True).scores
            for i, score in zip(nonempty, scores, strict=True):
                kiwi[i] = float(score)
        if nonempty:
            payload = [{"source": clean_text(rows[i]["source_text"]),
                        "hypothesis": clean_text(rows[i]["translation"])} for i in nonempty]
            request = urllib.request.Request(self.config["metricx_url"].rstrip("/") + "/score_batch",
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=self.config["timeout"]) as response:
                scores = json.load(response)
            for i, score in zip(nonempty, scores, strict=True):
                errors[i] = float(score["error"])
        results = []
        for i, row in enumerate(rows):
            language = row["target_language"].split(" (", 1)[0].strip()
            if language not in ACCEPTED_LID_LABELS:
                raise ValueError(f"Unsupported target language: {language!r}")
            text = clean_text(row["translation"])
            label, confidence, correct = "", 0.0, False
            if text:
                labels, probabilities = self.lid.predict(text.replace("\n", " "), k=1)
                label, confidence = labels[0].removeprefix("__label__"), float(probabilities[0])
                correct = label in ACCEPTED_LID_LABELS[language]
            results.append(reward_result(row, kiwi[i], errors[i], label, confidence, correct,
                                         self.config["kiwi_weight"]))
        print(f"[QE] batch={len(rows)} train={sum(r['split']=='train' for r in rows)} "
              f"valid={sum(r['teacher_valid'] for r in results):.0f}", flush=True)
        return results


class RewardHTTPServer(ThreadingHTTPServer):
    request_queue_size = 128


class Handler(BaseHTTPRequestHandler):
    engine = None
    score_lock = Lock()

    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_json(200 if self.path == "/health" else 404,
                       {"status": "ok", "validation_metric": "cometkiwi"})

    def do_POST(self):
        if self.path != "/score_batch":
            self.send_json(404, {"detail": "not found"})
            return
        try:
            rows = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            # Accept HTTP requests concurrently, but serialize model access and MetricX calls.
            with self.score_lock:
                results = self.engine.score_batch(rows)
            self.send_json(200, results)
        except Exception as error:
            traceback.print_exc()
            self.send_json(500, {"detail": str(error)})


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "training/config.yaml"))
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["reward_service"]
    Handler.engine = RewardEngine(config)
    print(f"QE server: {config['host']}:{config['port']}", flush=True)
    RewardHTTPServer((config["host"], config["port"]), Handler).serve_forever()
