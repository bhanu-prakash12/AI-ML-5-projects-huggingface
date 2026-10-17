"""Write tests/events/predict.json: a sample API Gateway HTTP v2 POST /predict event."""
import json
from pathlib import Path

from dropout_risk import features
from dropout_risk.data import RANGES

body = {c: float(RANGES[c][0]) for c in features.FEATURES}
event = {
    "version": "2.0", "routeKey": "POST /predict", "rawPath": "/predict", "rawQueryString": "",
    "headers": {"content-type": "application/json", "host": "example.lambda-url.local"},
    "requestContext": {
        "accountId": "000000000000", "apiId": "local", "domainName": "example.lambda-url.local",
        "http": {"method": "POST", "path": "/predict", "protocol": "HTTP/1.1",
                 "sourceIp": "127.0.0.1", "userAgent": "local-test"},
        "requestId": "local-1", "stage": "$default", "time": "11/Oct/2026:00:00:00 +0000",
        "timeEpoch": 1791676800000},
    "body": json.dumps(body), "isBase64Encoded": False,
}
out = Path(__file__).resolve().parents[1] / "tests" / "events" / "predict.json"
out.write_text(json.dumps(event, indent=2) + "\n")
print(f"wrote {out}")
