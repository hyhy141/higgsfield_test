#!/usr/bin/env bash
# Proves restart persistence end-to-end against the docker compose stack:
# write a fact -> `docker compose down` -> `docker compose up` -> recall it.
set -euo pipefail

BASE="${MEMORY_BASE_URL:-http://localhost:8080}"
UID_="persist_$RANDOM"

wait_health() {
  for _ in $(seq 1 120); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "service never became healthy"; exit 1
}

echo "==> bring stack up"
docker compose up -d
wait_health

echo "==> write a fact for $UID_"
curl -sf -X POST "$BASE/turns" -H 'Content-Type: application/json' -d "{
  \"session_id\":\"persist_s\",\"user_id\":\"$UID_\",
  \"messages\":[{\"role\":\"user\",\"content\":\"I live in Reykjavik and own a horse named Skjoni.\"}],
  \"timestamp\":\"2025-05-01T09:00:00Z\",\"metadata\":{}}" >/dev/null

echo "==> docker compose down (named volume is preserved)"
docker compose down

echo "==> docker compose up again"
docker compose up -d
wait_health

echo "==> recall after restart"
OUT=$(curl -sf -X POST "$BASE/recall" -H 'Content-Type: application/json' -d "{
  \"query\":\"Where does the user live?\",\"session_id\":\"after\",\"user_id\":\"$UID_\",\"max_tokens\":256}")
echo "$OUT"

if echo "$OUT" | grep -q "Reykjavik"; then
  echo "PASS: data survived restart"
  curl -sf -X DELETE "$BASE/users/$UID_" >/dev/null || true
  exit 0
else
  echo "FAIL: data did not survive restart"; exit 1
fi
