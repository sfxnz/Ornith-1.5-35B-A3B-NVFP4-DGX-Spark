#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-ornith-1.5-35b-nvfp4-official}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME is not present"
  exit 0
fi

docker stop "$NAME"
docker rm "$NAME"
echo "removed $NAME"
