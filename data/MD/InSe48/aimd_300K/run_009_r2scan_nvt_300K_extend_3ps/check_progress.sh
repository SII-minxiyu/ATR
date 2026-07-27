#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f OSZICAR ]]; then
  echo "OSZICAR not found; job has not started."
  exit 0
fi

awk '/ T=/{step=$1; t=$3; gsub("[^0-9.-]","",t); n++; sum+=t; last=step; lastt=t; if(n==1||t<min)min=t; if(n==1||t>max)max=t}
END{
  if(n==0){print "No MD steps found."; exit}
  printf("steps=%d/3000 last_step=%s last_T=%.1f avg_T=%.2f min_T=%.1f max_T=%.1f\n", n, last, lastt, sum/n, min, max)
}' OSZICAR

if [[ -f OUTCAR ]]; then
  awk '/LOOP\+/{sum+=$7; n++} END{if(n>0) printf("avg_LOOP_real_sec_per_step=%.3f over %d steps\n", sum/n, n)}' OUTCAR
  grep -q 'General timing and accounting informations' OUTCAR && echo "status=finished" || echo "status=running_or_incomplete"
fi
