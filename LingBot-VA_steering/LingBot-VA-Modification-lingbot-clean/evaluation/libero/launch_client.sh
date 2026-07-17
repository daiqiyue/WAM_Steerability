START=0
END=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=../../scripts/lqr/slurm_port.sh
source "${SCRIPT_DIR}/../../scripts/lqr/slurm_port.sh"

CAMERA_ARGS=${CAMERA_ARGS:-}
PYTHONPATH=. python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port "${PORT}" \
    --test-num 20 \
    --task-range $START $END \
    --out-dir outputs/libero/task0_camera_init_pos_0.2 \
    --eef-delta 0.00 0.20 0.00 \
    #--eef-delta 0.00 0.30 0.00 \




# Text_distractor_2:
#  put both the alphabet soup and the tomato sauce in the basket. the cream cheese, ketchup, orange juice, milk, and butter are also on the table.
