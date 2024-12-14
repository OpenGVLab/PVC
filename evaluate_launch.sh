set -x

CHECKPOINT=${1}
TASK=${2}
ARGS=("$@")

if [ "${TASK}" == "vqa-chartqa-test" ]; then
    sh evaluate.sh ${CHECKPOINT} ${TASK} --dynamic --max-num 12 "${ARGS[@]:3}"
elif [ "${TASK}" == "vqa-infovqa-val" -o "${TASK}" == "vqa-infovqa-test" ]; then
    sh evaluate.sh ${CHECKPOINT} ${TASK} --dynamic --max-num 24 "${ARGS[@]:3}"
elif [ "${TASK}" == "vqa-docvqa-val" -o "${TASK}" == "vqa-docvqa-test" ]; then
    sh evaluate.sh ${CHECKPOINT} ${TASK} --dynamic --max-num 18 "${ARGS[@]:3}"
elif [ "${TASK}" == "mvbench" ]; then
    sh evaluate.sh ${CHECKPOINT} ${TASK} --num_segments 96 "${ARGS[@]:3}"
else
    sh evaluate.sh ${CHECKPOINT} ${TASK} --dynamic --max-num 6 "${ARGS[@]:3}"
fi
