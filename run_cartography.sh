EMOTIONS=("anger" "fear" "joy" "sadness" "surprise")

for EMOTION in "${EMOTIONS[@]}"; do
  echo "Running cartography for: ${EMOTION}"

  python plot_cartography.py \
    --jsonl_file "cart_output_dir/training_dynamics_epoch_${EMOTION}.jsonl" \
    --csv_file "public_data_dev/track_a/train/eng.csv" \
    --emotion "${EMOTION}"
done