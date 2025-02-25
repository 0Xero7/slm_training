while true; do
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$gpu_usage" -lt 5 ]; then  # If GPU usage less than 5%
        # Wait another minute to confirm it's really done
        sleep 60
        gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [ "$gpu_usage" -lt 5 ]; then
            runpodctl stop pod
            break
        fi
    fi
    sleep 30  # Check every 30 seconds
done