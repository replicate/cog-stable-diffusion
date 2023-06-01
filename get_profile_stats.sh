# !/bin/bash

# Check if the number of times argument is provided
if [ $# -eq 0 ]; then
  echo "Please provide the number of times to run the command as an argument."
  exit 1
fi

# Read the number of times to run the command from the command-line argument
n=$1

# Run the command n times
for ((i=1; i<=n; i++)); do
  echo "Running command $i"
  cog predict -i prompt="monkey scuba diving"
done

cog run python profile_stats.py