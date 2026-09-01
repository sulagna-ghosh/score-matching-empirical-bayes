#!/bin/bash

# Define variables inside the Bash script
EXPERIMENT_C="c"
EXPERIMENT_D="d"
EXPERIMENT_E="e"
EXPERIMENT_F="f"
EXPERIMENT_G="g"
EXPERIMENT_H="h"
EXPERIMENT_I="i"
EXPERIMENT_J="j"

FIRST_FIFTY_START=0
FIRST_FIFTY_END=49
SECOND_FIFTY_START=50
SECOND_FIFTY_END=99

THIRD_FIFTY_START=100
THIRD_FIFTY_END=149
FOURTH_FIFTY_START=150
FOURTH_FIFTY_END=199

FIFTH_FIFTY_START=200
FIFTH_FIFTY_END=249
SIXTH_FIFTY_START=250
SIXTH_FIFTY_END=299

SEVENTH_FIFTY_START=300
SEVENTH_FIFTY_END=349
EIGHTH_FIFTY_START=350
EIGHTH_FIFTY_END=399

NINETH_FIFTY_START=400
NINETH_FIFTY_END=449
TENTH_FIFTY_START=450
TENTH_FIFTY_END=499

# Initialize conda for the current subshell
eval "$(conda shell.bash hook)"

# Activate your specific environment
conda activate nn_env

# Run the Python script and pass the variables using the named arguments
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_C" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"

# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_D" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"

python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_E" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"

# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_F" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"

# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_G" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"

# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_H" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"

# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_I" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"

# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$FIRST_FIFTY_START" --m_sim_end "$FIRST_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$SECOND_FIFTY_START" --m_sim_end "$SECOND_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$THIRD_FIFTY_START" --m_sim_end "$THIRD_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$FOURTH_FIFTY_START" --m_sim_end "$FOURTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$FIFTH_FIFTY_START" --m_sim_end "$FIFTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$SIXTH_FIFTY_START" --m_sim_end "$SIXTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$SEVENTH_FIFTY_START" --m_sim_end "$SEVENTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$EIGHTH_FIFTY_START" --m_sim_end "$EIGHTH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$NINETH_FIFTY_START" --m_sim_end "$NINETH_FIFTY_END"
# python3 submitit_heteroscedastic.py --experiment "$EXPERIMENT_J" --m_sim_start "$TENTH_FIFTY_START" --m_sim_end "$TENTH_FIFTY_END"