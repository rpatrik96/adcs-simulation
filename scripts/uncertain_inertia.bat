python ../src/main.py --core --f-st 1 --ecef --fixed-init --tsim 250 --tsample 250 --num-step 1500 --h-ref 0.007 --omega-norm 50 --uncertain  --nominal-only --num-runs 30 --log-dir "uncertain_ecef"
python ../src/main.py --core --f-st 1 --eci --fixed-init --tsim 250 --tsample 250 --num-step 1500 --h-ref 0.007 --omega-norm 50 --uncertain  --nominal-only --num-runs 30 --log-dir "uncertain_eci"
