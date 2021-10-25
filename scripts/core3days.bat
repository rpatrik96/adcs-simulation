python ../src/main.py --core --f-st 1 --ecef --fixed-init --tsim 250 --tsample 250 --num-step 1000000 --h-ref 0.007 --log-dir "core_10orbits_ecef"
timeout /t 5
python ../src/main.py --core --f-st 1 --eci --fixed-init --tsim 250 --tsample 250 --num-step 1000000 --h-ref 0.007 --log-dir "core_10orbits_eci"