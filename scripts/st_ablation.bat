for %%x in (1 2 5 10 20 50 100 200 500) do (
  python ../src/main.py --core --tumble-off --f-st %%x --ecef --fixed-init --tsim 250 --tsample 250 --num-step 80000 --h-ref 0.007 --log-dir "star_tracker_ablation_ecef"
  timeout /t 5
  python ../src/main.py --core --tumble-off --f-st %%x --eci --fixed-init --tsim 250 --tsample 250 --num-step 80000 --h-ref 0.007 --log-dir "star_tracker_ablation_eci"
  timeout /t 5
)