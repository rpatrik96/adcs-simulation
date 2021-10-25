for %%x in (42 654 85 7 33168 6496521 321165) do (
  python ../src/main.py --core --f-st 2 --seed %%x --ecef --fixed-init --tsim 250 --tsample 250 --num-step 30000 --h-ref 0.007 --log-dir "star_tracker_kf_cov_debug_2"
  timeout /t 5
  python ../src/main.py --core --f-st 20 --seed %%x --ecef --fixed-init --tsim 250 --tsample 250 --num-step 30000 --h-ref 0.007 --log-dir "star_tracker_kf_cov_debug_20"
  timeout /t 5
)