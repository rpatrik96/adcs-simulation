for %%x in (0 0.1 1 5 10 50) do (
  timeout /t 2
  python ../src/star_metric.py --num-experiments 2000 --num-stars 10 --mag-noise-std %%x 
  timeout /t 2
)