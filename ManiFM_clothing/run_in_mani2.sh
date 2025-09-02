
for c_value in {0..17}
do
  python scripts/pred.py --t 5 --c $c_value

  python emsemble.py --t 0 --c $c_value
done

