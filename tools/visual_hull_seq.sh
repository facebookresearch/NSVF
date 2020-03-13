for i in {0050..0199}; do
  python tools/visual_hull.py $i
  echo "$i done"
done