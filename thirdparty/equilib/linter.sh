echo "Running black"
black .
echo "Done"

echo "Running flake8"
flake8 .
echo "Done"
