# check if Python virtual environment env exists
if [ ! -f "env/bin/activate" ]; then
    echo "Environment env does not exist. Please install it via ./install.sh"
    exit 1
fi

# activate Python virtual environment
source env/bin/activate

# create output folder
rm -rf models/
mkdir models

# train a model for part 1 (regression-based ANN) and evaluate it
echo "#########   PART 1: regression-based model ################"
python scripts/train.py --model-type regresion_nn --data data/train/ --save scripts/models/regression_model.pth
python scripts/P1/evaluate.py --model scripts/models/regression_model.pth --data data/test/
echo ""

# train a model for part 2 - basic (classification-based ANN) and evaluate it
echo "#########   PART 2 (basic): classification-based model ################"
python scripts/train.py --model-type classification_nn --data data/train/ --save scripts/models/classification_basic.pth
python scripts/P2_1/evaluate.py --model scripts/models/classification_basic.pth --data data/test/
echo ""

echo "#########   PART 2 (advanced): cost-sensitive classification-based model ################"
python scripts/train.py --model-type classification_nn_cost --data data/train/ --save scripts/models/classification_advanced.pth 
python scripts/P2_2/evaluate.py --model scripts/models/classification_advanced.pth --data data/test/
echo ""

echo "#########   PART 3 (advanced): random-forest model ################"
python scripts/train.py --model-type random_forest --data data/train/ --save scripts/models/part3_rf.pt 
# python scripts/evaluate.py --model models/part3_rf.pt --data data/test/
echo ""

# train a model for part 3 - extension 1 and evaluate it
echo "#########   PART 3 (extension 1): pairwise cost-sensitive classification model ################"
# YOUR CODE HERE: please add commands for each extension following the same template as above. 
# REMEMBER TO PRINT OUT THE DESCRIPTION (those "echo ..." lines) before every pair of train and evaluate command).


# deactivate Python virtual environment
deactivate
