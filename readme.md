
# Running
To run the script, first install the following python packages:
```
pip install opencv-contrib-python skimage matplotlib numpy scipy 
```
Then run the `stich.py` file with Python3 with the following arguments:
```
usage: stitch.py [-h] [--left LEFT] [--right RIGHT] [--output OUTPUT] [--plot-matches] [--sift-descriptors]
[--harr-descriptors] [--n-best-matches N_BEST_MATCHES]
[--random-sample-size RANDOM_SAMPLE_SIZE] [--ransac-threshold RANSAC_THRESHOLD]
[--ransac-iters RANSAC_ITERS]

options:
-h, --help            show this help message and exit
--left LEFT           path to left image
--right RIGHT         path to right image
--output OUTPUT       path to output image
--plot-matches        plot the matches
--sift-descriptors    plot the sift descriptors
--harr-descriptors    plot the harris descriptors
--n-best-matches N_BEST_MATCHES
number of best matches
--random-sample-size RANDOM_SAMPLE_SIZE
number of random sample size for ransac
--ransac-threshold RANSAC_THRESHOLD
ransac threshold
--ransac-iters RANSAC_ITERS
ransac iterations
```

For example, for running with the default hyperparameters run the following command:
```
python stitch.py --left leftImage.png --right rightImage.png --out stitched.png
```
Or select the hyperparameters such as the error threshold in ransac:
```
python stitch.py --left leftImage.png --right rightImage.png --out stitched.png --n-best-matches 200 --random-sample-size 5 --ransac-threshold 0.05 --ransac-iters 1000
```
