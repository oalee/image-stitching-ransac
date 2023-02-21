
# Image Stitching with RANSAC

This is a python implementation of image stitching using RANSAC. The code is based on the following paper:

[1] David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, vol. 60, no. 2, pp. 91-110, 2004.


# Requirements
To run the script, you can either install the required packages located in `ransac/experiments/ransac/requirements.txt` and `ransac/models/ransac/requirements.txt` or use [Yerbmaté](github.com/oalee/yerbamate):

```
pip install yerbamate

```

Then, run the following command to install the experiment into your project:

```
mate install https://github.com/oalee/image-stitching-ransac/tree/main/ransac/experiments/ransac -yo pip
# or conda
mate install https://github.com/oalee/image-stitching-ransac/tree/main/ransac/experiments/ransac -yo conda

```

## Running the experiment
To run the experiment, you can use the following command:

```
python -m ransac.experiments.ransac.stich run left=leftImage.png right=rightImage.png 
```

## Environment variables
The following environment variables are used by the experiment:

```
{
    "results": "./results"
}
```
You can set the enviroment variables in shell or in a `env.json` file.



## Exported code modules
You can use Yerbamaté to directly install the code from this repository into your project. The following modules are exported:
|    | type        | name   | url                                                                                 | short_url                                              | dependencies                                                                                                                 |
|----|-------------|--------|-------------------------------------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| 0  | models      | ransac | https://github.com/oalee/image-stitching-ransac/tree/main/ransac/models/ransac      | oalee/image-stitching-ransac/ransac/models/ransac      | ['opencv_contrib_python~=4.7.0.68', 'numpy~=1.24.2', 'scikit_image~=0.19.3', 'matplotlib~=3.6.2', 'scipy~=1.9.1']            |
| 1  | experiments | ransac | https://github.com/oalee/image-stitching-ransac/tree/main/ransac/experiments/ransac | oalee/image-stitching-ransac/ransac/experiments/ransac | ['yerbamate~=0.9.21', 'matplotlib~=3.6.2', 'https://github.com/oalee/image-stitching-ransac/tree/main/ransac/models/ransac'] |

<!-- 

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
``` -->
