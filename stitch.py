import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import scipy.spatial.distance as dist


try:
    matplotlib.use("TkAgg")
except:
    pass


class ImageStitcher:
    def __init__(
        self,
        images,
        n_best_matches=100,
        random_sample_size=4,
        error_threshold=0.01,
        num_iterations=1000,
    ):

        self.n_best_matches = n_best_matches
        self.random_sample_size = random_sample_size
        self.error_threshold = error_threshold
        self.num_iterations = num_iterations

        self.images = [self.read_image(path) for path in images]
        self.gray_images = [img[0] for img in self.images]
        self.rgb_images = [
            cv2.normalize(img[2].astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
            for img in self.images
        ]
        self.sift = [self.SIFT(img[0]) for img in self.images]
        self.smoothing_window_size = 1

        self.matches = self.find_matches()
        (
            self.best_inliers,
            self.affine_transform,
            self.best_error,
            self.avg_residual,
        ) = self.ransac(self.matches)

    def visualize_best_inliers(self):
        self.plot_matches(self.best_inliers, "best_inliers.png")

    # harris corner detector
    def harris_corner_detector(
        self,
        sensitivity=0.04,
        threshold=0.01,
        window_size=3,
        min_distance=2,
        file_name="harris_corners.png",
    ):

        first_image = self.gray_images[0]
        second_image = self.gray_images[1]

        img_harris_0 = corner_harris(first_image, method="k", k=sensitivity)
        peaks_0 = corner_peaks(
            img_harris_0, min_distance=min_distance, threshold_rel=threshold
        )
        subpix_0 = corner_subpix(first_image, peaks_0, window_size=window_size)

        # same thing for second_image
        img_harris_1 = corner_harris(second_image, method="k", k=sensitivity)
        peaks_1 = corner_peaks(
            img_harris_1, min_distance=min_distance, threshold_rel=threshold
        )
        subpix_1 = corner_subpix(second_image, peaks_1, window_size=window_size)


        # plot the harris corners, on both images
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(first_image, cmap="gray")
        ax[0].plot(peaks_0[:, 1], peaks_0[:, 0], "r+")
        ax[0].plot(subpix_0[:, 1], subpix_0[:, 0], "g+")
        ax[0].set_title("Left Image")

        ax[1].imshow(second_image, cmap="gray")
        ax[1].plot(peaks_1[:, 1], peaks_1[:, 0], "r+")
        ax[1].plot(subpix_1[:, 1], subpix_1[:, 0], "g+")
        ax[1].set_title("Right Image")

        plt.savefig(file_name)

        # print the hyper parameters in markdown table format:
        print(
            f"| Hyperparameter | Value |\n| --- | --- |\n| Sensitivity | {sensitivity} |\n| Threshold | {threshold} |\n| Window Size | {window_size} |\n| Min Distance | {min_distance} |"
        )



    def save_image_opencv(self, image, filename="image.png"):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def plot_sift_descriptors(self):

        key_desc = []
        for i in range(len(self.images)):
            points_draw = cv2.drawKeypoints(
                self.gray_images[i],
                self.sift[i][0],
                self.images[i][2].copy(),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            key_desc.append(points_draw)

        # concatenate all images
        img_concat = np.concatenate(key_desc, axis=1)
        cv2.imwrite("images/sift_descriptors.png", img_concat)

    def find_matches(self):
        max_matches = self.n_best_matches
        matches = []

        distance = dist.cdist(self.sift[0][1], self.sift[1][1], "euclidean")
        distance_ = dist.cdist(self.sift[0][1], self.sift[1][1], "correlation")

        # the sift descriptors are already normalized
        distance = dist.cdist(self.sift[0][1], self.sift[1][1], "euclidean")

        self.euclidean_distance = np.sum(distance)
        self.normalized_correlation_distance = np.sum(distance_)

        # find the minimum distance between descriptors
        for i in range(len(distance)):
            minIndx = np.argmin(distance[i])  # index of the minimum distance
            matches.append(
                (i, minIndx, distance[i][minIndx])
            )  # (index of the first image, index of the second image, distance)

        # sort the matches by distance
        matches = sorted(matches, key=lambda x: x[2])

        # take the first n_best_matches
        matches = matches[:max_matches]

        # map indices to sift descriptor points
        matches = np.array(
            list(
                map(
                    lambda x: list(self.sift[1][0][x[1]].pt + self.sift[0][0][x[0]].pt),
                    matches,
                )
            )
        )

        return matches

    def plot_matches(self, matches, filename="matches.png"):
        total_img = np.concatenate(self.rgb_images, axis=1)

        offset = total_img.shape[1] / 2
        _, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.imshow(total_img)

        # matches are reversed for finding homography since homography is used to transform right image to left
        reverse_matches = matches[:, [2, 1, 0, 3]]
        x1 = reverse_matches[:, 0]
        y1 = reverse_matches[:, 1]
        x2 = reverse_matches[:, 2]
        y2 = reverse_matches[:, 3]
        ax.plot(x1, y1, "ro", markersize=5)
        ax.plot(x2 + offset, y2, "ro", markersize=5)
        ax.plot([x1, x2 + offset], [y1, y2], "b-")
        plt.savefig(filename, bbox_inches="tight")

    # returns sift keypoints and descriptors of the image
    def SIFT(self, img):
        siftDetector = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = siftDetector.detectAndCompute(img, None)
        # normalize sift descriptors, minmax scaling
        descriptors = cv2.normalize(descriptors, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return keypoints, descriptors

    # visualization of sift keypoints
    def sift_keypoint_(self, gray, rgb, kp):
        img = cv2.drawKeypoints(
            gray, kp, rgb.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return img

    # Read image and returns a touple of gray image, img, and rgb image
    def read_image(self, path):
        img = cv2.imread(path)

        # remove borders of image by 10 pixels
        img = img[10 : img.shape[0] - 10, 10 : img.shape[1] - 10]

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img_gray, img, img_rgb

    def homography(self, matches):
        # find homography
        src_pts = matches[:, :2]  # src_pts is the first image
        dst_pts = matches[:, 2:]  # dst_pts is the second image

        # add 1 to the last column of src_pts
        src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))

        # use least squares to find transformation matrix
        H = np.linalg.lstsq(src_pts, dst_pts)[0]

        H = H.T
        # append [0, 0, 1] to the end of H
        H = np.vstack((H, [0, 0, 1]))

        return H

    # Find the error when using H to transform the points, H is a matrix of 3x3 and the points are 2x2
    def get_erros(self, H, points, normalize=True):

        src_pts = points[:, :2]
        dst_pts = points[:, 2:]

        # add axis to the points
        all_pts = np.concatenate((src_pts, np.ones((src_pts.shape[0], 1))), axis=1)

        transformed_src = np.dot(H, all_pts.T)

        # estimated error of the points
        error = np.linalg.norm((transformed_src.T[:, :2]) - dst_pts, axis=1) ** 2

        # nomalize the error with min and max
        if normalize:
            error = (error - np.min(error)) / (np.max(error) - np.min(error))

        return error

    # RANSAC algorithm to find the best homography
    def ransac(self, matches):

        threshold = self.error_threshold  # threshold for the error
        num_random_points = self.random_sample_size  # number of random points to sample
        iters = self.num_iterations  # number of iterations

        # At first we dont have any inliers
        num_best_inliers = 0
        best_error = np.inf

        # for each iteration of the RANSAC algorithm
        for i in range(iters):
            # random sample points
            permutated = np.random.permutation(matches)
            random_points = permutated[
                :num_random_points
            ]  # random_points is a list of random points
            rest = permutated[num_random_points:]  # rest is a list of rest points

            # find the transformation matrix
            H = self.homography(random_points)
            errors = self.get_erros(H, rest)  # errors is a list of errors
            all_error = np.sum(
                self.get_erros(H, matches)
            )  # all_error is the sum of all errors
            # find the inliers with the threshold
            inliers = rest[np.where(errors < threshold)[0]]

            # update the best inliers
            num_inliers = len(inliers)
            if all_error < best_error:
                best_error = all_error
                best_inliers = inliers.copy()  # save the best inliers
                num_best_inliers = num_inliers  # update the number of best inliers
                best_error_mean = np.mean(
                    self.get_erros(H, matches, normalize=False)
                )  # best error is the mean of the errors of all the matches
                best_H = H.copy()  # save the best H
                avg_residual_inliers = np.average(
                    self.get_erros(H, inliers, True)
                )  # average of the residual inliers

        return (
            best_inliers,
            best_H,
            best_error_mean,
            avg_residual_inliers,
        )  # return the best inliers, best H, and best error

    # stitch the images with the affine transformation
    def stitch(self, affineTransformation):

        # use the RGB images to stich
        # rbg_images[0] is the left image
        # rbg_images[1] is the right image
        imageLeft = self.rgb_images[0]
        imageRight = self.rgb_images[1]
        new_shape = (
            imageLeft.shape[1] + imageRight.shape[1],
            imageLeft.shape[0],
        )  # new shape of the image
        stitched_image = cv2.warpPerspective(
            imageRight, affineTransformation, new_shape
        )  # warp the image
        stitched_image[
            : imageLeft.shape[0], : imageLeft.shape[1]
        ] = imageLeft  # put the left image in the stitched image

        # Crop black part of the stitched image
        # from a point, if its black, it will be cropped
        # from the point, if its not black, it will be kept
        # and the rest will be cropped
        stitched_image = self.crop_black(stitched_image)

        return stitched_image

    # crop the black part of the image
    def crop_black(self, image):
        # find the top and bottom of the image
        top = 0
        bottom = image.shape[0]
        for i in range(image.shape[0]):
            if np.sum(image[i, :]) != 0:
                top = i
                break
        for i in range(image.shape[0] - 1, 0, -1):
            if np.sum(image[i, :]) != 0:
                bottom = i
                break
        # find the left and right of the image
        left = 0
        right = image.shape[1]
        for i in range(image.shape[1]):
            if np.sum(image[:, i]) != 0:
                left = i
                break
        for i in range(image.shape[1] - 1, 0, -1):
            if np.sum(image[:, i]) != 0:
                right = i
                break
        # crop the image
        image = image[top:bottom, left:right]
        return image

    def save_stitched_image(self, path):
        stitched_image = self.stitch(self.affine_transform)

        # denormalize the image
        stitched_image = (stitched_image * 255).astype(np.uint8)

        # RGB 2 BGR
        stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)

        # save the stitched image
        cv2.imwrite(path, stitched_image)

    # prints hyperparameters and error in markdown table format
    def print_hyper_parameters(self):
        print("| Parameter | Value |")
        print("| --- | --- |")
        print("| Random Sample Size | {} |".format(self.random_sample_size))
        print("| Num Iterations | {} |".format(self.num_iterations))
        print("| Error Threshold | {} |".format(self.error_threshold))
        print("| Number of Selected Best Matches | {} |".format(self.n_best_matches))
        print("| Number of Inliers | {} |".format(len(self.best_inliers)))
        print(
            "| Number of Outliers | {} |".format(
                self.n_best_matches - len(self.best_inliers)
            )
        )
        print(
            "| Accuracy | {} |".format(round(self.best_error, 6))
        )  # round to 2 decimal places
        print("| Average Residual Inliers | {} |".format(round(self.avg_residual, 6)))

    def save_matches(self, id):
        # save the matches
        self.plot_matches(self.matches, f"images/all_matches_{id}.png")
        self.plot_matches(self.best_inliers, f"images/best_inliers_{id}.png")


def run(params):


    images = [params.left, params.right]
    stitcher = ImageStitcher(
        images,
        n_best_matches=params.n_best_matches,
        random_sample_size=params.random_sample_size,
        error_threshold=params.ransac_threshold,
        num_iterations=params.ransac_iters,
    )
    stitcher.save_stitched_image(params.output)
    stitcher.print_hyper_parameters()
    stitcher.save_matches("grail")

   


def run_experiment_1(params):
    n_best_matches = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    accuracy = []

    for n in n_best_matches:
        params.n_best_matches = n
        images = [params.left, params.right]
        stitcher = ImageStitcher(
            images,
            n_best_matches=params.n_best_matches,
            random_sample_size=params.random_sample_size,
            error_threshold=params.ransac_threshold,
            num_iterations=params.ransac_iters,
        )
        accuracy.append(stitcher.best_error)
    
    plt.plot(n_best_matches, accuracy)
    plt.xlabel("Number of Best Matches")
    plt.ylabel("Accuracy (Less is better)")
    plt.title("Accuracy vs Number of Best Matches")
    plt.savefig("images/accuracy_vs_n_best_matches.png")


def run_experiment_on_threshold(params):

    thresholds = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    accuracy = []

    for t in thresholds:
        params.ransac_threshold = t
        images = [params.left, params.right]
        stitcher = ImageStitcher(
            images,
            n_best_matches=params.n_best_matches,
            random_sample_size=params.random_sample_size,
            error_threshold=params.ransac_threshold,
            num_iterations=params.ransac_iters,
        )
        accuracy.append(stitcher.best_error)
    
    # plot dots for each threshold and accuracy
    plt.plot(thresholds, accuracy)
    plt.xlabel("Threshold")
    # show all the points on x axis
    # plt.xticks(thresholds)
    plt.ylabel("Accuracy (Less is better)")
    plt.title("Accuracy vs Threshold")
    plt.savefig("images/accuracy_vs_threshold_2.png")


    

# arg parser for image stitching
parser = argparse.ArgumentParser()
parser.add_argument(
    "--left", type=str, help="path to left image", default="leftImage.png"
)
parser.add_argument(
    "--right", type=str, help="path to right image", default="rightImage.png"
)
parser.add_argument(
    "--output", type=str, help="path to output image", default="stitched.png"
)
parser.add_argument(
    "--plot-matches", action="store_true", help="plot the matches", default=False
)
parser.add_argument(
    "--sift-descriptors",
    action="store_true",
    help="plot the sift descriptors",
    default=False,
)
parser.add_argument(
    "--harr-descriptors",
    action="store_true",
    help="plot the harris descriptors",
    default=False,
)
parser.add_argument(
    "--n-best-matches", type=int, help="number of best matches", default=50
)
parser.add_argument(
    "--random-sample-size",
    type=int,
    help="number of random sample size for ransac",
    default=4,
)
parser.add_argument(
    "--ransac-threshold", type=float, help="ransac threshold", default=0.01
)
parser.add_argument("--ransac-iters", type=int, help="ransac iterations", default=1000)

parser = parser.parse_args()


run(parser)

