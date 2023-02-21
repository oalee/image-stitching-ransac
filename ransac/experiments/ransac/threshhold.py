


from matplotlib import pyplot as plt
from ransac.models.ransac.stitch import ImageStitcher


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


    