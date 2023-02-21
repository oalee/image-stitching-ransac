


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

