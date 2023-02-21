

# def run(params):


#     images = [params.left, params.right]
#     stitcher = ImageStitcher(
#         images,
#         n_best_matches=params.n_best_matches,
#         random_sample_size=params.random_sample_size,
#         error_threshold=params.ransac_threshold,
#         num_iterations=params.ransac_iters,
#     )
#     stitcher.save_stitched_image(params.output)
#     stitcher.print_hyper_parameters()
#     stitcher.save_matches("grail")

   



# # arg parser for image stitching
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--left", type=str, help="path to left image", default="leftImage.png"
# )
# parser.add_argument(
#     "--right", type=str, help="path to right image", default="rightImage.png"
# )
# parser.add_argument(
#     "--output", type=str, help="path to output image", default="stitched.png"
# )
# parser.add_argument(
#     "--plot-matches", action="store_true", help="plot the matches", default=False
# )
# parser.add_argument(
#     "--sift-descriptors",
#     action="store_true",
#     help="plot the sift descriptors",
#     default=False,
# )
# parser.add_argument(
#     "--harr-descriptors",
#     action="store_true",
#     help="plot the harris descriptors",
#     default=False,
# )
# parser.add_argument(
#     "--n-best-matches", type=int, help="number of best matches", default=50
# )
# parser.add_argument(
#     "--random-sample-size",
#     type=int,
#     help="number of random sample size for ransac",
#     default=4,
# )
# parser.add_argument(
#     "--ransac-threshold", type=float, help="ransac threshold", default=0.01
# )
# parser.add_argument("--ransac-iters", type=int, help="ransac iterations", default=1000)

# parser = parser.parse_args()


# run(parser)

