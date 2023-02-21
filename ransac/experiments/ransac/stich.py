import os
from ...models.ransac.stitch import ImageStitcher

import yerbamate

env = yerbamate.Environment()

sticher = ImageStitcher(
    [env.left, env.right],
    100,
    4,
    0.01,
    1000,
)

stiched_image_path = os.path.join(env["results"], "stitched.png")
sticher.save_stitched_image(stiched_image_path)
sticher.print_hyper_parameters()
