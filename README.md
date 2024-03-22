# PixelDINO

Retrogressive Thaw Slumps are a permafrost disturbance comparable to landslides induced by permafrost thaw. Their detection and monitoring is important for understanding the dynamics of permafrost thaw and the vulnerability of permafrost across the Arctic. To do this efficiently with deep learning, large amounts of annotated data are needed, of which currently we do not have enough.

![https://khdlr.github.io/PixelDINO/map.png](Overview map of the training sites)

In order to address this without needing to manually digitize vast areas across the Arctic, we propose a semi-supervised learning approach which is able to combine existing labelled data with additional unlabelled data.

![https://khdlr.github.io/PixelDINO/pixeldino.png](Model Architecture)

This is done by asking the model to derive pseudo-classes, according to which it will segment the unlabelled images. For these pseudo-classes, consistency across data augmentations is enforced, which provides valuable training feedback to the model even for unlabelled tiles.

## Results

![https://khdlr.github.io/PixelDINO/herschel_semi.svg]()
![https://khdlr.github.io/PixelDINO/lena_semi.svg]()

## Paper

Currently under review.

Pre-print available at https://arxiv.org/abs/2401.09271
