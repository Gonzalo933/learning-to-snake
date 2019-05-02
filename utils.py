import numpy as np


def preprocess_board(board, downsampling, binary_colors=False):
    ###### DEBUG SPACE
    # import matplotlib.pyplot as plt
    #
    # plt.title("ASD")
    # plt.imshow(board)
    # plt.show()
    # breakpoint()
    #####
    board = board[::downsampling, ::downsampling, :]  # downsample by factor of 2
    if binary_colors:
        food_color = 1.0
        snake_head_color = 1.0
        snake_torso_color = 1.0
        background_color = 0.0
    else:
        rgb = 255
        food_color = 1.0 * rgb
        snake_head_color = 0.5 * rgb
        snake_torso_color = 0.75 * rgb
        background_color = 0
    processed_board = np.copy(board)
    processed_board[board[:, :, 2] == 255] = food_color  # Food
    processed_board[board[:, :, 0] == 255] = snake_head_color  # Snake head
    processed_board[board[:, :, 0] == 1] = snake_torso_color  # Snake torso
    processed_board[board[:, :, 1] == 255] = background_color  # Background
    processed_board = processed_board[:, :, 0]
    return processed_board.astype(np.float).ravel()
